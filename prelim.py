"""Train-only short-horizon selection and reproducible preliminary submission.

No test labels are used to choose models, clipping bounds, or ensemble weights.
At origin t, input observations through t are available, never observations >t.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from pipeline import IndustrialDataPipeline

TARGETS = ('generator_1', 'generator_all')
HORIZONS = tuple(range(1, 9))
UPPER = {'generator_1': 200., 'generator_all': 440.,
         'blast_furnace_gas_holder_1': 200000., 'blast_furnace_gas_holder_2': 300000.}


def read_data(directory):
    loader = IndustrialDataPipeline()
    loader.load(directory)
    return loader.observations.copy()


def future(index, h):
    ix = pd.DatetimeIndex(index)
    if hasattr(ix, 'as_unit'):
        ix = ix.as_unit('ns')
    return pd.DatetimeIndex(ix.asi8+900_000_000_000*int(h), name=ix.name)


def score(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ok = np.isfinite(y) & (np.abs(y) > 1e-8)
    if not np.isfinite(p[ok]).all():
        raise ValueError('Non-finite predictions on observed labels')
    return float(np.mean(np.abs(y[ok]-p[ok])/np.abs(y[ok]))) if ok.any() else None


class InputTransform:
    """Training-fitted schema and conservative sensor limits; preserve raw labels.

    All-empty sensors are unavailable, not fabricated zero measurements. Constant
    engineered fields are removed using training history only. Negative gas flow
    and readings beyond instrument/plant bounds are invalid inputs, not labels.
    """
    def fit(self, raw):
        self.columns = [c for c in raw if raw[c].notna().any() and raw[c].nunique() > 1]
        self.excluded = {c: ('all_missing' if raw[c].notna().sum() == 0 else 'constant')
                         for c in raw if c not in self.columns}
        self.bounds = {}
        for c in self.columns:
            s = raw[c].replace([np.inf, -np.inf], np.nan).dropna()
            q1, q3 = s.quantile([.25, .75])
            # Guard only gross sensor errors; intermittent zero-flow is valid.
            upper = UPPER.get(c, max(float(s.quantile(.999)), float(q3+6*(q3-q1)), 1.))
            self.bounds[c] = (0., upper)
        frame = self._build(raw)
        self.feature_columns = frame.columns[frame.nunique() > 1].tolist()
        return self

    def _build(self, raw):
        clean = pd.DataFrame(index=raw.index)
        for c in self.columns:
            lo, hi = self.bounds[c]
            s = raw[c].replace([np.inf, -np.inf], np.nan)
            s = s.mask((s < lo) | (s > hi))
            clean[c] = s.ffill().fillna(0.)
        d = {c: clean[c] for c in clean}
        idx = raw.index
        for t in TARGETS:
            s = clean[t]
            for n in (1, 2, 3, 4, 6, 8, 12, 24, 48, 96):
                d[f'feat_{t}_lag_{n}'] = s.shift(n).fillna(s)
            for n in (2, 4, 8, 16, 32, 96):
                d[f'feat_{t}_ema_{n}'] = s.ewm(span=n, adjust=False).mean()
                d[f'feat_{t}_mean_{n}'] = s.rolling(n, min_periods=1).mean()
            for n in (4, 16, 96):
                d[f'feat_{t}_std_{n}'] = s.rolling(n, min_periods=2).std().fillna(0.)
            for n in (1, 4, 8):
                d[f'feat_{t}_change_{n}'] = s.diff(n).fillna(0.)
        for c in clean:
            if c in TARGETS:
                continue
            for n in (4, 16):
                d[f'feat_{c}_mean_{n}'] = clean[c].rolling(n, min_periods=1).mean()
            d[f'feat_{c}_change_4'] = clean[c].diff(4).fillna(0.)
        d['feat_other_power'] = clean.generator_all-clean.generator_1
        for k in (1, 2, 3):
            phase = 2*np.pi*k*(idx.hour*60+idx.minute)/1440
            d[f'feat_clock_sin_{k}'] = np.sin(phase)
            d[f'feat_clock_cos_{k}'] = np.cos(phase)
        # Common features are exactly those exported in input.csv.
        return pd.DataFrame(d, index=raw.index).astype('float32')

    def transform(self, raw):
        return self._build(raw)[self.feature_columns]


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str
    days: int = 30
    trees: int = 240
    alpha: float = 30.
    half_life: float = 0.


SPECS = (
    Spec('persistence', 'last'), Spec('ema4', 'ema', days=4),
    Spec('ema8', 'ema', days=8),
    Spec('ridge7', 'ridge', days=7, alpha=30.),
    Spec('ridge21', 'ridge', days=21, alpha=100.),
    Spec('ridge60', 'ridge', days=60, alpha=300., half_life=14.),
    Spec('lgb14', 'lgb', days=14), Spec('lgb60', 'lgb', days=60, half_life=14.),
    Spec('cat21', 'cat', days=21),
)


class ShortModel:
    def __init__(self, spec, threads=4):
        self.spec, self.threads = spec, threads
        self.models, self.scalers, self.audit = {}, {}, []

    def fit(self, raw, x, cutoff):
        self.cutoff = pd.Timestamp(cutoff)
        cfg = self.spec
        if cfg.kind in ('last', 'ema'):
            return self
        # A stable, low-dimensional ARX for linear members. Gas-use and holder
        # signals complement load history without hundreds of collinear sensors.
        if cfg.kind == 'ridge':
            self.columns = [c for c in x if c in TARGETS or
                c.startswith(('feat_generator_1_', 'feat_generator_all_', 'feat_clock_',
                              'feat_generator_use_', 'feat_blast_furnace_gas_holder_')) or
                c.startswith('generator_use_') or c == 'feat_other_power']
        else:
            self.columns = list(x.columns)
        for t in TARGETS:
            for h in HORIZONS:
                end = future([self.cutoff], -h)[0]
                start = future([self.cutoff], -96*cfg.days)[0]
                rows = x.index[(x.index > start) & (x.index <= end)]
                y = raw[t].reindex(future(rows, h)).to_numpy()
                valid = np.isfinite(y) & (y > 0.) & (y <= UPPER[t]) & raw.loc[rows, t].notna().to_numpy()
                rows, y = rows[valid], y[valid]
                if len(rows) < 96:
                    raise ValueError('Insufficient observed training labels')
                z = x.loc[rows, self.columns].to_numpy(dtype=float)
                baseline = x.loc[rows, t].to_numpy()
                residual = y-baseline
                w = 1/np.maximum(y, 1.)
                if cfg.half_life > 0:
                    age = np.asarray((self.cutoff-rows).total_seconds())/86400
                    w *= np.exp2(-age/cfg.half_life)
                w /= w.mean()
                if cfg.kind == 'ridge':
                    scaler = StandardScaler().fit(z, sample_weight=w)
                    z = scaler.transform(z)
                    model = Ridge(alpha=cfg.alpha)
                    self.scalers[t, h] = scaler
                elif cfg.kind == 'lgb':
                    model = LGBMRegressor(objective='regression_l1', n_estimators=cfg.trees,
                        num_leaves=15, learning_rate=.035, min_child_samples=40,
                        reg_lambda=8., colsample_bytree=.85, random_state=2026,
                        n_jobs=self.threads, verbosity=-1, deterministic=True, force_col_wise=True)
                elif cfg.kind == 'cat':
                    model = CatBoostRegressor(loss_function='MAE', iterations=cfg.trees,
                        depth=5, learning_rate=.045, l2_leaf_reg=8., thread_count=self.threads,
                        random_seed=2026, verbose=False, allow_writing_files=False)
                else:
                    raise ValueError(cfg.kind)
                model.fit(z, residual, sample_weight=w)
                self.models[t, h] = model
                self.audit.append(dict(member=cfg.name, target=t, horizon=h, rows=len(rows),
                    last_origin=str(rows[-1]), last_label=str(future(rows[-1:], h)[0]), cutoff=str(self.cutoff)))
        return self

    def predict(self, raw, x, origins):
        if min(origins) < self.cutoff:
            raise ValueError('Forecast origins precede training cutoff')
        result = {}
        cfg = self.spec
        for t in TARGETS:
            now = x.loc[origins, t].to_numpy()
            for h in HORIZONS:
                if cfg.kind == 'last':
                    p = now
                elif cfg.kind == 'ema':
                    p = x.loc[origins, f'feat_{t}_ema_{cfg.days}'].to_numpy()
                else:
                    z = x.loc[origins, self.columns].to_numpy(dtype=float)
                    if cfg.kind == 'ridge':
                        z = self.scalers[t, h].transform(z)
                    p = now+self.models[t, h].predict(z)
                result[t, h] = np.clip(p, 0, UPPER[t])
        return result


def convex_weights(y, p):
    """Exact weighted MAE blend, sparse LP (no quadratic-memory identity)."""
    valid = np.isfinite(y) & (np.abs(y) > 1e-8) & np.isfinite(p).all(axis=1)
    y, p = y[valid], p[valid]
    n, k = p.shape
    if n == 0:
        raise ValueError('No labels for weight selection')
    a = sparse.csr_matrix(p/np.abs(y[:, None]))
    eye = sparse.eye(n, format='csr')
    constraints = sparse.vstack([sparse.hstack([a, -eye]), sparse.hstack([-a, -eye])], format='csr')
    rhs = y/np.abs(y)
    eq = sparse.hstack([sparse.csr_matrix(np.ones((1, k))), sparse.csr_matrix((1, n))])
    res = linprog(np.r_[np.zeros(k), np.full(n, 1/n)], A_ub=constraints,
        b_ub=np.r_[rhs, -rhs], A_eq=eq, b_eq=[1.], bounds=(0., None), method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    w = np.clip(res.x[:k], 0, 1)
    return w/w.sum()


def band(h):
    return '15_30' if h <= 2 else '45_120'


def fit_weights(records, names):
    weights = {}
    for t in TARGETS:
        for b in ('15_30', '45_120'):
            rr = [r for r in records if r['target'] == t and band(r['h']) == b]
            y = np.concatenate([r['y'] for r in rr])
            p = np.concatenate([r['p'] for r in rr])
            weights[f'{t}/{b}'] = dict(zip(names, convex_weights(y, p).tolist()))
    return weights


def ensemble_predictions(predictions, weights, extreme=None, gating=None):
    """Blend member predictions. With gating, origins flagged in `extreme`
    use gating['extreme'] weights instead of the base (calm) weights."""
    result = {}
    for t in TARGETS:
        for h in HORIZONS:
            w = weights.get(f'{t}/h{h}') or weights[f'{t}/{band(h)}']
            p = sum(weight*predictions[name][t, h] for name, weight in w.items() if weight > 1e-8)
            if extreme is not None and gating and extreme.any():
                w = gating['extreme'][f'{t}/h{h}']
                pe = sum(weight*predictions[name][t, h] for name, weight in w.items() if weight > 1e-8)
                p = np.where(extreme, pe, p)
            result[f'{t}_t+{15*h}_pred'] = p
    for h in HORIZONS:
        group, total = f'generator_1_t+{15*h}_pred', f'generator_all_t+{15*h}_pred'
        result[group] = np.minimum(result[group], result[total])
    return result


def gating_extreme_mask(x, gating, origins):
    """Origin-row-only gate; NaN features resolve to calm (base weights)."""
    thr = gating['thresholds']
    s = x.loc[origins, gating['extreme_feature']].to_numpy(dtype=float)
    lvl = x.loc[origins, gating['level_feature']].to_numpy(dtype=float)
    with np.errstate(invalid='ignore'):
        ext = (np.abs(s) >= thr['slope16_abs']) | (lvl < thr['low_level']) | (lvl > thr['high_level'])
    return np.nan_to_num(ext.astype(float), nan=0.).astype(bool)


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def develop(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = read_data(args.train_dir)
    raw = raw.loc[raw.index < '2025-05-01']
    cutoffs = args.cutoffs
    specs = [s for s in SPECS if not args.members or s.name in args.members]
    names = [s.name for s in specs]
    all_records, metrics = [], []
    for start in cutoffs:
        cutoff = future([pd.Timestamp(start)], -1)[0]
        origins = pd.date_range(start, periods=192, freq='15min', name='datetime')
        if not origins.isin(raw.index).all():
            raise ValueError('Development origins must lie within training history')
        transform = InputTransform().fit(raw.loc[:cutoff])
        x = transform.transform(raw)
        predictions = {}
        for spec in specs:
            cache = out/f'fold_{start}_{spec.name}.joblib'
            # Resume only identical code/data/config folds.
            fingerprint = hashlib.sha256(Path(__file__).read_bytes()+
                pd.util.hash_pandas_object(raw, index=True).values.tobytes()+
                json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest()
            saved = joblib.load(cache) if cache.exists() else None
            if saved and saved['fingerprint'] == fingerprint:
                pred = saved['predictions']
            else:
                tic = time.perf_counter()
                model = ShortModel(spec, args.threads).fit(raw, x, cutoff)
                pred = model.predict(raw, x, origins)
                joblib.dump(dict(fingerprint=fingerprint, predictions=pred, audit=model.audit), cache)
                print(f'{start} {spec.name} fitted in {time.perf_counter()-tic:.1f}s', flush=True)
            predictions[spec.name] = pred
        for t in TARGETS:
            for h in HORIZONS:
                y = raw[t].reindex(future(origins, h)).to_numpy()
                p = np.column_stack([predictions[n][t, h] for n in names])
                all_records.append(dict(fold=start, target=t, h=h, y=y, p=p))
                for i, name in enumerate(names):
                    metrics.append(dict(fold=start, target=t, horizon=h, member=name, mape=score(y, p[:, i])))
        summary = pd.DataFrame(metrics)
        print(summary[summary.fold == start].groupby(['target', 'member']).mape.mean().unstack().round(5).to_string(), flush=True)
    tuning = [r for r in all_records if r['fold'] != cutoffs[-1]]
    weights = fit_weights(tuning, names)
    # Leave-one-development-fold-out predictions expose weight overfit.
    for fold in cutoffs:
        w = weights if fold == cutoffs[-1] else fit_weights([r for r in tuning if r['fold'] != fold], names)
        for r in [r for r in all_records if r['fold'] == fold]:
            ww = np.array([w[f"{r['target']}/{band(r['h'])}"][n] for n in names])
            metrics.append(dict(fold=fold, target=r['target'], horizon=r['h'],
                member='ensemble_heldout', mape=score(r['y'], r['p']@ww)))
    pd.DataFrame(metrics).to_csv(out/'development_metrics.csv', index=False)
    joblib.dump(all_records, out/'oof_predictions.joblib')
    selection = dict(version=1, names=names, specs=[asdict(s) for s in specs], weights=weights,
        tuning_cutoffs=cutoffs[:-1], holdout_cutoff=cutoffs[-1], train_only=True,
        protocol='Frozen model; observations available at each rolling origin; 8 exact horizons',
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.train_dir.glob('*.csv')})
    dump_json(out/'selection.json', selection)
    print(pd.DataFrame(metrics).groupby(['fold', 'target', 'member']).mape.mean().unstack().round(5).to_string(), flush=True)
    print(json.dumps(weights, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['develop'])
    p.add_argument('--train-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path('artifacts/prelim_v2'))
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--members', nargs='+', choices=[s.name for s in SPECS])
    p.add_argument('--cutoffs', nargs='+', default=['2025-04-21', '2025-04-23', '2025-04-25', '2025-04-29'])
    args = p.parse_args()
    develop(args)


if __name__ == '__main__':
    main()
