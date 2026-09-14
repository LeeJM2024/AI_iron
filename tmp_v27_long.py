"""V27: long-horizon probes — how much room is left at h=6..8, and does stacking find it?

Two probes, both April-LODO over the five windows (each fold scored by models that never
saw its labels):

(a) direct high-capacity learner on the compact row schema, per horizon h=6,7,8;
(b) a per-horizon STACKED meta-learner: inputs are the pool members' relative predictions
    plus origin state features (volatility, clock, holder state, member spread), target is
    the realised relative change. This is strictly more general than the fixed banded
    weights, so it can express conditioning the blend cannot.

Reference numbers are the deployed V15 blend and the best member per horizon. May never
read.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from prelim import TARGETS, HORIZONS, UPPER, read_data, future, score, band
from compact_inputs import CompactInputTransform

ROOT = Path('.')
OUT = ROOT / 'artifacts' / 'v27_long'
FOLDS = ['2025-04-21', '2025-04-23', '2025-04-25', '2025-04-27', '2025-04-29']
BASELINE = json.loads((ROOT / 'artifacts/adaptive_selected/selection.json').read_text(encoding='utf-8'))
V15 = json.loads((ROOT / 'artifacts/v15_selected/selection.json').read_text(encoding='utf-8'))
DIRS = {'relative_linear': 'artifacts/relative_linear_v10', 'online': 'artifacts/online_v10'}
MEMBERS = ['pooled_relative', 'lgb60', 'online7_6h', 'online21_2h', 'relative60', 'relative60_mae',
           'relative21', 'relative14_mae']
HOLDER = 'blast_furnace_gas_holder_2'


def load(raw):
    store = {}
    for fold in FOLDS:
        origins = pd.date_range(fold, periods=192, freq='15min', name='datetime')
        cutoff = future(origins[:1], -1)[0]
        fold_raw = raw.loc[:origins[-1]]
        x = CompactInputTransform().fit(fold_raw.loc[:cutoff]).transform(fold_raw)
        members = {}
        for name in MEMBERS:
            kind = next(s['kind'] for s in BASELINE['specs'] if s['name'] == name)
            members[name] = joblib.load(ROOT / DIRS.get(kind, 'artifacts/compact_v5')
                                        / f'fold_{fold}_{name}.joblib')['predictions']
        for t in TARGETS:
            now = x.loc[origins, t].to_numpy(dtype=float)
            for h in HORIZONS:
                members.setdefault('persistence', {})[t, h] = now.copy()
                members.setdefault('ema8', {})[t, h] = x.loc[origins, f'feat_{t}_ema_8'].to_numpy(dtype=float)
                members.setdefault('ema16', {})[t, h] = x.loc[origins, f'feat_{t}_ema_16'].to_numpy(dtype=float)
        store[fold] = dict(origins=origins, raw=fold_raw, x=x, members=members)
    return store


def state_features(entry, t):
    x = entry['x']
    origins = entry['origins']
    now = x.loc[origins, t].to_numpy(dtype=float)
    f = {}
    f['clock_sin'] = x.loc[origins, 'feat_clock_sin_1'].to_numpy(dtype=float)
    f['clock_cos'] = x.loc[origins, 'feat_clock_cos_1'].to_numpy(dtype=float)
    for n in (1, 4, 48):
        col = f'feat_{t}_lag_{n}'
        if col in x:
            f[f'd_{n}'] = (now - x.loc[origins, col].to_numpy(dtype=float)) / np.maximum(now, 1.)
    f['ema16_ratio'] = now / np.maximum(x.loc[origins, f'feat_{t}_ema_16'].to_numpy(dtype=float), 1.) - 1.
    holder = x.loc[origins, HOLDER].to_numpy(dtype=float) / 1e5
    f['holder'] = holder
    f['holder_chg4'] = (x.loc[origins, HOLDER].to_numpy(dtype=float) -
                        x.loc[origins, HOLDER].shift(4).to_numpy(dtype=float)) / 1e4
    return pd.DataFrame(f, index=origins)


def stack_frame(entry, t, h):
    now = entry['x'].loc[entry['origins'], t].to_numpy(dtype=float)
    cols = {}
    for name in MEMBERS + ['persistence', 'ema8', 'ema16']:
        p = entry['members'][name][t, h]
        cols[f'm_{name}'] = (p - now) / np.maximum(now, 1.)
    frame = pd.DataFrame(cols, index=entry['origins'])
    state = state_features(entry, t)
    frame = pd.concat([frame, state], axis=1)
    member_block = frame[[f'm_{n}' for n in MEMBERS]]
    frame['spread'] = member_block.std(axis=1)
    frame['spread_abs'] = member_block.abs().mean(axis=1)
    return frame


def direct_frame(entry, t):
    return entry['x'].loc[entry['origins']]


def fit_lgb(X, y, w, params):
    model = LGBMRegressor(objective='regression_l1', random_state=2026, deterministic=True,
                          force_col_wise=True, verbosity=-1, n_jobs=4, **params)
    model.fit(X, y, sample_weight=w)
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_data(next(p.parent for p in ROOT.glob('**/Pre_load.csv'))).loc[lambda d: d.index < '2025-05-01']
    store = load(raw)
    rows = []
    for t in TARGETS:
        for h in HORIZONS:
            for fold in FOLDS:
                entry = store[fold]
                times = future(entry['origins'], h)
                y = entry['raw'][t].reindex(times).to_numpy(dtype=float).copy()
                y[times > entry['origins'][-1]] = np.nan
                now = entry['x'].loc[entry['origins'], t].to_numpy(dtype=float)
                rel = (y - now) / np.maximum(now, 1.)
                w = np.maximum(now, 1.) / np.where(np.isfinite(y) & (y > 0), y, np.nan)
                ok = np.isfinite(y) & (y > 0) & np.isfinite(rel)
                # --- references
                wb = V15['weights'][f'{t}/{band(h)}']
                blend = sum(wb[n] * entry['members'][n][t, h] for n in [m for m in V15['weights'][f'{t}/{band(h)}'] if wb[m] > 1e-8])
                rows.append(dict(target=t, h=h, fold=fold, model='v15_blend', mape=score(y, blend)))
                for name in ('lgb60', 'pooled_relative', 'relative60_mae'):
                    rows.append(dict(target=t, h=h, fold=fold, model=f'member_{name}',
                                     mape=score(y, entry['members'][name][t, h])))
                # --- stacking meta-learner (trained on the other four folds)
                train = [f for f in FOLDS if f != fold]
                Xtr, ytr, wtr = [], [], []
                for f2 in train:
                    e2 = store[f2]
                    t2 = future(e2['origins'], h)
                    y2 = e2['raw'][t].reindex(t2).to_numpy(dtype=float).copy()
                    y2[t2 > e2['origins'][-1]] = np.nan
                    n2 = e2['x'].loc[e2['origins'], t].to_numpy(dtype=float)
                    rel2 = (y2 - n2) / np.maximum(n2, 1.)
                    ok2 = np.isfinite(y2) & (y2 > 0)
                    X2 = stack_frame(e2, t, h).to_numpy(dtype=float)[ok2]
                    Xtr.append(X2)
                    ytr.append(rel2[ok2])
                    wtr.append((np.maximum(n2, 1.) / y2)[ok2])
                Xtr = np.vstack(Xtr)
                ytr = np.concatenate(ytr)
                wtr = np.concatenate(wtr)
                wtr = wtr / wtr.mean()
                for label, params in (('stack_small', dict(n_estimators=300, num_leaves=7, learning_rate=.03, min_child_samples=200, reg_lambda=20.)),
                                      ('stack_mid', dict(n_estimators=600, num_leaves=15, learning_rate=.02, min_child_samples=100, reg_lambda=10.))):
                    model = fit_lgb(Xtr, ytr, wtr, params)
                    Xte = stack_frame(entry, t, h).to_numpy(dtype=float)[ok]
                    rel_hat = model.predict(Xte)
                    pred = now.copy()
                    pred[ok] = now[ok] + np.maximum(now[ok], 1.) * rel_hat
                    rows.append(dict(target=t, h=h, fold=fold, model=label, mape=score(y, pred)))
                # --- direct high-capacity learner on the row schema
                Xd, yd, wd = [], [], []
                for f2 in train:
                    e2 = store[f2]
                    t2 = future(e2['origins'], h)
                    y2 = e2['raw'][t].reindex(t2).to_numpy(dtype=float).copy()
                    y2[t2 > e2['origins'][-1]] = np.nan
                    n2 = e2['x'].loc[e2['origins'], t].to_numpy(dtype=float)
                    rel2 = (y2 - n2) / np.maximum(n2, 1.)
                    ok2 = np.isfinite(y2) & (y2 > 0)
                    Xd.append(direct_frame(e2, t).to_numpy(dtype=float)[ok2])
                    yd.append(rel2[ok2])
                    wd.append((np.maximum(n2, 1.) / y2)[ok2])
                Xd = np.vstack(Xd)
                yd = np.concatenate(yd)
                wd = np.concatenate(wd)
                wd = wd / wd.mean()
                direct = fit_lgb(Xd, yd, wd, dict(n_estimators=1200, num_leaves=31, learning_rate=.015,
                                                  min_child_samples=40, reg_lambda=20.))
                rel_hat = direct.predict(direct_frame(entry, t).to_numpy(dtype=float)[ok])
                pred = now.copy()
                pred[ok] = now[ok] + np.maximum(now[ok], 1.) * rel_hat
                rows.append(dict(target=t, h=h, fold=fold, model='direct_hc', mape=score(y, pred)))
        print(f'{t} done', flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'probe.csv', index=False)
    piv = table.groupby(['target', 'h', 'model']).mape.mean().unstack() * 100
    print('\n=== LODO MAPE by model (5 folds) ===', flush=True)
    print(piv.round(3).to_string(), flush=True)
    print('\n=== long-horizon comparison (h=6..8) ===', flush=True)
    for t in TARGETS:
        sub = piv.loc[t].loc[[6, 7, 8]]
        base = sub['v15_blend']
        print(t, flush=True)
        for m in sub.columns:
            print(f'   {m:22s} ' + '  '.join(f'h{h} {sub.loc[h, m]:.3f} ({100*(sub.loc[h, m]/base[h]-1):+.1f}%)'
                                             for h in (6, 7, 8)), flush=True)
    print('No May data read.', flush=True)


if __name__ == '__main__':
    main()
