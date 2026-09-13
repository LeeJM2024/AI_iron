"""Regularized/robust ARX challengers, evaluated only on April development folds."""
from __future__ import annotations
import argparse
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from prelim import Spec, ShortModel, InputTransform, read_data, future, score, TARGETS, HORIZONS, UPPER


class RobustRidge:
    """IRLS smooth-L1 approximation with fixed scale from training residuals.

    Each reweighted ridge fit uses inverse-load weights, so large plant transients
    do not dominate the fit as they do under squared error. No future calibration.
    """
    def __init__(self, alpha=100., iterations=8):
        self.alpha, self.iterations = alpha, iterations

    def fit(self, x, y, sample_weight=None):
        base = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        self.model = Ridge(alpha=self.alpha).fit(x, y, sample_weight=base)
        residual = y-self.model.predict(x)
        epsilon = max(.2, .15*float(np.median(np.abs(residual))))
        for _ in range(self.iterations):
            residual = y-self.model.predict(x)
            w = base/np.sqrt(residual**2+epsilon**2)
            w /= w.mean()
            self.model = Ridge(alpha=self.alpha).fit(x, y, sample_weight=w)
        return self

    def predict(self, x):
        return self.model.predict(x)


EXTRA_SPECS = (
    Spec('ridge7_strong', 'ridge', days=7, alpha=300.),
    Spec('ridge21_strong', 'ridge', days=21, alpha=1000.),
    Spec('ridge60_strong', 'ridge', days=60, alpha=1000., half_life=14.),
    Spec('robust21', 'ridge', days=21, alpha=100.),
    Spec('robust60', 'ridge', days=60, alpha=300., half_life=14.),
)


def fit_candidate(spec, raw, x, cutoff, threads=4):
    fitted = ShortModel(spec, threads).fit(raw, x, cutoff)
    if not spec.name.startswith('robust'):
        return fitted
    for t in TARGETS:
        for h in HORIZONS:
            end = cutoff-pd.Timedelta(15*h, unit='min')
            rows = x.index[(x.index > cutoff-pd.Timedelta(spec.days, unit='D')) & (x.index <= end)]
            y = raw[t].reindex(future(rows, h)).to_numpy()
            ok = np.isfinite(y) & (y > 0) & (y <= UPPER[t]) & raw.loc[rows, t].notna().to_numpy()
            rows, y = rows[ok], y[ok]
            w = 1/np.maximum(y, 1.)
            if spec.half_life > 0:
                age = np.asarray((cutoff-rows).total_seconds())/86400
                w *= np.exp2(-age/spec.half_life)
            w /= w.mean()
            z = fitted.scalers[t, h].transform(x.loc[rows, fitted.columns].to_numpy(dtype=float))
            fitted.models[t, h] = RobustRidge(spec.alpha).fit(z, y-x.loc[rows, t].to_numpy(), sample_weight=w)
    return fitted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path('artifacts/prelim_linear'))
    p.add_argument('--cutoffs', nargs='+', default=['2025-04-21', '2025-04-23', '2025-04-25', '2025-04-29'])
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_data(args.train_dir)
    raw = raw.loc[raw.index < '2025-05-01']
    metrics = []
    for fold in args.cutoffs:
        cutoff = pd.Timestamp(fold)-pd.Timedelta(15, unit='min')
        origins = pd.date_range(fold, periods=192, freq='15min', name='datetime')
        tx = InputTransform().fit(raw.loc[:cutoff])
        x = tx.transform(raw)
        for spec in EXTRA_SPECS:
            tic = time.perf_counter()
            m = fit_candidate(spec, raw, x, cutoff)
            pred = m.predict(raw, x, origins)
            joblib.dump(dict(predictions=pred, audit=m.audit), args.output_dir/f'fold_{fold}_{spec.name}.joblib')
            for t in TARGETS:
                for h in HORIZONS:
                    y = raw[t].reindex(future(origins, h)).to_numpy()
                    metrics.append(dict(fold=fold, target=t, horizon=h, member=spec.name, mape=score(y, pred[t,h])))
            print(f'{fold} {spec.name} {time.perf_counter()-tic:.1f}s', flush=True)
    table = pd.DataFrame(metrics)
    table.to_csv(args.output_dir/'metrics.csv', index=False)
    print(table.groupby(['fold','target','member']).mape.mean().unstack().round(5).to_string(), flush=True)


if __name__ == '__main__':
    main()
