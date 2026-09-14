"""V36: paired sweep of the tree members' regularisation on the compact schema.

The ridge members' penalties were far too weak (relative60 and relative21 alpha sweeps),
and tightening relative60's alpha is the change that carried V30 through the local gate.
The tree members' own regularisation (leaves / min_child_samples / learning rate / L2 /
colsample) has never been swept on the deployed compact schema: earlier regularisation
tests were run on a different schema and pool. Same paired protocol as V28/V31: each
configuration is fitted per fold on data at or before that fold's cut-off and scored on
the fold, over nine quiet April windows. May never read.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from prelim import TARGETS, HORIZONS, UPPER, read_data, future, score
from compact_inputs import CompactInputTransform
from pooled_short import augment as pooled_augment

ROOT = Path('.')
OUT = ROOT / 'artifacts' / 'v36_trees'
FOLDS = ['2025-04-20', '2025-04-21', '2025-04-22', '2025-04-23', '2025-04-24',
         '2025-04-25', '2025-04-26', '2025-04-27', '2025-04-28']
POOLED_HP = {
    'pool_base (deployed)': dict(trees=500, leaves=23, mcs=120, lr=.03, lam=10., cs=.9),
    'pool_tighter': dict(trees=500, leaves=15, mcs=240, lr=.03, lam=20., cs=.9),
    'pool_wider': dict(trees=700, leaves=31, mcs=60, lr=.03, lam=10., cs=.9),
    'pool_strong_l2': dict(trees=500, leaves=23, mcs=240, lr=.03, lam=40., cs=.8),
}
LGB_HP = {
    'lgb_base (deployed)': dict(trees=240, leaves=15, mcs=40, lr=.035, lam=8., cs=.85),
    'lgb_tighter': dict(trees=400, leaves=15, mcs=80, lr=.03, lam=20., cs=.85),
    'lgb_smaller': dict(trees=300, leaves=9, mcs=60, lr=.03, lam=20., cs=.9),
    'lgb_strong_l2': dict(trees=400, leaves=15, mcs=120, lr=.03, lam=40., cs=.7),
}


def fit_pooled(x, raw, target, days, hl, hp, cutoff, threads=4):
    matrices, labels, weights = [], [], []
    for h in HORIZONS:
        rows = x.index[(x.index > cutoff - pd.Timedelta(days=days)) &
                       (x.index <= cutoff - pd.Timedelta(minutes=15 * h))]
        y = raw[target].reindex(future(rows, h)).to_numpy(dtype=float)
        ok = np.isfinite(y) & (y > 0) & (y <= UPPER[target]) & raw.loc[rows, target].notna().to_numpy()
        rows, y = rows[ok], y[ok]
        now = x.loc[rows, target].to_numpy(dtype=float)
        residual = (y - now) / np.maximum(now, 1.)
        w = np.maximum(now, 1.) / y
        if hl:
            age = np.asarray((cutoff - rows).total_seconds()) / 86400.
            w = w * np.exp2(-age / hl)
        w = w / w.sum()
        matrices.append(pooled_augment(x.loc[rows], h, target))
        labels.append(residual)
        weights.append(w)
    z = pd.concat(matrices, ignore_index=True)
    label = np.concatenate(labels)
    w = np.concatenate(weights)
    w = w / w.mean()
    model = LGBMRegressor(objective='regression_l1', n_estimators=hp['trees'], num_leaves=hp['leaves'],
                          learning_rate=hp['lr'], min_child_samples=hp['mcs'], reg_lambda=hp['lam'],
                          reg_alpha=.1, colsample_bytree=hp['cs'], n_jobs=threads, random_state=2026,
                          deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(z, label, sample_weight=w)
    return model


def fit_lgb_horizon(x, raw, target, h, days, hl, hp, cutoff, threads=4):
    end = future([cutoff], -h)[0]
    start = future([cutoff], -96 * days)[0]
    rows = x.index[(x.index > start) & (x.index <= end)]
    y = raw[target].reindex(future(rows, h)).to_numpy(dtype=float)
    ok = np.isfinite(y) & (y > 0) & (y <= UPPER[target]) & raw.loc[rows, target].notna().to_numpy()
    rows, y = rows[ok], y[ok]
    z = x.loc[rows].to_numpy(dtype=float)
    baseline = x.loc[rows, target].to_numpy(dtype=float)
    residual = y - baseline
    w = 1. / np.maximum(y, 1.)
    if hl:
        age = np.asarray((cutoff - rows).total_seconds()) / 86400.
        w = w * np.exp2(-age / hl)
    w = w / w.mean()
    model = LGBMRegressor(objective='regression_l1', n_estimators=hp['trees'], num_leaves=hp['leaves'],
                          learning_rate=hp['lr'], min_child_samples=hp['mcs'], reg_lambda=hp['lam'],
                          colsample_bytree=hp['cs'], random_state=2026, deterministic=True,
                          force_col_wise=True, verbosity=-1, n_jobs=threads)
    model.fit(z, residual, sample_weight=w)
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_data(next(p.parent for p in ROOT.glob('**/Pre_load.csv'))).loc[lambda d: d.index < '2025-05-01']
    records = []
    for fold in FOLDS:
        origins = pd.date_range(fold, periods=192, freq='15min', name='datetime')
        cutoff = future(origins[:1], -1)[0]
        fold_raw = raw.loc[:origins[-1]]
        x = CompactInputTransform().fit(fold_raw.loc[:cutoff]).transform(fold_raw)
        for label, hp in POOLED_HP.items():
            path = OUT / f'{fold}_{label}.joblib'
            if path.exists():
                pred = joblib.load(path)['predictions']
            else:
                pred = {}
                models = {t: fit_pooled(x, fold_raw, t, 90, 21., hp, cutoff) for t in TARGETS}
                for t in TARGETS:
                    now = x.loc[origins, t].to_numpy(dtype=float)
                    for h in HORIZONS:
                        r = models[t].predict(pooled_augment(x.loc[origins], h, t))
                        pred[t, h] = np.clip(now + np.maximum(now, 1.) * r, 0, UPPER[t])
                joblib.dump(dict(predictions=pred), path)
            for t in TARGETS:
                for h in HORIZONS:
                    times = future(origins, h)
                    y = fold_raw[t].reindex(times).to_numpy(dtype=float)
                    y[times > origins[-1]] = np.nan
                    records.append(dict(fold=fold, target=t, h=h, config=label, mape=score(y, pred[t, h])))
        for label, hp in LGB_HP.items():
            path = OUT / f'{fold}_{label}.joblib'
            if path.exists():
                pred = joblib.load(path)['predictions']
            else:
                pred = {}
                for t in TARGETS:
                    now = x.loc[origins, t].to_numpy(dtype=float)
                    for h in HORIZONS:
                        model = fit_lgb_horizon(x, fold_raw, t, h, 60, 21., hp, cutoff)
                        p = now + model.predict(x.loc[origins].to_numpy(dtype=float))
                        pred[t, h] = np.clip(p, 0, UPPER[t])
                joblib.dump(dict(predictions=pred), path)
            for t in TARGETS:
                for h in HORIZONS:
                    times = future(origins, h)
                    y = fold_raw[t].reindex(times).to_numpy(dtype=float)
                    y[times > origins[-1]] = np.nan
                    records.append(dict(fold=fold, target=t, h=h, config=label, mape=score(y, pred[t, h])))
        print(f'{fold} done', flush=True)
    table = pd.DataFrame(records)
    table.to_csv(OUT / 'tree_knobs.csv', index=False)
    piv = table.pivot_table(index=['target', 'fold'], columns='config', values='mape')
    print('\n=== paired comparison vs deployed ===', flush=True)
    for fam, ref in (('pool_', 'pool_base (deployed)'), ('lgb_', 'lgb_base (deployed)')):
        for cand in [c for c in piv.columns if c.startswith(fam) and c != ref]:
            for t in TARGETS:
                rel = (piv.loc[t, cand] / piv.loc[t, ref] - 1) * 100
                print(f'{cand:22s} {t:14s} mean {rel.mean():+.2f}%  sd {rel.std():.2f}  '
                      f'improved {int((rel < 0).sum())}/{len(rel)} windows', flush=True)
    print('No May data read.', flush=True)


if __name__ == '__main__':
    main()
