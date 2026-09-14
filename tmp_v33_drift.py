"""V33: pure horizon-drift members. Is the drift term what the ridge member was really worth?

The alpha sweep showed relative21 improves monotonically up to alpha ~1e3-2e3, i.e. its
value comes from a horizon-specific average relative change rather than from feature
coefficients. This tests that signal directly: p = y_t * (1 + lambda * d_h) where d_h is
the recency-weighted median of observed relative changes over the training window.
Paired by window over the nine quiet April folds; May never read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prelim import TARGETS, HORIZONS, UPPER, read_data, future, score

ROOT = Path('.')
OUT = ROOT / 'artifacts' / 'v33_drift'
FOLDS = ['2025-04-20', '2025-04-21', '2025-04-22', '2025-04-23', '2025-04-24',
         '2025-04-25', '2025-04-26', '2025-04-27', '2025-04-28']
WINDOW_DAYS = (21, 60, 90)
HALF_LIVES = (0., 14., 21.)
LAMBDAS = (0.5, 0.75, 1.0)


def wmedian(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, 0.5)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_data(next(p.parent for p in ROOT.glob('**/Pre_load.csv'))).loc[lambda d: d.index < '2025-05-01']
    rows = []
    for fold in FOLDS:
        origins = pd.date_range(fold, periods=192, freq='15min', name='datetime')
        cutoff = future(origins[:1], -1)[0]
        fr = raw.loc[:origins[-1]]
        for t in TARGETS:
            now = fr[t].reindex(origins).ffill().bfill().to_numpy(dtype=float)
            for h in HORIZONS:
                times = future(origins, h)
                y = fr[t].reindex(times).to_numpy(dtype=float).copy()
                y[times > origins[-1]] = np.nan
                rows.append(dict(fold=fold, target=t, h=h, y=y, now=now,
                                 pred={}, train=fr[t], cutoff=cutoff))
    records = []
    for r in rows:
        fr, cutoff, t, h = r['train'], r['cutoff'], r['target'], r['h']
        for days in WINDOW_DAYS:
            rows_ix = fr.index[(fr.index > cutoff - pd.Timedelta(days=days)) &
                               (fr.index <= cutoff - pd.Timedelta(minutes=15 * h))]
            y_tr = fr.reindex(rows_ix).to_numpy(dtype=float)
            base_tr = y_tr  # current value equals the label at the same row
            lab = fr.reindex(future(rows_ix, h)).to_numpy(dtype=float)
            ok = (np.isfinite(lab) & (lab > 0) & (lab <= UPPER[t]) &
                  np.isfinite(base_tr) & (base_tr > 0))
            rows_ix, lab, base_tr = rows_ix[ok], lab[ok], base_tr[ok]
            if len(rows_ix) < 96:
                continue
            rel = lab / base_tr - 1.
            w = np.maximum(base_tr, 1.) / lab
            for hl in HALF_LIVES:
                ww = w.copy()
                if hl:
                    age = np.asarray((cutoff - rows_ix).total_seconds()) / 86400.
                    ww = ww * np.exp2(-age / hl)
                d = wmedian(rel, ww)
                for lam in LAMBDAS:
                    pred = np.clip(r['now'] * (1. + lam * d), 0, UPPER[t])
                    records.append(dict(fold=r['fold'], target=t, h=h, days=days, hl=hl, lam=lam,
                                        d=d, mape=score(r['y'], pred)))
        # reference: deployed relative21 (alpha=100) predictions are not cached here;
        # persistence is reported instead as a scale reference.
        records.append(dict(fold=r['fold'], target=t, h=h, days=0, hl=0., lam=0.,
                            d=0., mape=score(r['y'], r['now'])))
    table = pd.DataFrame(records)
    table.to_csv(OUT / 'drift.csv', index=False)
    best = table[table.days > 0].groupby(['target', 'days', 'hl', 'lam']).mape.mean().reset_index()
    for t in TARGETS:
        sub = best[best.target == t].sort_values('mape')
        print(f'=== {t}: best drift configurations (9 windows) ===', flush=True)
        print(sub.head(8).round(5).to_string(index=False), flush=True)
    print('\n=== persistence reference ===', flush=True)
    print((table[table.days == 0].groupby('target').mape.mean() * 100).round(3).to_string(), flush=True)
    # per-figure drift values for the best global config
    print('\n=== drift values d_h (days=60, hl=14), per fold, generator_all ===', flush=True)
    sub = table[(table.days == 60) & (table.hl == 14) & (table.lam == 1.0) & (table.target == 'generator_all')]
    print((sub.pivot_table(index='h', columns='fold', values='d') * 100).round(2).to_string(), flush=True)
    print('No May data read.', flush=True)


if __name__ == '__main__':
    main()
