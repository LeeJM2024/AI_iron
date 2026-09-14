"""V28: re-tune member recency/regularisation knobs on nine quiet April windows.

The deployed members' knobs were chosen on four windows (weak power). This scores each
candidate configuration as a standalone member on nine quiet post-break windows
(4/20..4/28), each fold trained only on data at or before its cut-off. Only knobs the
existing Spec supports are varied (days / trees / half_life for tree members, days / alpha
for the ridge member), so any winner needs no new model class.

Deployment stays compact-schema; May is never read.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prelim import Spec, TARGETS, HORIZONS, read_data, future, score
from compact_inputs import CompactInputTransform
from linear_challengers import fit_candidate

ROOT = Path('.')
OUT = ROOT / 'artifacts' / 'v28_tune'
FOLDS = ['2025-04-20', '2025-04-21', '2025-04-22', '2025-04-23', '2025-04-24',
         '2025-04-25', '2025-04-26', '2025-04-27', '2025-04-28']
LGB = Spec('lgb60', 'lgb', days=60, half_life=14.)
POOLED = Spec('pooled_relative', 'pooled', days=90, trees=500, half_life=21.)
REL60 = Spec('relative60', 'relative_linear', days=60, alpha=300., half_life=14.)
CONFIGS = {
    # tree member: recency decay and capacity knobs reachable through Spec
    'lgb_hl07': ('lgb', dict(name='lgb_hl07', kind='lgb', days=60, half_life=7.)),
    'lgb_hl14 (deployed)': ('lgb', dict(name='lgb_hl14', kind='lgb', days=60, half_life=14.)),
    'lgb_hl21': ('lgb', dict(name='lgb_hl21', kind='lgb', days=60, half_life=21.)),
    'lgb_hl30_t400': ('lgb', dict(name='lgb_hl30_t400', kind='lgb', days=60, trees=400, half_life=30.)),
    'lgb_d90_hl14': ('lgb', dict(name='lgb_d90_hl14', kind='lgb', days=90, half_life=14.)),
    # pooled member: window, trees, decay
    'pool90_hl21 (deployed)': ('pooled', dict(name='p', kind='pooled', days=90, trees=500, half_life=21.)),
    'pool90_hl10': ('pooled', dict(name='p', kind='pooled', days=90, trees=500, half_life=10.)),
    'pool90_hl30': ('pooled', dict(name='p', kind='pooled', days=90, trees=500, half_life=30.)),
    'pool60_hl21': ('pooled', dict(name='p', kind='pooled', days=60, trees=500, half_life=21.)),
    'pool90_hl21_t800': ('pooled', dict(name='p', kind='pooled', days=90, trees=800, half_life=21.)),
    # ridge member: decay and regularisation
    'rel60_hl14 (deployed)': ('relative_linear', dict(name='r', kind='relative_linear', days=60, alpha=300., half_life=14.)),
    'rel60_hl07': ('relative_linear', dict(name='r', kind='relative_linear', days=60, alpha=300., half_life=7.)),
    'rel60_hl30': ('relative_linear', dict(name='r', kind='relative_linear', days=60, alpha=300., half_life=30.)),
    'rel60_a100_hl14': ('relative_linear', dict(name='r', kind='relative_linear', days=60, alpha=100., half_life=14.)),
    'rel60_a600_hl14': ('relative_linear', dict(name='r', kind='relative_linear', days=60, alpha=600., half_life=14.)),
}


def build_model(kind, spec, fold_raw, x, cutoff, threads=4):
    if kind == 'pooled':
        from pooled_short import PooledShort
        return PooledShort(spec, threads).fit(fold_raw, x, cutoff)
    if kind == 'relative_linear':
        from relative_linear import RelativeLinear
        return RelativeLinear(spec, threads).fit(fold_raw, x, cutoff)
    return fit_candidate(spec, fold_raw, x, cutoff, threads)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_data(next(p.parent for p in ROOT.glob('**/Pre_load.csv'))).loc[lambda d: d.index < '2025-05-01']
    records = []
    for fold in FOLDS:
        origins = pd.date_range(fold, periods=192, freq='15min', name='datetime')
        cutoff = future(origins[:1], -1)[0]
        fold_raw = raw.loc[:origins[-1]]
        x = CompactInputTransform().fit(fold_raw.loc[:cutoff]).transform(fold_raw)
        for label, (kind, cfg) in CONFIGS.items():
            path = OUT / f'fold_{fold}_{abs(hash(label)) % 10**8}.joblib'
            if path.exists():
                pred = joblib.load(path)['predictions']
            else:
                model = build_model(kind, Spec(**cfg), fold_raw, x, cutoff)
                pred = model.predict(fold_raw, x, origins)
                joblib.dump(dict(predictions=pred), path)
            for t in TARGETS:
                for h in HORIZONS:
                    times = future(origins, h)
                    y = fold_raw[t].reindex(times).to_numpy(dtype=float)
                    y[times > origins[-1]] = np.nan
                    records.append(dict(fold=fold, target=t, h=h, config=label, mape=score(y, pred[t, h])))
        print(f'{fold} done', flush=True)
    table = pd.DataFrame(records)
    table.to_csv(OUT / 'members.csv', index=False)
    fam = {'lgb': 'lgb_', 'pool': 'pool', 'rel': 'rel'}
    for prefix, label in (('lgb_', 'LGB (per-horizon, days=60)'), ('pool', 'POOLED'), ('rel', 'RELATIVE (ridge)')):
        sub = table[table.config.str.startswith(prefix)]
        if sub.empty:
            continue
        print(f'\n=== {label} : mean over 9 quiet windows ===', flush=True)
        piv = sub.groupby(['target', 'config']).mape.mean().unstack() * 100
        print(piv.round(4).to_string(), flush=True)
        print('relative to deployed:', flush=True)
        for t in TARGETS:
            row = piv.loc[t]
            ref = [c for c in row.index if 'deployed' in c]
            if not ref:
                continue
            for c in row.index:
                print(f'  {t:14s} {c:26s} {100*(row[c]/row[ref[0]]-1):+.2f}%', flush=True)
    print('No May data read.', flush=True)


if __name__ == '__main__':
    main()
