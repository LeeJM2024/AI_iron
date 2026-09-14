"""V31: paired knob sweep for the remaining weighted members (nine quiet windows).

V28 showed the ridge penalty of relative60 was too weak (alpha 300 -> 600 improved
generator_1 in 9/9 and generator_all in 8/9 paired windows). This applies the same
paired protocol to the other members that carry ensemble weight: the MAE-regularised
relative members and the two online ridge members. Each configuration is fitted per fold
on data at or before that fold's cut-off and scored on the fold itself, so every
comparison is paired by window. Stable cache names; May never read.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prelim import Spec, TARGETS, HORIZONS, read_data, future, score
from compact_inputs import CompactInputTransform

ROOT = Path('.')
OUT = ROOT / 'artifacts' / 'v31_knobs'
FOLDS = ['2025-04-20', '2025-04-21', '2025-04-22', '2025-04-23', '2025-04-24',
         '2025-04-25', '2025-04-26', '2025-04-27', '2025-04-28']
CONFIGS = {
    'rel60mae_a300 (deployed)': dict(name='relative60_mae', kind='relative_linear', days=60, alpha=300., half_life=14.),
    'rel60mae_a600': dict(name='relative60_mae', kind='relative_linear', days=60, alpha=600., half_life=14.),
    'rel60mae_a1000': dict(name='relative60_mae', kind='relative_linear', days=60, alpha=1000., half_life=14.),
    'rel60mae_a600_hl21': dict(name='relative60_mae', kind='relative_linear', days=60, alpha=600., half_life=21.),
    'rel21_a100 (deployed)': dict(name='relative21', kind='relative_linear', days=21, alpha=100., half_life=0.),
    'rel21_a300': dict(name='relative21', kind='relative_linear', days=21, alpha=300., half_life=0.),
    'rel21_a600': dict(name='relative21', kind='relative_linear', days=21, alpha=600., half_life=0.),
    'rel21_a1000': dict(name='relative21', kind='relative_linear', days=21, alpha=1000., half_life=0.),
    'rel21_a2000': dict(name='relative21', kind='relative_linear', days=21, alpha=2000., half_life=0.),
    'rel21_a4000': dict(name='relative21', kind='relative_linear', days=21, alpha=4000., half_life=0.),
    'rel14mae_a100 (deployed)': dict(name='relative14_mae', kind='relative_linear', days=14, alpha=100., half_life=0.),
    'rel14mae_a300': dict(name='relative14_mae', kind='relative_linear', days=14, alpha=300., half_life=0.),
    'rel14mae_a600': dict(name='relative14_mae', kind='relative_linear', days=14, alpha=600., half_life=0.),
    'online7_a300 (deployed)': dict(name='online7_6h', kind='online', days=7, alpha=300.),
    'online7_a600': dict(name='online7_6h', kind='online', days=7, alpha=600.),
    'online21_a100 (deployed)': dict(name='online21_2h', kind='online', days=21, alpha=100.),
    'online21_a300': dict(name='online21_2h', kind='online', days=21, alpha=300.),
}


def build(cfg, fold_raw, x, cutoff):
    spec = Spec(**cfg)
    if spec.kind == 'relative_linear':
        from relative_linear import RelativeLinear
        return RelativeLinear(spec, 4).fit(fold_raw, x, cutoff)
    if spec.kind == 'online':
        from online_model import OnlineRidge
        return OnlineRidge(spec, 4).fit(fold_raw, x, cutoff)
    raise ValueError(spec.kind)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_data(next(p.parent for p in ROOT.glob('**/Pre_load.csv'))).loc[lambda d: d.index < '2025-05-01']
    records = []
    for fold in FOLDS:
        origins = pd.date_range(fold, periods=192, freq='15min', name='datetime')
        cutoff = future(origins[:1], -1)[0]
        fold_raw = raw.loc[:origins[-1]]
        x = CompactInputTransform().fit(fold_raw.loc[:cutoff]).transform(fold_raw)
        for label, cfg in CONFIGS.items():
            safe = label.replace(' ', '_').replace('(', '').replace(')', '')
            path = OUT / f'{fold}_{safe}.joblib'
            if path.exists():
                pred = joblib.load(path)['predictions']
            else:
                model = build(cfg, fold_raw, x, cutoff)
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
    table.to_csv(OUT / 'knobs.csv', index=False)
    piv = table.pivot_table(index=['target', 'fold'], columns='config', values='mape')
    families = {'rel60mae': 'rel60mae_a300 (deployed)', 'rel21': 'rel21_a100 (deployed)',
                'rel14mae': 'rel14mae_a100 (deployed)', 'online7': 'online7_a300 (deployed)',
                'online21': 'online21_a100 (deployed)'}
    print('\n=== paired comparison vs deployed config ===', flush=True)
    for fam, ref in families.items():
        cands = [c for c in piv.columns if c.startswith(fam) and c != ref]
        for cand in cands:
            for t in TARGETS:
                a, b = piv.loc[t, cand], piv.loc[t, ref]
                rel = (a / b - 1) * 100
                print(f'{fam:10s} {cand:26s} {t:14s} mean {rel.mean():+.2f}%  sd {rel.std():.2f}  '
                      f'improved {int((rel < 0).sum())}/{len(rel)} windows', flush=True)
    print('No May data read.', flush=True)


if __name__ == '__main__':
    main()
