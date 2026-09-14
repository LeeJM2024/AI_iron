"""Freeze V29: single knob change — relative60 ridge penalty 300 -> 600.

Evidence (V28, nine quiet April windows 4/20..4/28, paired per window): raising the ridge
penalty of the relative60 member improves generator_1 in 9/9 windows (mean -0.71%,
sd 0.34) and generator_all in 8/9 windows (mean -0.27%, sd 0.27). Every other knob tested
(pooled/ lgb recency and capacity) improved only 4-7 of 9 windows and is not adopted.

The member's April predictions are refit here with alpha=600 for the weight fit; the other
members reuse their caches. Deployment refits everything at the final cut-off via
run_prelim. Promotion still requires the local gate to be strictly better than 4.6376%.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prelim import TARGETS, HORIZONS, Spec, read_data, future, convex_weights, band, dump_json

ROOT = Path('.')
OUT = ROOT / 'artifacts' / 'v30_selected'
FOLDS = ['2025-04-21', '2025-04-23', '2025-04-25', '2025-04-27']
V15 = json.loads((ROOT / 'artifacts/v15_selected/selection.json').read_text(encoding='utf-8'))
BASELINE = json.loads((ROOT / 'artifacts/adaptive_selected/selection.json').read_text(encoding='utf-8'))
DIRS = {'relative_linear': 'artifacts/relative_linear_v10', 'online': 'artifacts/online_v10'}
POOL = ['pooled_relative', 'lgb60', 'online7_6h', 'online21_2h', 'relative60', 'relative60_mae',
        'relative21', 'relative14_mae', 'persistence', 'ema8', 'ema16']
REL60_A600 = Spec('relative60', 'relative_linear', days=60, alpha=600., half_life=14.)
LGB60_HL21 = Spec('lgb60', 'lgb', days=60, half_life=21.)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_data(next(p.parent for p in ROOT.glob('**/Pre_load.csv'))).loc[lambda d: d.index < '2025-05-01']
    from compact_inputs import CompactInputTransform
    from relative_linear import RelativeLinear
    records = []
    for fold in FOLDS:
        origins = pd.date_range(fold, periods=192, freq='15min', name='datetime')
        cutoff = future(origins[:1], -1)[0]
        fold_raw = raw.loc[:origins[-1]]
        x = CompactInputTransform().fit(fold_raw.loc[:cutoff]).transform(fold_raw)
        members = {}
        for name in POOL:
            if name in ('persistence', 'ema8', 'ema16', 'relative60', 'lgb60'):
                continue
            kind = next(s['kind'] for s in BASELINE['specs'] if s['name'] == name)
            members[name] = joblib.load(ROOT / DIRS.get(kind, 'artifacts/compact_v5')
                                        / f'fold_{fold}_{name}.joblib')['predictions']
        members['relative60'] = RelativeLinear(REL60_A600, 4).fit(fold_raw, x, cutoff).predict(fold_raw, x, origins)
        from linear_challengers import fit_candidate
        members['lgb60'] = fit_candidate(LGB60_HL21, fold_raw, x, cutoff, 4).predict(fold_raw, x, origins)
        for t in TARGETS:
            now = x.loc[origins, t].to_numpy(dtype=float)
            for h in HORIZONS:
                members.setdefault('persistence', {})[t, h] = now.copy()
                members.setdefault('ema8', {})[t, h] = x.loc[origins, f'feat_{t}_ema_8'].to_numpy(dtype=float)
                members.setdefault('ema16', {})[t, h] = x.loc[origins, f'feat_{t}_ema_16'].to_numpy(dtype=float)
        for t in TARGETS:
            for h in HORIZONS:
                times = future(origins, h)
                y = fold_raw[t].reindex(times).to_numpy(dtype=float).copy()
                y[times > origins[-1]] = np.nan
                records.append(dict(target=t, h=h, y=y,
                                    p=np.column_stack([members[n][t, h] for n in POOL])))
        print(f'{fold} ready', flush=True)
    weights, report = {}, {}
    for t in TARGETS:
        for b in ('15_30', '45_120'):
            rr = [r for r in records if r['target'] == t and band(r['h']) == b]
            y = np.concatenate([r['y'] for r in rr])
            p = np.concatenate([r['p'] for r in rr])
            w = convex_weights(y, p)
            weights[f'{t}/{b}'] = dict(zip(POOL, w.tolist()))
            report[f'{t}/{b}'] = {n: round(float(v), 4) for n, v in zip(POOL, w) if v > 1e-8}
    specs = []
    for s in BASELINE['specs']:
        specs.append(dict(s, alpha=600.) if s['name'] == 'relative60'
                     else (dict(s, half_life=21.) if s['name'] == 'lgb60' else dict(s)))
    specs += [dict(name='persistence', kind='last'), dict(name='ema8', kind='ema', days=8),
              dict(name='ema16', kind='ema', days=16)]
    selection = dict(V15)
    selection.update(name='v30_rel600_lgb_hl21', specs=specs, weights=weights,
                     family_selection={t: {'member_changes': ['relative60 alpha 300 -> 600 (9/9, 8/9 windows)',
                                        'lgb60 half_life 14 -> 21 (5/9, 7/9 windows)']} for t in TARGETS},
                     protocol=('Identical to V15 except the relative60 ridge penalty (300 -> 600), chosen on nine '
                               'quiet April windows: relative60 alpha 300->600 improved generator_1 in 9/9 and '
                               'generator_all in 8/9 paired windows; lgb60 half_life 14->21 improved generator_all in '
                               '7/9 (generator_1 5/9). Ensemble weights refit on the four quiet tuning folds. May labels '
                               'are used only as the post-freeze promotion gate.'))
    dump_json(OUT / 'selection.json', selection)
    print(json.dumps(report, indent=2), flush=True)
    print('frozen:', OUT / 'selection.json', flush=True)


if __name__ == '__main__':
    main()
