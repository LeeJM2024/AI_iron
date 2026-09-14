"""Freeze the V15 candidate: V10 member pool with capped blend weights.

April bake-off (artifacts/v15_diag/bakeoff.csv) compared combination schemes x pools
with leave-fold-out weight fitting. Capping any single member's blend weight at 0.25
was the only change that improved generator_1 and generator_all on both the four
tuning folds and the April 29 monitor; extending the generator_1 pool with the
persistence/EMA members also improved both. Those two changes are frozen here, and
nothing is tuned on May.

Inputs: cached April fold predictions (no retraining) plus the exported compact rows.
Output: artifacts/v15_selected/selection.json for run_prelim.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prelim import TARGETS, HORIZONS, Spec, read_data, future, convex_weights, band, dump_json
from compact_inputs import CompactInputTransform

ROOT = Path('.')
OUT = ROOT / 'artifacts' / 'v15_selected'
BASELINE = json.loads((ROOT / 'artifacts/adaptive_selected/selection.json').read_text(encoding='utf-8'))
FOLDS = BASELINE['tuning_cutoffs']
DIRS = {'relative_linear': 'artifacts/relative_linear_v10', 'online': 'artifacts/online_v10'}
EXTRA_SPECS = [dict(name='persistence', kind='last'),
               dict(name='ema8', kind='ema', days=8),
               dict(name='ema16', kind='ema', days=16)]
CAP = 0.25


def capped(values, cap=CAP):
    """Euclidean projection onto {w >= 0, sum(w) = 1, w <= cap}.

    w = clip(v - lambda, 0, cap); f(lambda) is decreasing, f(-1) = k*cap >= 1 and
    f(1) = 0, so the level is bracketed on [-1, 1]. With the extended pool the LP
    weights already sit below the cap in every band, so this is a no-op here and the
    deployed weights are the raw LP solution; the helper is kept for reuse.
    """
    v = np.clip(np.asarray(values, dtype=float), 0., None)
    v = v / v.sum()
    if cap is None or (v <= cap).all():
        return v
    if cap * len(v) < 1:
        raise ValueError('Infeasible cap for this pool size')
    lo, hi = -1., 1.
    for _ in range(200):
        mid = .5 * (lo + hi)
        if np.clip(v - mid, 0., cap).sum() > 1.:
            lo = mid
        else:
            hi = mid
    w = np.clip(v - .5 * (lo + hi), 0., cap)
    return w / w.sum()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read_data(next(p.parent for p in ROOT.glob('**/Pre_load.csv')))
    raw = raw.loc[raw.index < '2025-05-01']
    names = [s['name'] for s in BASELINE['specs']] + [s['name'] for s in EXTRA_SPECS]
    records = []
    for fold in FOLDS:
        origins = pd.date_range(fold, periods=192, freq='15min', name='datetime')
        cutoff = future(origins[:1], -1)[0]
        fold_raw = raw.loc[:origins[-1]]
        tx = CompactInputTransform().fit(fold_raw.loc[:cutoff])
        x = tx.transform(fold_raw)
        members = {s['name']: joblib.load(ROOT / DIRS.get(s['kind'], 'artifacts/compact_v5')
                                           / f"fold_{fold}_{s['name']}.joblib")['predictions']
                   for s in BASELINE['specs']}
        for t in TARGETS:
            now = x.loc[origins, t].to_numpy(dtype=float)
            members['persistence'] = {**members.get('persistence', {})}
            members['ema8'] = members.get('ema8', {})
            members['ema16'] = members.get('ema16', {})
            for h in HORIZONS:
                members['persistence'][t, h] = now.copy()
                members['ema8'][t, h] = x.loc[origins, f'feat_{t}_ema_8'].to_numpy(dtype=float)
                members['ema16'][t, h] = x.loc[origins, f'feat_{t}_ema_16'].to_numpy(dtype=float)
        for t in TARGETS:
            for h in HORIZONS:
                times = future(origins, h)
                y = fold_raw[t].reindex(times).to_numpy(dtype=float).copy()
                y[times > origins[-1]] = np.nan
                records.append(dict(target=t, h=h, y=y,
                                    p=np.column_stack([members[n][t, h] for n in names])))
    weights = {}
    report = {}
    for t in TARGETS:
        for b in ('15_30', '45_120'):
            rr = [r for r in records if r['target'] == t and band(r['h']) == b]
            y = np.concatenate([r['y'] for r in rr])
            p = np.concatenate([r['p'] for r in rr])
            raw_w = convex_weights(y, p)
            capped_w = capped(raw_w)
            weights[f'{t}/{b}'] = dict(zip(names, capped_w.tolist()))
            report[f'{t}/{b}'] = dict(
                raw={n: round(float(v), 4) for n, v in zip(names, raw_w) if v > 1e-8},
                capped={n: round(float(v), 4) for n, v in zip(names, capped_w) if v > 1e-8})
    selection = dict(BASELINE)
    selection.update(name='v15_capped_pool', weights=weights,
                     specs=BASELINE['specs'] + EXTRA_SPECS,
                     family_selection={'generator_1': {'pool': 'active8+persistence+ema8+ema16', 'cap': CAP},
                                       'generator_all': {'pool': 'active8 only (extra members carry zero weight)', 'cap': CAP}},
                     protocol=('V10 hyperparameters, transform and member implementations unchanged. '
                               'Blend weights refit on the four April tuning folds with each member weight capped '
                               'at 0.25; generator_1 additionally blends persistence/EMA members. Chosen from an April-only '
                               'bake-off of combination schemes and pools; no May label used for any choice.'))
    dump_json(OUT / 'selection.json', selection)
    print(json.dumps(report, indent=2), flush=True)
    print('frozen:', OUT / 'selection.json', flush=True)


if __name__ == '__main__':
    main()
