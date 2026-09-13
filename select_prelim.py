"""Merge cached April challengers; freeze weights before any May evaluation."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prelim import SPECS, TARGETS, HORIZONS, read_data, future, fit_weights, band, score, dump_json
from linear_challengers import EXTRA_SPECS


CALIBRATIONS = {name+'_cal': dict(base=name, span=16, gain=.5)
                for name in ('ridge21', 'ridge60', 'lgb60', 'ridge21_strong')}


def causal_calibrate(predictions, x, origins, span=16, gain=.5, observations=None):
    """An error matures at origin+h, not when a forecast is issued.

    Only errors from forecasts issued inside the requested deployment interval
    are eligible. The first h origins start uncalibrated (no in-sample residuals).
    This is a stateful deployment procedure: replay from the same first origin.
    """
    result = {}
    for t in TARGETS:
        observed = (x if observations is None else observations).loc[origins, t]
        observed = observed.where((observed >= 0) & (observed <= (200 if t == TARGETS[0] else 440)))
        for h in HORIZONS:
            previous = pd.Series(predictions[t, h], index=future(origins, h)).reindex(origins)
            error = observed-previous
            adjustment = error.ewm(span=span, adjust=False, min_periods=1).mean().fillna(0)
            # Avoid committing a full initial observation error as the initial bias.
            count = error.notna().cumsum().to_numpy()
            warmup = np.minimum(count/span, 1.)
            result[t, h] = np.clip(predictions[t, h]+gain*warmup*adjustment.to_numpy(),
                                   0, 200 if t == TARGETS[0] else 440)
    return result


def add_calibrations(predictions, x, origins, calibrations, observations=None):
    result = dict(predictions)
    for name, cfg in calibrations.items():
        result[name] = causal_calibrate(result[cfg['base']], x, origins, cfg['span'], cfg['gain'], observations)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir', type=Path, required=True)
    p.add_argument('--base-dir', type=Path, default=Path('artifacts/prelim_v2'))
    p.add_argument('--linear-dir', type=Path, default=Path('artifacts/prelim_linear'))
    p.add_argument('--output-dir', type=Path, default=Path('artifacts/prelim_selected'))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_data(args.train_dir)
    raw = raw.loc[raw.index < '2025-05-01']
    folds = ['2025-04-21', '2025-04-23', '2025-04-25', '2025-04-29']
    specs = list(SPECS)+list(EXTRA_SPECS)
    names = [s.name for s in specs]+list(CALIBRATIONS)
    records, metrics = [], []
    from prelim import InputTransform
    for fold in folds:
        origins = pd.date_range(fold, periods=192, freq='15min', name='datetime')
        tx = InputTransform().fit(raw.loc[raw.index < origins.min()])
        x = tx.transform(raw)
        predictions = {}
        for spec in specs:
            root = args.base_dir if spec in SPECS else args.linear_dir
            predictions[spec.name] = joblib.load(root/f'fold_{fold}_{spec.name}.joblib')['predictions']
        predictions = add_calibrations(predictions, x, origins, CALIBRATIONS, raw)
        for t in TARGETS:
            for h in HORIZONS:
                y = raw[t].reindex(future(origins, h)).to_numpy()
                matrix = np.column_stack([predictions[n][t,h] for n in names])
                records.append(dict(fold=fold, target=t, h=h, y=y, p=matrix))
                for i, name in enumerate(names):
                    metrics.append(dict(fold=fold, target=t, horizon=h, member=name, mape=score(y, matrix[:,i])))
    tuning = [r for r in records if r['fold'] != folds[-1]]
    weights = fit_weights(tuning, names)
    for fold in folds:
        ww = weights if fold == folds[-1] else fit_weights([r for r in tuning if r['fold'] != fold], names)
        for r in [r for r in records if r['fold'] == fold]:
            w = np.array([ww[f"{r['target']}/{band(r['h'])}"][n] for n in names])
            metrics.append(dict(fold=fold, target=r['target'], horizon=r['h'], member='ensemble_heldout', mape=score(r['y'], r['p']@w)))
    # Complexity guard: extra candidate families must improve leave-fold-out
    # development MAPE by >=0.5% relative, not merely training-blend MAPE.
    # The final April holdout is explicitly excluded from this decision.
    base_names = [s.name for s in SPECS]
    base_records = [{**r, 'p': r['p'][:, :len(base_names)]} for r in records]
    base_tuning = [r for r in base_records if r['fold'] != folds[-1]]
    base_weights = fit_weights(base_tuning, base_names)
    base_metrics = []
    for fold in folds:
        ww = base_weights if fold == folds[-1] else fit_weights([r for r in base_tuning if r['fold'] != fold], base_names)
        for r in [r for r in base_records if r['fold'] == fold]:
            w = np.array([ww[f"{r['target']}/{band(r['h'])}"][n] for n in base_names])
            base_metrics.append(dict(fold=fold, target=r['target'], horizon=r['h'], member='ensemble_base', mape=score(r['y'], r['p']@w)))
    base_cv = np.mean([r['mape'] for r in base_metrics if r['fold'] != folds[-1]])
    expanded_cv = np.mean([r['mape'] for r in metrics if r['member'] == 'ensemble_heldout' and r['fold'] != folds[-1]])
    use_expanded = expanded_cv < base_cv*.995
    for r in metrics:
        if r['member'] == 'ensemble_heldout':
            r['member'] = 'ensemble_expanded'
    metrics.extend(base_metrics)
    chosen_member = 'ensemble_expanded' if use_expanded else 'ensemble_base'
    metrics.extend([{**r, 'member': 'ensemble_selected'} for r in metrics if r['member'] == chosen_member])
    if not use_expanded:
        names, specs, weights = base_names, list(SPECS), base_weights
    selection = dict(version=1, train_only=True, names=names, specs=[asdict(s) for s in specs],
        weights=weights, calibrations=CALIBRATIONS if use_expanded else {}, tuning_cutoffs=folds[:-1], holdout_cutoff=folds[-1],
        family_selection=dict(base_leave_fold_out_mape=base_cv, expanded_leave_fold_out_mape=expanded_cv,
                              minimum_relative_gain=.005, selected=chosen_member, holdout_used=False),
        protocol='Frozen parameters. 15-minute rolling observations. Bias correction uses only matured past forecasts.')
    dump_json(args.output_dir/'selection.json', selection)
    table = pd.DataFrame(metrics)
    table.to_csv(args.output_dir/'development_metrics.csv', index=False)
    joblib.dump(records, args.output_dir/'oof_predictions.joblib')
    print(table.groupby(['fold','target','member']).mape.mean().unstack().round(5).to_string(), flush=True)
    print({k: {n:round(w,4) for n,w in v.items() if w>1e-6} for k,v in weights.items()}, flush=True)


if __name__ == '__main__':
    main()
