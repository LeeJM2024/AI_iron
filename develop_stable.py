"""Compare v2/stable-input preprocessing using April-only out-of-time labels."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from prelim import (SPECS, Spec, TARGETS, HORIZONS, ShortModel, read_data, future,
                    score, fit_weights, band, dump_json)
from stable_inputs import StableInputTransform


FOLDS = ['2025-04-21', '2025-04-23', '2025-04-25', '2025-04-27', '2025-04-29']
# Ridge tests run in seconds; tree fits use the same proven model settings.
SPECS_STABLE = [s for s in SPECS if s.name in ('persistence', 'ridge21', 'ridge60', 'lgb60', 'cat21')]


def run(args):
    raw = read_data(args.train_dir)
    raw = raw.loc[raw.index < '2025-05-01']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = [s for s in SPECS_STABLE if not args.members or s.name in args.members]
    names = [s.name for s in specs]
    records, metrics = [], []
    config = dict(active_days=7, robust_raw=args.robust_raw)
    fingerprint = hashlib.sha256(Path('stable_inputs.py').read_bytes()+Path('prelim.py').read_bytes()+
        pd.util.hash_pandas_object(raw).values.tobytes()+json.dumps(config).encode()).hexdigest()
    for fold in FOLDS:
        cutoff = future([pd.Timestamp(fold)], -1)[0]
        origins = pd.date_range(fold, periods=192, freq='15min', name='datetime')
        tx = StableInputTransform(**config).fit(raw.loc[:cutoff])
        x = tx.transform(raw)
        predictions = {}
        for spec in specs:
            path = args.output_dir/f'fold_{fold}_{spec.name}.joblib'
            saved = joblib.load(path) if path.exists() else None
            if saved and saved.get('fingerprint') == fingerprint and saved.get('spec') == asdict(spec):
                pred = saved['predictions']
            else:
                tic = time.perf_counter()
                m = ShortModel(spec, args.threads).fit(raw, x, cutoff)
                pred = m.predict(raw, x, origins)
                joblib.dump(dict(fingerprint=fingerprint, spec=asdict(spec), predictions=pred, audit=m.audit), path)
                print(f'{fold} {spec.name} {time.perf_counter()-tic:.1f}s', flush=True)
            predictions[spec.name] = pred
        for t in TARGETS:
            for h in HORIZONS:
                y = raw[t].reindex(future(origins, h)).to_numpy()
                y[future(origins,h) > origins.max()] = np.nan
                p = np.column_stack([predictions[n][t,h] for n in names])
                records.append(dict(fold=fold, target=t, h=h, y=y, p=p))
                for i,n in enumerate(names):
                    metrics.append(dict(fold=fold, target=t, horizon=h, member=n, mape=score(y,p[:,i])))
        print(pd.DataFrame(metrics).query('fold == @fold').groupby(['target','member']).mape.mean().unstack().round(5).to_string(), flush=True)
    tuning = [r for r in records if r['fold'] != FOLDS[-1]]
    weights = fit_weights(tuning, names)
    for fold in FOLDS:
        w = weights if fold == FOLDS[-1] else fit_weights([r for r in tuning if r['fold'] != fold], names)
        for r in [r for r in records if r['fold'] == fold]:
            ww = np.array([w[f"{r['target']}/{band(r['h'])}"][n] for n in names])
            metrics.append(dict(fold=fold, target=r['target'], horizon=r['h'], member='ensemble_heldout', mape=score(r['y'],r['p']@ww)))
    table = pd.DataFrame(metrics)
    table.to_csv(args.output_dir/'development_metrics.csv', index=False)
    joblib.dump(records, args.output_dir/'oof_predictions.joblib')
    selection = dict(version=1, train_only=True, names=names, specs=[asdict(s) for s in specs],
        weights=weights, calibrations={}, transform=dict(kind='stable', **config),
        tuning_cutoffs=FOLDS[:-1], holdout_cutoff=FOLDS[-1],
        protocol='April-only parameter selection. Frozen active schema. Rolling 15-minute observations.')
    dump_json(args.output_dir/'selection.json', selection)
    print(table.groupby(['fold','target','member']).mape.mean().unstack().round(5).to_string(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path('artifacts/stable_v3'))
    p.add_argument('--robust-raw', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--members', nargs='+')
    run(p.parse_args())


if __name__ == '__main__':
    main()
