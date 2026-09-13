"""Train frozen short-period models, export and verify a two-file contest ZIP."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import zipfile

import joblib
import numpy as np
import pandas as pd

from prelim import (InputTransform, ShortModel, Spec, TARGETS, HORIZONS, UPPER,
                    read_data, future, score, ensemble_predictions, dump_json)
from linear_challengers import fit_candidate
from select_prelim import add_calibrations


def replay_artifact(artifact, raw, x, origins):
    members = {n: m.predict(raw, x, origins) for n, m in artifact['models'].items()}
    members = add_calibrations(members, x, origins, artifact.get('calibrations', {}), raw)
    return pd.DataFrame(ensemble_predictions(members, artifact['weights']), index=origins)


def csv_frame(frame):
    result = frame.copy()
    result.index.name = 'datetime'
    result = result.reset_index()
    result['datetime'] = pd.to_datetime(result.datetime).dt.strftime('%Y-%m-%d %H:%M:%S')
    return result


def atomic_csv(frame, path):
    temporary = path.with_suffix('.csv.tmp')
    # Round-trip precision also preserves tiny Fourier features around zero.
    # Fixed decimal rounding can cross a tree split even with a tiny value delta.
    frame.to_csv(temporary, index=False, encoding='utf-8', float_format='%.17g')
    temporary.replace(path)


def quality_audit(inputs, predictions, origins, transform):
    expected = ['datetime']+[f'{t}_t+{15*h}_pred' for t in TARGETS for h in HORIZONS]
    if list(predictions.columns) != expected:
        raise ValueError('Incorrect prediction schema/order')
    audit = {}
    for name, table in [('input', inputs), ('s_result', predictions)]:
        numeric = table.drop(columns='datetime')
        dt = pd.DatetimeIndex(pd.to_datetime(table.datetime), name='datetime')
        if not dt.equals(origins):
            raise ValueError(f'{name}: missing, duplicate or misaligned origins')
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f'{name}: non-finite value')
        audit[name] = dict(rows=len(table), numeric_columns=len(numeric.columns),
            missing_cells=int(numeric.isna().sum().sum()), duplicate_times=int(dt.duplicated().sum()),
            constant_columns=numeric.columns[numeric.nunique() <= 1].tolist())
    bad = [c for c in inputs if c != 'datetime' and c not in transform.columns and not c.startswith('feat_')]
    if bad:
        raise ValueError(f'Invalid input fields: {bad}')
    for t in TARGETS:
        values = predictions[[f'{t}_t+{15*h}_pred' for h in HORIZONS]].to_numpy()
        if (values < 0).any() or (values > UPPER[t]).any():
            raise ValueError('Prediction exceeds physical bounds')
    # Report statistical tail observations, do not claim an invented official score.
    x = inputs.drop(columns='datetime')
    z = (x-x.mean())/x.std().replace(0, np.nan)
    audit['diagnostic_abs_z_gt_3_cells'] = int(z.abs().gt(3).sum().sum())
    audit['diagnostic_abs_z_gt_3_by_column'] = {
        c: int(v) for c,v in z.abs().gt(3).sum().items() if v > 0}
    audit['negative_cells'] = int(x.lt(0).sum().sum())
    audit['duplicate_numeric_columns'] = x.columns[x.T.duplicated()].tolist()
    audit['excluded_training_columns'] = transform.excluded
    audit['raw_sensor_bounds'] = transform.bounds
    audit['official_quality_score'] = None
    audit['note'] = ('Official out/invalid_col thresholds are unavailable. Statistical tails and '
        'legitimate inactive sensors are reported, not relabeled to manufacture a perfect score.')
    return audit


def evaluate(raw, origins, candidates):
    records = []
    for name, predictions in candidates.items():
        for t in TARGETS:
            for h in HORIZONS:
                y = raw[t].reindex(future(origins, h)).to_numpy()
                p = predictions[f'{t}_t+{15*h}_pred'].to_numpy()
                ok = np.isfinite(y) & (np.abs(y) > 1e-8)
                records.append(dict(model=name, target=t, horizon_minutes=15*h,
                    mape=score(y, p), scored_pairs=int(ok.sum()), missing_pairs=int((~ok).sum())))
    return pd.DataFrame(records)


def run(args):
    tic = time.perf_counter()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    if selection.get('version') != 1 or not selection.get('train_only'):
        raise ValueError('Expected train-only preliminary model selection')
    # Read only test timestamps until parameters, feature schema and weights freeze.
    time_files = list(args.test_dir.glob('*load.csv'))
    if len(time_files) != 1:
        raise ValueError('Expected exactly one test load file')
    times = pd.to_datetime(pd.read_csv(time_files[0], usecols=['datetime']).datetime)
    origins = pd.DatetimeIndex(times.drop_duplicates().sort_values(), name='datetime')
    expected = pd.date_range(origins.min(), origins.max(), freq='15min', name='datetime')
    if not origins.equals(expected):
        raise ValueError('Test origin grid has gaps/off-grid timestamps; supply official origins')
    cutoff = origins.min()-pd.Timedelta(15, unit='min')
    train = read_data(args.train_dir).loc[:cutoff]
    reuse = None
    if getattr(args, 'reuse_model', False):
        # Only local, trusted joblib files from this run should ever be loaded.
        reuse = joblib.load(out/'prelim_model.joblib')
        if reuse['selection'] != selection or reuse['cutoff'] != cutoff:
            raise ValueError('Saved model selection/cutoff does not match requested run')
        previous_metadata = json.loads((out/'run_metadata.json').read_text(encoding='utf-8'))
        for directory in (args.train_dir, args.test_dir):
            for path in directory.glob('*.csv'):
                expected_hash = previous_metadata['source_hashes'].get(str(path.resolve()))
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError('Input data changed since model was fitted')
    if reuse:
        transformer = reuse['transform']
    elif selection.get('transform', {}).get('kind') == 'stable':
        from stable_inputs import StableInputTransform
        transform_config = {k: v for k,v in selection['transform'].items() if k != 'kind'}
        transformer = StableInputTransform(**transform_config).fit(train)
    else:
        transformer = InputTransform().fit(train)
    x_train = transformer.transform(train)
    warm = None
    warm_path = getattr(args, 'warm_start_model', None)
    if warm_path:
        warm_path = Path(warm_path).resolve()
        warm = joblib.load(warm_path)
        warm_metadata = json.loads((warm_path.parent/'run_metadata.json').read_text(encoding='utf-8'))
        if warm['cutoff'] != cutoff:
            raise ValueError('Warm-start model has a different training cutoff')
        if warm['selection'].get('transform') != selection.get('transform'):
            raise ValueError('Warm-start preprocessing differs')
        for path in args.train_dir.glob('*.csv'):
            if warm_metadata['source_hashes'].get(str(path.resolve())) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Warm-start training data hash mismatch')
        pd.testing.assert_frame_equal(warm['transform'].transform(train), x_train)
    weights = selection['weights']
    active = {n for w in weights.values() for n, v in w.items() if v > 1e-8}
    calibrations = {n: c for n,c in selection.get('calibrations', {}).items() if n in active}
    active |= {c['base'] for c in calibrations.values()}
    models, audits, reused_members = {}, [], []
    for config in selection['specs']:
        spec = Spec(**config)
        if spec.name not in active:
            continue
        start = time.perf_counter()
        if reuse:
            model = reuse['models'][spec.name]
        elif warm and spec.name in warm['models'] and warm['models'][spec.name].spec == spec:
            model = warm['models'][spec.name]
            reused_members.append(spec.name)
        elif spec.kind == 'pooled':
            from pooled_short import PooledShort
            model = PooledShort(spec,args.threads).fit(train,x_train,cutoff)
        elif spec.name.startswith('regime'):
            from regime_ridge import fit_regime
            model = fit_regime(spec, train, x_train, cutoff, args.threads)
        else:
            model = fit_candidate(spec, train, x_train, cutoff, args.threads)
        models[spec.name] = model
        audits.extend(model.audit)
        print(f'Final {spec.name}: {time.perf_counter()-start:.1f}s', flush=True)
    artifact = dict(transform=transformer, models=models, weights=weights, cutoff=cutoff, calibrations=calibrations,
                    feature_columns=transformer.feature_columns, selection=selection)
    joblib.dump(artifact, out/'prelim_model.joblib', compress=3)
    training_seconds = time.perf_counter()-tic
    # Parameters now frozen. May observations can only enter their own/past origins.
    test = read_data(args.test_dir)
    combined = pd.concat([train, test.loc[test.index > cutoff]]).sort_index()
    if combined.index.has_duplicates:
        raise ValueError('Unexpected duplicate train/test times')
    combined = combined.reindex(pd.date_range(combined.index.min(), combined.index.max(), freq='15min', name='datetime'))
    x = transformer.transform(combined)
    infer = time.perf_counter()
    pred = replay_artifact(artifact, combined, x, origins)
    inference_seconds = time.perf_counter()-infer
    inputs = csv_frame(x.loc[origins])
    predictions = csv_frame(pred)
    audit = quality_audit(inputs, predictions, origins, transformer)
    audit['input_preprocessing'] = selection.get('transform', {'kind': 'v2'})
    audit['changed_raw_cells_by_column'] = {
        c: int((test[c].loc[origins].notna().to_numpy() &
                ~np.isclose(test[c].loc[origins].to_numpy(), x.loc[origins,c].to_numpy(),
                            rtol=1e-6, atol=1e-6)).sum())
        for c in transformer.columns if c in x.columns and c in test.columns}
    atomic_csv(inputs, out/'input.csv')
    atomic_csv(predictions, out/'s_result.csv')
    # Verify serialization, prediction replay, and exact future-data invariance.
    saved = joblib.load(out/'prelim_model.joblib')
    check_origins = origins[[0, len(origins)//2, -1]]
    replay = replay_artifact(saved, combined, x, origins)
    np.testing.assert_allclose(replay.to_numpy(), pred.to_numpy(), rtol=0, atol=1e-8)
    for origin in check_origins:
        prefix = combined.loc[:origin]
        prefix_x = saved['transform'].transform(prefix)
        # .loc[[t]] drops pandas frequency metadata; actual timestamps/values
        # must match exactly, but a missing freq annotation is not data leakage.
        pd.testing.assert_frame_equal(prefix_x.tail(1), x.loc[[origin]], check_freq=False)
        prefix_origins = origins[origins <= origin]
        one = replay_artifact(saved, prefix, prefix_x, prefix_origins)
        np.testing.assert_allclose(one.tail(1).to_numpy(), pred.loc[[origin]].to_numpy(), rtol=0, atol=1e-8)
    read_inputs = pd.read_csv(out/'input.csv')
    read_predictions = pd.read_csv(out/'s_result.csv')
    quality_audit(read_inputs, read_predictions, origins, transformer)
    csv_x = read_inputs.set_index(pd.DatetimeIndex(pd.to_datetime(read_inputs.datetime), name='datetime'))
    csv_x = csv_x.drop(columns='datetime').astype('float32')
    csv_replay = replay_artifact(saved, combined, csv_x, origins)
    np.testing.assert_allclose(csv_replay.to_numpy(), read_predictions.drop(columns='datetime').to_numpy(), rtol=0, atol=1e-6)
    audit['serialization_replay'] = 'passed at all origins'
    audit['submitted_input_replay'] = 'passed: actual input.csv reproduces s_result.csv within 1e-6'
    audit['prefix_causality'] = 'passed: full history vs truncated history at 3 origins'
    pd.DataFrame(audits).to_csv(out/'training_audit.csv', index=False)
    dump_json(out/'quality_audit.json', audit)
    dump_json(out/'selection_frozen.json', selection)
    # ZIP has only the two files explicitly required by the published prelim rules.
    destination = out/'LeeJM_gas_predict_prelim.zip'
    temporary = destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in ('input.csv', 's_result.csv'):
            archive.write(out/filename, arcname=filename)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() or archive.namelist() != ['input.csv', 's_result.csv']:
            raise ValueError('Invalid submission archive')
    temporary.replace(destination)
    baseline = pd.DataFrame({f'{t}_t+{15*h}_pred': x.loc[origins, t].to_numpy()
                             for t in TARGETS for h in HORIZONS}, index=origins)
    candidates = {selection.get('name', 'prelim_v2'): pred, 'persistence': baseline}
    for path in args.compare:
        previous = pd.read_csv(path, index_col='datetime', parse_dates=True)
        if not origins.equals(pd.DatetimeIndex(previous.index, name='datetime')):
            raise ValueError(f'Comparison origin mismatch: {path}')
        candidates[str(path)] = previous
    scores = evaluate(combined, origins, candidates)
    scores.to_csv(out/'test_metrics.csv', index=False)
    summary = scores.groupby(['model', 'target']).mape.mean().unstack()
    summary.to_csv(out/'test_summary.csv')
    import lightgbm, catboost, sklearn, scipy
    metadata = dict(training_cutoff=str(cutoff), origin_start=str(origins.min()), origin_end=str(origins.max()),
        training_seconds=training_seconds, model_reused=bool(reuse), inference_seconds=inference_seconds,
        warm_start_members=reused_members,
        total_seconds=time.perf_counter()-tic, active_members=list(models),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      lightgbm=lightgbm.__version__, catboost=catboost.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__),
        source_hashes={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (args.train_dir, args.test_dir) for p in directory.glob('*.csv')},
        protocol='May labels NOT used for weights/hyperparameters. Rolling observations through each origin only.',
        selection_name=selection.get('name', 'prelim_v2'),
        code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('run_prelim.py','prelim.py','stable_inputs.py','regime_ridge.py','pooled_short.py','pipeline.py')
            if Path(__file__).with_name(name).exists()},
        metric='Mean of eight per-horizon MAPEs; nonzero observed raw labels only. Not official total score.',
        zip_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    dump_json(out/'run_metadata.json', metadata)
    print(summary.to_string(), flush=True)
    print(f'Verified submission: {destination}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir', type=Path, required=True)
    p.add_argument('--test-dir', type=Path, required=True)
    p.add_argument('--selection', type=Path, default=Path('artifacts/prelim_selected/selection.json'))
    p.add_argument('--output-dir', type=Path, default=Path('output/prelim_v2'))
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--compare', nargs='*', type=Path, default=[])
    p.add_argument('--reuse-model', action='store_true', help='Re-export a trusted saved model after verifying source hashes; no retraining')
    p.add_argument('--warm-start-model', type=Path, help='Reuse identical members from a trusted prior local artifact, after checking training hashes/config/features')
    run(p.parse_args())


if __name__ == '__main__':
    main()
