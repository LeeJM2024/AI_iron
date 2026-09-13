"""Short-period contract, causal feature, label-purge and serialization tests."""
from pathlib import Path
import tempfile
from dataclasses import asdict
from types import SimpleNamespace
import json
import zipfile
import pytest

import joblib
import numpy as np
import pandas as pd

from prelim import (InputTransform, ShortModel, Spec, convex_weights, score,
                    ensemble_predictions, TARGETS, HORIZONS)
from run_prelim import csv_frame, quality_audit, run
from select_prelim import causal_calibrate


def example():
    idx = pd.date_range('2025-01-01', periods=1000, freq='15min', name='datetime')
    a = np.arange(len(idx))
    raw = pd.DataFrame({'generator_1': 80+10*np.sin(a/20),
        'generator_all': 240+20*np.sin(a/25), 'blast_furnace_1': 2e5+1e4*np.sin(a/10),
        'generator_use_coke_gas': 5000+1000*np.cos(a/20),
        'blast_furnace_3': np.nan, 'constant_sensor': 2.}, index=idx)
    raw.iloc[200, 0] = np.nan
    return raw


def test_schema_removes_unavailable_not_fabricates_measurements():
    raw = example()
    transform = InputTransform().fit(raw.iloc[:700])
    x = transform.transform(raw)
    assert transform.excluded == {'blast_furnace_3': 'all_missing', 'constant_sensor': 'constant'}
    assert not any('blast_furnace_3' in c for c in x)
    assert np.isfinite(x.to_numpy()).all()
    assert raw.iloc[200, 0] != raw.iloc[200, 0]  # raw label stays NaN


def test_future_does_not_change_inputs_or_training():
    raw = example()
    cutoff = raw.index[700]
    changed = raw.copy()
    changed.iloc[701:] = 999999.
    a = InputTransform().fit(raw.loc[:cutoff])
    b = InputTransform().fit(changed.loc[:cutoff])
    x, z = a.transform(raw), b.transform(changed)
    pd.testing.assert_frame_equal(x.loc[:cutoff], z.loc[:cutoff])
    cfg = Spec('test_ridge', 'ridge', days=6)
    ma = ShortModel(cfg, 1).fit(raw, x, cutoff)
    mb = ShortModel(cfg, 1).fit(changed, z, cutoff)
    assert all(pd.Timestamp(r['last_label']) <= cutoff for r in ma.audit)
    pa = ma.predict(raw, x, pd.DatetimeIndex([cutoff]))
    pb = mb.predict(changed, z, pd.DatetimeIndex([cutoff]))
    for k in pa:
        np.testing.assert_allclose(pa[k], pb[k], atol=1e-12, rtol=0)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)/'model.joblib'
        joblib.dump(ma, path)
        reloaded = joblib.load(path)
        pc = reloaded.predict(raw, x, pd.DatetimeIndex([cutoff]))
        for k in pa:
            np.testing.assert_array_equal(pa[k], pc[k])


def test_sparse_convex_blend_and_missing_metric():
    y = np.array([10., 20., 30., np.nan, 0.])
    p = np.column_stack([np.ones(5)*10, np.array([10., 20., 30., 7., 8.])])
    w = convex_weights(y, p)
    np.testing.assert_allclose(w, [0, 1], atol=1e-9)
    assert score(y, p@w) == 0.


def test_zip_input_contract_and_target_order():
    raw = example()
    tx = InputTransform().fit(raw.iloc[:700])
    x = tx.transform(raw)
    origins = raw.index[700:710]
    model = ShortModel(Spec('persistence', 'last')).fit(raw, x, raw.index[699])
    weights = {f'{t}/{b}': {'persistence': 1.} for t in TARGETS for b in ('15_30', '45_120')}
    p = ensemble_predictions({'persistence': model.predict(raw, x, origins)}, weights)
    prediction = csv_frame(pd.DataFrame(p, index=origins))
    audit = quality_audit(csv_frame(x.loc[origins]), prediction, origins, tx)
    assert len(prediction.columns) == 1+2*len(HORIZONS)
    assert audit['input']['missing_cells'] == 0
    assert audit['official_quality_score'] is None


def test_sensor_spike_is_input_only_and_no_future_fill():
    raw = example()
    tx = InputTransform().fit(raw.iloc[:700])
    raw.iloc[750, raw.columns.get_loc('generator_all')] = 1e8
    raw.iloc[751, raw.columns.get_loc('generator_all')] = 300.
    x = tx.transform(raw)
    assert x.iloc[750].generator_all == x.iloc[749].generator_all
    assert x.iloc[751].generator_all == 300.
    assert raw.iloc[750].generator_all == 1e8


def test_calibration_only_uses_matured_forecasts():
    raw = example()
    ix = raw.index[:100]
    p = {(t,h): np.full(100, 80. if t == TARGETS[0] else 240.) for t in TARGETS for h in HORIZONS}
    a = causal_calibrate(p, raw, ix)
    changed = raw.copy()
    changed.iloc[51:] = 1e6
    b = causal_calibrate(p, changed, ix)
    for t in TARGETS:
        for h in HORIZONS:
            np.testing.assert_array_equal(a[t,h][:51], b[t,h][:51])
            np.testing.assert_array_equal(a[t,h][:h], p[t,h][:h])
    prefix = causal_calibrate({k: v[:51] for k,v in p.items()}, raw.iloc[:51], ix[:51])
    for k in a:
        np.testing.assert_array_equal(a[k][:51], prefix[k])


@pytest.mark.parametrize('variant', ['v2', 'stable', 'pooled', 'compact'])
def test_end_to_end_submission_replays_exported_inputs(tmp_path, variant):
    raw = example()[list(TARGETS)]
    train_dir, test_dir, out = tmp_path/'train', tmp_path/'test', tmp_path/'output'
    train_dir.mkdir()
    test_dir.mkdir()
    csv_frame(raw.iloc[:700]).to_csv(train_dir/'Pre_load.csv', index=False)
    csv_frame(raw.iloc[700:]).to_csv(test_dir/'Pre_test_load.csv', index=False)
    spec = Spec('pooled_relative','pooled',days=6,trees=3,half_life=14.) if variant=='pooled' else Spec('ridge21', 'ridge', days=6, alpha=30.)
    weights = {f'{t}/{b}': {spec.name: 1.} for t in TARGETS for b in ('15_30', '45_120')}
    selection = tmp_path/'selection.json'
    payload = dict(version=1, train_only=True, specs=[asdict(spec)], weights=weights)
    if variant != 'v2':
        payload['transform'] = dict(kind='stable', active_days=3, robust_raw=True)
    if variant in ('compact','dynamics'):
        payload['transform'] = dict(kind='compact', active_days=3, robust_raw=True)
    selection.write_text(json.dumps(payload), encoding='utf-8')
    args = SimpleNamespace(output_dir=out, selection=selection, train_dir=train_dir,
                           test_dir=test_dir, threads=1, compare=[], reuse_model=False)
    run(args)
    audit = json.loads((out/'quality_audit.json').read_text(encoding='utf-8'))
    assert audit['submitted_input_replay'].startswith('passed')
    with zipfile.ZipFile(out/'LeeJM_gas_predict_prelim.zip') as z:
        assert z.namelist() == ['input.csv', 's_result.csv']
        assert z.testzip() is None
    first = pd.read_csv(out/'s_result.csv')
    args.reuse_model = True
    run(args)
    pd.testing.assert_frame_equal(first, pd.read_csv(out/'s_result.csv'))
    if variant == 'stable':
        args.reuse_model = False
        args.warm_start_model = out/'prelim_model.joblib'
        args.output_dir = tmp_path/'warm_output'
        run(args)
        pd.testing.assert_frame_equal(first, pd.read_csv(args.output_dir/'s_result.csv'))
        meta = json.loads((args.output_dir/'run_metadata.json').read_text(encoding='utf-8'))
        assert meta['warm_start_members'] == [spec.name]
        # Any training-source mutation must prevent cached-member reuse.
        f = pd.read_csv(train_dir/'Pre_load.csv')
        f.loc[0,'generator_1'] += 1
        f.to_csv(train_dir/'Pre_load.csv',index=False)
        with pytest.raises(ValueError,match='training data hash mismatch'):
            run(args)
