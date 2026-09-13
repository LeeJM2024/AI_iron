import numpy as np
import pandas as pd
import pytest
from stable_inputs import StableInputTransform
from prelim import TARGETS


def data():
    ix = pd.date_range('2025-01-01', periods=1300, freq='15min', name='datetime')
    n = np.arange(len(ix))
    return pd.DataFrame({'generator_1': 80+10*np.sin(n/20),
        'generator_all': 240+20*np.cos(n/20),
        'gas_sensor': 10000+1000*np.sin(n/10),
        'inactive_sensor': np.r_[np.arange(100),np.zeros(1200)],
        'unobserved_sensor': np.nan}, index=ix)


@pytest.mark.parametrize('robust', [False, True])
def test_active_schema_and_future_invariance(robust):
    raw = data()
    before = raw.copy()
    tx = StableInputTransform(robust_raw=robust).fit(raw.iloc[:900])
    assert 'inactive_sensor' in tx.excluded
    assert 'unobserved_sensor' in tx.excluded
    full = tx.transform(raw)
    changed = raw.copy()
    changed.iloc[1001:] = 1e9
    future_changed = tx.transform(changed)
    pd.testing.assert_frame_equal(full.iloc[:1001], future_changed.iloc[:1001])
    pd.testing.assert_frame_equal(full.iloc[:1001], tx.transform(raw.iloc[:1001]))
    pd.testing.assert_frame_equal(raw, before)
    assert np.isfinite(full.to_numpy()).all()
    assert (full.to_numpy() >= 0).all()
    assert not any('inactive_sensor' in c for c in full.columns)
    for t in TARGETS:
        np.testing.assert_allclose(full[t],raw[t],rtol=1e-6)


def test_robust_input_clips_sensor_spikes_not_labels():
    raw = data()
    tx = StableInputTransform(robust_raw=True).fit(raw.iloc[:900])
    raw.loc[raw.index[1000],'gas_sensor'] = 1e9
    result = tx.transform(raw)
    assert result.loc[raw.index[1000],'gas_sensor'] < 12000
    assert raw.loc[raw.index[1000],'gas_sensor'] == 1e9
    # New features encode direction around .5 without negative values.
    change = result['feat_generator_1_change_4']
    assert change.between(0,1).all()
    assert change.min()<.5<change.max()


def test_constant_targets_retained_and_invalid_window_fails():
    with pytest.raises(ValueError):
        StableInputTransform(active_days=0)
    raw = data()
    raw['generator_1'] = 80.
    raw['generator_all'] = 80.
    tx = StableInputTransform().fit(raw)
    assert set(TARGETS).issubset(tx.feature_columns)


def test_local_ridge_weights_follow_only_current_state():
    from regime_ridge import LocalRidge
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300,5))
    y = np.where(x[:,0]<0, x[:,1], -x[:,1])
    m = LocalRidge(alpha=1.,state_index=0,bandwidth=.5).fit(x,y,np.ones(300))
    p = m.predict(x)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(m.predict(x[:10]),p[:10],atol=1e-12)
