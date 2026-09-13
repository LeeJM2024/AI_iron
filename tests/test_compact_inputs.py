import numpy as np
import pandas as pd
import pytest
from compact_inputs import CompactInputTransform,CORE
from pooled_short import augment
from prelim import TARGETS


def sample():
    ix=pd.date_range('2025-01-01',periods=1000,freq='15min',name='datetime')
    n=np.arange(len(ix))
    return pd.DataFrame({'generator_1':80+8*np.sin(n/15),
        'generator_all':240+15*np.cos(n/15),
        'generator_use_blast_furnace_gas':800000+30000*np.sin(n/20),
        'blast_furnace_gas_holder_2':100000+5000*np.cos(n/10),
        'upstream_sensor':n},index=ix)


@pytest.mark.parametrize('dynamic,quantile',[(False,False),(True,False),(True,True)])
def test_compact_schema_causal_and_units(dynamic,quantile):
    raw=sample()
    original=raw.copy()
    tx=CompactInputTransform(dynamic_features=dynamic,quantile_features=quantile).fit(raw.iloc[:801])
    x=tx.transform(raw)
    assert list(x.columns)==tx.feature_columns
    assert 'upstream_sensor' not in x
    assert all(c in CORE or c.startswith('feat_') for c in x)
    assert np.isfinite(x.to_numpy()).all()
    altered=raw.copy()
    altered.iloc[901:]=99999999.
    pd.testing.assert_frame_equal(x.iloc[:901],tx.transform(altered).iloc[:901])
    pd.testing.assert_frame_equal(x.iloc[:901],tx.transform(raw.iloc[:901]))
    pd.testing.assert_frame_equal(raw,original)
    np.testing.assert_allclose(x.generator_1,raw.generator_1,rtol=1e-6)
    if quantile:
        assert all(c.endswith('_quantile') or c.startswith('feat_clock_') or c in CORE for c in x)
        z=augment(x.iloc[-5:],1,TARGETS[0])
        assert 'feat_relative_lag_1' not in z
