import joblib
import numpy as np
import pandas as pd
import pytest
from prelim import Spec
from compact_inputs import CompactInputTransform
from online_model import OnlineRidge


def data():
    ix=pd.date_range('2025-01-01',periods=1000,freq='15min',name='datetime')
    a=np.arange(len(ix))
    return pd.DataFrame({'generator_1':80+8*np.sin(a/12),
                         'generator_all':240+20*np.cos(a/15)},index=ix)


def test_online_updates_never_use_later_labels_and_replay_csv_history(tmp_path):
    raw=data()
    cutoff=raw.index[800]
    tx=CompactInputTransform().fit(raw.iloc[:801])
    x=tx.transform(raw)
    spec=Spec('online21_2h','online',days=6,alpha=100.)
    model=OnlineRidge(spec).fit(raw,x,cutoff)
    assert model.history_raw.index.max()==cutoff
    ix=raw.index[801:841]
    a=model.predict(raw,x,ix)
    for row in model.update_audit:
        assert pd.Timestamp(row['last_label'])<pd.Timestamp(row['update_origin'])
    changed=raw.copy()
    changed.iloc[822:]=1e7
    z=tx.transform(changed)
    b=model.predict(changed,z,ix)
    for key in a:
        np.testing.assert_allclose(a[key][:21],b[key][:21],rtol=0,atol=1e-9)
    path=tmp_path/'online.joblib'
    joblib.dump(model,path)
    restored=joblib.load(path)
    c=restored.predict(raw.loc[:ix[-1]],x.loc[ix],ix)
    prefix=restored.predict(raw.loc[:ix[20]],x.loc[ix[:21]],ix[:21])
    for key in a:
        np.testing.assert_allclose(a[key],c[key],rtol=0,atol=1e-9)
        np.testing.assert_allclose(a[key][:21],prefix[key],rtol=0,atol=1e-9)


def test_online_rejects_missing_replay_observations():
    raw=data()
    tx=CompactInputTransform().fit(raw.iloc[:801])
    x=tx.transform(raw)
    model=OnlineRidge(Spec('online21_2h','online',days=6,alpha=100.)).fit(raw,x,raw.index[800])
    with pytest.raises(ValueError,match='Missing arrived observations'):
        model.predict(raw,x.loc[raw.index[820:830]],raw.index[820:830])
