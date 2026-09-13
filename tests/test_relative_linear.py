import joblib
import numpy as np
import pandas as pd
import pytest
from prelim import Spec
from compact_inputs import CompactInputTransform
from relative_linear import RelativeLinear,features


@pytest.mark.parametrize('mae',[False,True])
def test_relative_regression_is_causal_and_serializable(mae,tmp_path):
    ix=pd.date_range('2025-01-01',periods=1000,freq='15min',name='datetime')
    n=np.arange(1000)
    raw=pd.DataFrame({'generator_1':80+10*np.sin(n/21),
        'generator_all':240+20*np.cos(n/21)},index=ix)
    cutoff=ix[800]
    tx=CompactInputTransform().fit(raw.iloc[:801])
    x=tx.transform(raw)
    altered=raw.copy()
    altered.iloc[801:]=1e8
    spec=Spec('relative60_mae' if mae else 'relative60','relative_linear',days=6,alpha=100.,half_life=14.)
    a=RelativeLinear(spec).fit(raw,x,cutoff)
    b=RelativeLinear(spec).fit(altered,tx.transform(altered),cutoff)
    p=a.predict(raw,x,ix[800:805])
    q=b.predict(altered,tx.transform(altered),ix[800:801])
    for k in p:
        np.testing.assert_allclose(p[k][0],q[k][0],rtol=0,atol=1e-10)
    assert all(pd.Timestamp(r['last_label'])<=cutoff for r in a.audit)
    path=tmp_path/'linear.joblib'
    joblib.dump(a,path)
    r=joblib.load(path).predict(raw,x.loc[ix[800:805]],ix[800:805])
    for k in p:
        np.testing.assert_allclose(p[k],r[k],rtol=0,atol=1e-10)
