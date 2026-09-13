import joblib
import numpy as np
import pandas as pd
import pytest
from prelim import Spec, TARGETS
from stable_inputs import StableInputTransform
from pooled_short import PooledShort, augment


@pytest.mark.parametrize('relative',[False,True])
def test_shared_horizons_purge_labels_and_replay(tmp_path,relative):
    ix=pd.date_range('2025-01-01',periods=1000,freq='15min',name='datetime')
    n=np.arange(len(ix))
    raw=pd.DataFrame({'generator_1':80+10*np.sin(n/20),
                     'generator_all':240+20*np.cos(n/20)},index=ix)
    cutoff=ix[800]
    tx=StableInputTransform().fit(raw.iloc[:801])
    x=tx.transform(raw)
    name='pooled_relative' if relative else 'pooled60'
    spec=Spec(name,'pooled',days=7,trees=3,half_life=14.)
    model=PooledShort(spec,1).fit(raw,x,cutoff)
    assert all(pd.Timestamp(a['last_label'])<=cutoff for a in model.audit)
    pred=model.predict(raw,x,ix[800:805])
    changed=raw.copy()
    changed.iloc[801:]=99999.
    changed_x=tx.transform(changed)
    other=PooledShort(spec,1).fit(changed,changed_x,cutoff)
    mutated=other.predict(changed,changed_x,ix[800:801])
    for k in pred:
        np.testing.assert_allclose(pred[k][:1],mutated[k],atol=1e-10,rtol=0)
    path=tmp_path/'model.joblib'
    joblib.dump(model,path)
    replay=joblib.load(path).predict(raw,x,ix[800:805])
    for k in pred:
        np.testing.assert_array_equal(pred[k],replay[k])
    for h in (1,8):
        pd.testing.assert_frame_equal(augment(x.iloc[800:801],h,TARGETS[0]),
                                      augment(changed_x.iloc[800:801],h,TARGETS[0]))
