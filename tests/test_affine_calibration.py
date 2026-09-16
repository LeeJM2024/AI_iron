import numpy as np
import pytest
from affine_score_calibration import AffineMix

def test_offset_only_recovers_shared_bias():
    y=np.linspace(80,120,100)
    p=np.column_stack([y-2,y-3])
    model=AffineMix('offset').fit(p,y)
    np.testing.assert_allclose(model.predict(p),y,atol=1e-7)
    assert abs(model.weight_sum-1)<1e-8
    assert (model.weights>=0).all()

def test_inference_replays_from_parameters_without_labels():
    import json
    y=np.linspace(200,300,100)
    p=np.column_stack([y*.97,y*.94])
    model=AffineMix('scale_offset').fit(p,y)
    data=json.loads(json.dumps(model.manifest(['a','b'])))
    replay=p@np.array([data['weights'][n] for n in ['a','b']])+data['bias']
    np.testing.assert_allclose(replay,model.predict(p),rtol=0,atol=1e-10)
    assert .9-1e-8<=model.weight_sum<=1.1+1e-8

def test_missing_truths_are_not_fabricated():
    y=np.linspace(100,200,100)
    y[:7]=np.nan
    p=np.column_stack([np.linspace(101,201,100),np.linspace(99,199,100)])
    model=AffineMix().fit(p,y)
    assert model.training_samples==93
    with pytest.raises(ValueError,match='Insufficient'):
        AffineMix().fit(p[:10],np.full(10,np.nan))
