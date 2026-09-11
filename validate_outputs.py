"""Independent checks of saved CSVs, raw test labels, and resource conservation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiment import discover, read_observations
from dispatch import observed_surplus
from pipeline import GAS_TYPES, GasResourceForecaster, PriceSchedule, write_json
from forecasting import TARGETS, shifted, features


def validate(root, output):
    output=Path(output)
    metadata=json.loads((output/'run_metadata.json').read_text(encoding='utf-8'))
    directory=discover(root)
    test_files=list(directory.parent.rglob('Pre_test_load.csv'))
    assert len(test_files)==1, 'Ambiguous test data'
    raw_test=read_observations(test_files[0].parent)
    frames={name:pd.read_csv(output/(name+'.csv'),parse_dates=['datetime']).set_index('datetime')
        for name in ('input','s_result','l_result','opt_result','dispatch_audit')}
    for name,frame in frames.items():
        assert frame.index.is_unique and frame.index.is_monotonic_increasing, name
        assert np.isfinite(frame.to_numpy()).all(), name
    long=frames['l_result']; short=frames['s_result']; origins=long.index
    expected=[f'{t}_t+{h*15}_pred' for t in TARGETS for h in range(1,97)]
    assert list(long)==expected, 'Incorrect wide output schema'
    assert list(short)==[f'{t}_t+{h*15}_pred' for t in TARGETS for h in range(1,9)]
    assert len(origins)==metadata['origins'] and len(frames['opt_result'])==96
    pd.testing.assert_frame_equal(short,long[short.columns])
    assert origins.equals(frames['input'].index)
    assert all(c in raw_test or c.startswith('feat_') for c in frames['input'])
    assert all(c.startswith('opt_') for c in frames['opt_result'])
    payload=json.loads((output/'s_result.json').read_text(encoding='utf-8'))
    assert payload['columns']==['datetime',*short.columns]
    np.testing.assert_allclose(np.array(payload['data'],dtype=object)[:,1:].astype(float),short,atol=1e-6)
    audit=pd.read_csv(output/'training_audit.csv',parse_dates=['last_label','cutoff'])
    assert len(audit)==192 and (audit.last_label<=audit.cutoff).all()
    assert audit.cutoff.max()<origins.min()

    # Actuals come from the original test observations, never filled labels or input.csv.
    results=[]; errors={}; baseline={}
    current=raw_test.ffill().reindex(origins)
    for t in TARGETS:
        for h in range(1,97):
            y=raw_test[t].reindex(shifted(origins,h)).to_numpy()
            p=long[f'{t}_t+{15*h}_pred'].to_numpy()
            valid=np.isfinite(y)&(np.abs(y)>1e-8)
            assert valid.any(), 'No observed labels to score'
            error=np.full(len(y),np.nan); base=error.copy()
            error[valid]=np.abs(y[valid]-p[valid])/np.abs(y[valid])
            base[valid]=np.abs(y[valid]-current[t].to_numpy()[valid])/np.abs(y[valid])
            errors[t,h]=error; baseline[t,h]=base
            results.append(dict(target=t,horizon=h,mape=float(np.nanmean(error)),
                persistence_mape=float(np.nanmean(base)),scored=int(valid.sum())))
    independent=pd.DataFrame(results)
    recorded=pd.read_csv(output/'test_metrics_by_horizon.csv').query("model=='selected_ensemble'")
    joined=independent.merge(recorded,on=['target','horizon'],suffixes=('_independent','_recorded'))
    np.testing.assert_allclose(joined.mape_independent,joined.mape_recorded,atol=5.1e-7)
    summaries=[]
    for t in TARGETS:
        for period,horizons in [('short_8',range(1,9)),('long_96',range(1,97))]:
            selected=independent[(independent.target==t)&independent.horizon.isin(horizons)]
            summaries.append(dict(target=t,period=period,mape=float(selected.mape.mean()),
                score_1_minus_mape=1-float(selected.mape.mean()),
                persistence_mape=float(selected.persistence_mape.mean()),
                valid_cells=int(selected.scored.sum()),
                unavailable_cells=int(len(origins)*len(horizons)-selected.scored.sum())))
    # Daily rolling errors expose difficult regimes hidden by the overall mean.
    daily=[]
    for t in TARGETS:
        for period,horizons in [('short_8',range(1,9)),('long_96',range(1,97))]:
            def row_mean(values):
                count=np.isfinite(values).sum(axis=1)
                return np.divide(np.nansum(values,axis=1),count,
                    out=np.full(len(count),np.nan),where=count>0)
            e=row_mean(np.column_stack([errors[t,h] for h in horizons]))
            b=row_mean(np.column_stack([baseline[t,h] for h in horizons]))
            df=pd.DataFrame({'mape':e,'persistence_mape':b},index=origins)
            for day,row in df.groupby(df.index.date).mean().iterrows():
                daily.append(dict(date=str(day),target=t,period=period,**row.to_dict()))
    pd.DataFrame(daily).to_csv(output/'test_metrics_by_day.csv',index=False)

    train=read_observations(directory).loc[:pd.Timestamp(metadata['training_cutoff'])]
    combined=pd.concat([train,raw_test]).sort_index()
    # Reload the actual delivered artifact and reproduce first/middle/last origins.
    model=joblib.load(output/'forecast_model.joblib')
    selection=json.loads((output/'selection_frozen.json').read_text(encoding='utf-8'))
    sample=origins[[0,len(origins)//2,-1]]
    reproduced=model.predict(combined,features(combined),sample,selection['weights'])
    for h in range(1,97):
        group=f'generator_1_t+{h*15}_pred'; total=f'generator_all_t+{h*15}_pred'
        reproduced[group]=np.minimum(reproduced[group],reproduced[total])
    np.testing.assert_allclose(reproduced[long.columns],long.loc[sample],rtol=0,atol=1e-6)
    origin=origins.max()
    net,caps,initial,capacity,_=observed_surplus(combined,origin,96)
    eta=GasResourceForecaster(combined.ffill().fillna(0)).estimate_efficiency(origin)
    plan=frames['opt_result']; a=frames['dispatch_audit']; dt=.25; tol=1e-3
    q={g:plan['opt_generator_use_'+g+'_gas'].to_numpy() for g in GAS_TYPES}
    stock=np.r_[initial,a.holder_m3.to_numpy()]
    flare={g:net[g]-q[g] for g in GAS_TYPES}
    flare['blast_furnace']-=np.diff(stock)/dt
    for g in GAS_TYPES:
        assert q[g].min()>=-tol and (q[g]<=caps[g]+tol).all()
        assert flare[g].min()>=-tol, f'{g}: negative flare implies invented gas'
    np.testing.assert_allclose(sum(flare.values()),a.flare_m3h,atol=tol,rtol=0)
    np.testing.assert_allclose(sum(q[g]*eta[g] for g in GAS_TYPES),a.generator_all,atol=tol,rtol=0)
    assert stock.min()>=capacity*.15-tol and stock.max()<=capacity*.9+tol
    assert stock[-1]>=initial-tol
    for count,power,rating,max_units in [(a.online_50mw,a.generator_1,50,4),
            (a.online_120mw,a.generator_all-a.generator_1,120,2)]:
        np.testing.assert_allclose(count,np.rint(count),atol=tol)
        assert count.min()>=0 and count.max()<=max_units
        assert (power>=.6*rating*count-tol).all() and (power<=rating*count+tol).all()
    tariff=PriceSchedule.from_excel(directory/'price.xlsx').prices(plan.index).to_numpy()
    revenue=float(np.dot(a.generator_all,tariff)*250)
    assert abs(revenue-metadata['selected_tariff_schedule_revenue'])<.1
    resource=pd.DataFrame({'datetime':plan.index,'price':tariff})
    for g in GAS_TYPES:
        resource[g+'_net_m3h']=net[g]; resource[g+'_cap_m3h']=caps[g]
        resource[g+'_mw_per_m3h']=eta[g]
    resource.to_csv(output/'dispatch_inputs.csv',index=False)
    versions={n:importlib.metadata.version(n) for n in
        ['numpy','pandas','scipy','lightgbm','catboost','scikit-learn','openpyxl','joblib']}
    source_files=['pipeline.py','forecasting.py','dispatch.py','experiment.py','run_official.py','main.py']
    report=dict(status='passed',python=platform.python_version(),versions=versions,
        serialized_model_reproduced_origins=len(sample),
        scores=summaries,independent_dispatch_revenue=revenue,
        model_bytes=(output/'forecast_model.joblib').stat().st_size,
        source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in source_files},
        test_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in test_files[0].parent.glob('*.csv')})
    write_json(output/'validation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,default=Path('.'))
    parser.add_argument('--output-dir',type=Path,default=Path('output/official'))
    args=parser.parse_args()
    validate(args.input_dir,args.output_dir)
