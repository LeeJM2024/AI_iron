"""April-only compact-input ablation with reusable prediction caches."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from prelim import Spec,ShortModel,TARGETS,HORIZONS,read_data,future,score,dump_json,fit_weights,band
from pooled_short import PooledShort
from compact_inputs import CompactInputTransform
from develop_stable import FOLDS


COMPACT_SPECS=[Spec('ridge21','ridge',days=21,alpha=100.),
               Spec('ridge60','ridge',days=60,alpha=300.,half_life=14.),
               Spec('lgb60','lgb',days=60,trees=240,half_life=14.),
               Spec('pooled_relative','pooled',days=90,trees=500,half_life=21.)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/compact_v5'))
    p.add_argument('--members',nargs='+')
    p.add_argument('--smooth-inputs',action='store_true')
    p.add_argument('--dynamic-features',action='store_true')
    p.add_argument('--quantile-features',action='store_true')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=read_data(args.train_dir)
    raw=raw.loc[raw.index<'2025-05-01']
    specs=[s for s in COMPACT_SPECS if not args.members or s.name in args.members]
    names=[s.name for s in specs]
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+Path('compact_inputs.py').read_bytes()+
        Path('stable_inputs.py').read_bytes()+pd.util.hash_pandas_object(raw).values.tobytes()+
        str((args.smooth_inputs,args.dynamic_features,args.quantile_features)).encode()).hexdigest()
    records,metrics,quality=[],[],[]
    for fold in FOLDS:
        cutoff=future([pd.Timestamp(fold)],-1)[0]
        ix=pd.date_range(fold,periods=192,freq='15min',name='datetime')
        tx=CompactInputTransform(smooth_inputs=args.smooth_inputs,dynamic_features=args.dynamic_features,
                                 quantile_features=args.quantile_features).fit(raw.loc[:cutoff])
        x=tx.transform(raw)
        view=x.loc[ix]
        z=((view-view.mean())/view.std()).abs()
        quality.append(dict(fold=fold,columns=len(view.columns),constant=int((view.nunique()<=1).sum()),
                            z3=int(z.gt(3).sum().sum())))
        predictions={}
        for spec in specs:
            path=args.output_dir/f'fold_{fold}_{spec.name}.joblib'
            saved=joblib.load(path) if path.exists() else None
            if saved and saved.get('fingerprint')==fingerprint:
                pred=saved['predictions']
            else:
                tic=time.perf_counter()
                cls=PooledShort if spec.kind=='pooled' else ShortModel
                m=cls(spec,args.threads).fit(raw,x,cutoff)
                pred=m.predict(raw,x,ix)
                joblib.dump(dict(fingerprint=fingerprint,predictions=pred,audit=m.audit,spec=asdict(spec)),path)
                print(f'{fold} {spec.name} {time.perf_counter()-tic:.1f}s',flush=True)
            predictions[spec.name]=pred
        for t in TARGETS:
            for h in HORIZONS:
                times=future(ix,h)
                y=raw[t].reindex(times).to_numpy()
                y[times>ix.max()]=np.nan
                matrix=np.column_stack([predictions[n][t,h] for n in names])
                records.append(dict(fold=fold,target=t,h=h,y=y,p=matrix))
                for i,n in enumerate(names):
                    metrics.append(dict(fold=fold,target=t,horizon=h,member=n,mape=score(y,matrix[:,i])))
        print(pd.DataFrame(metrics).query('fold==@fold').groupby(['target','member']).mape.mean().unstack().round(5).to_string(),flush=True)
    tuning=[r for r in records if r['fold']!=FOLDS[-1]]
    weights=fit_weights(tuning,names)
    for fold in FOLDS:
        w=weights if fold==FOLDS[-1] else fit_weights([r for r in tuning if r['fold']!=fold],names)
        for r in [r for r in records if r['fold']==fold]:
            ww=np.array([w[f"{r['target']}/{band(r['h'])}"][n] for n in names])
            metrics.append(dict(fold=fold,target=r['target'],horizon=r['h'],member='ensemble_heldout',mape=score(r['y'],r['p']@ww)))
    table=pd.DataFrame(metrics)
    table.to_csv(args.output_dir/'development_metrics.csv',index=False)
    pd.DataFrame(quality).to_csv(args.output_dir/'input_diagnostics.csv',index=False)
    joblib.dump(records,args.output_dir/'oof_predictions.joblib')
    dump_json(args.output_dir/'selection.json',dict(version=1,name='compact_v5',train_only=True,names=names,
        specs=[asdict(s) for s in specs],weights=weights,calibrations={},
        transform=dict(kind='compact',active_days=7,robust_raw=True,smooth_inputs=args.smooth_inputs,
                       dynamic_features=args.dynamic_features,quantile_features=args.quantile_features),
        tuning_cutoffs=FOLDS[:-1],holdout_cutoff=FOLDS[-1],
        protocol='Core-signal feature ablation; April-only tuning; per-horizon purging; no May-based selection.'))
    print(table.groupby(['fold','target','member']).mape.mean().unstack().round(5).to_string(),flush=True)
    print(pd.DataFrame(quality).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
