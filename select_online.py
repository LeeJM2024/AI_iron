"""Admit periodic refits only when April leave-window-out performance improves."""
from dataclasses import asdict
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from prelim import TARGETS,HORIZONS,read_data,future,dump_json
from online_model import SPECS
from develop_compact import COMPACT_SPECS
from develop_stable import FOLDS
from select_stable import cv_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/online_v10_selected'))
    p.add_argument('--include-relative',action='store_true')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=read_data(args.train_dir)
    raw=raw.loc[raw.index<'2025-05-01']
    specs=COMPACT_SPECS+SPECS
    if args.include_relative:
        from relative_linear import SPECS as RELATIVE_SPECS
        specs+=RELATIVE_SPECS
    names=[s.name for s in specs]
    records=[]
    for fold in FOLDS:
        ix=pd.date_range(fold,periods=192,freq='15min',name='datetime')
        pred=[]
        for spec in specs:
            directory=Path('artifacts/compact_v5' if spec in COMPACT_SPECS else
                           ('artifacts/relative_linear_v10' if spec.kind=='relative_linear' else 'artifacts/online_v10'))
            pred.append(joblib.load(directory/f'fold_{fold}_{spec.name}.joblib')['predictions'])
        for t in TARGETS:
            for h in HORIZONS:
                times=future(ix,h)
                y=raw[t].reindex(times).to_numpy()
                y[times>ix.max()]=np.nan
                records.append(dict(fold=fold,target=t,h=h,y=y,p=np.column_stack([v[t,h] for v in pred])))
    base_names=names[:len(COMPACT_SPECS)]
    base_records=[{**r,'p':r['p'][:,:len(base_names)]} for r in records]
    base,bw=cv_metrics(base_records,base_names)
    challenger,cw=cv_metrics(records,names)
    choices,weights={},{}
    for t in TARGETS:
        a=base[(base.target==t)&(base.fold!=FOLDS[-1])].mape.mean()
        b=challenger[(challenger.target==t)&(challenger.fold!=FOLDS[-1])].mape.mean()
        use=b<a*.995
        choices[t]=dict(base_cv=float(a),online_cv=float(b),accepted=bool(use),minimum_relative_gain=.005)
        for band in ('15_30','45_120'):
            key=f'{t}/{band}'
            selected=cw[key] if use else bw[key]
            weights[key]={n:float(selected.get(n,0.)) for n in names}
    selected=pd.concat([challenger[challenger.target==t] if choices[t]['accepted'] else base[base.target==t] for t in TARGETS])
    table=pd.concat([base.assign(model='v5'),challenger.assign(model='online'),selected.assign(model='selected')])
    table.to_csv(args.output_dir/'development_metrics.csv',index=False)
    dump_json(args.output_dir/'selection.json',dict(version=1,name='online_v10',train_only=True,
        specs=[asdict(s) for s in specs],weights=weights,calibrations={},family_selection=choices,
        transform=dict(kind='compact',active_days=7,robust_raw=True,smooth_inputs=False,dynamic_features=False,quantile_features=False),
        tuning_cutoffs=FOLDS[:-1],holdout_cutoff=FOLDS[-1],
        protocol='Hyperparameters/ensemble weights/transform fixed using April. Ridge coefficients refit using only observations strictly before each scheduled update. Never labels after forecast origin.'))
    print(table.groupby(['fold','target','model']).mape.mean().unstack().round(6).to_string(),flush=True)
    print(choices,flush=True)
    print({k:{n:round(v,4) for n,v in w.items() if v>1e-6} for k,w in weights.items()},flush=True)

if __name__=='__main__':
    main()
