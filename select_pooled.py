"""Admit shared-horizon models only with April leave-fold-out evidence."""
from dataclasses import asdict
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from develop_stable import FOLDS,SPECS_STABLE
from pooled_short import POOLED_SPECS
from select_stable import cv_metrics
from prelim import TARGETS,HORIZONS,read_data,future,dump_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--base-dir',type=Path,default=Path('artifacts/stable_v3_robust_linear'))
    p.add_argument('--pooled-dir',type=Path,default=Path('artifacts/pooled_v4'))
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/pooled_v4_selected'))
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=read_data(args.train_dir)
    raw=raw.loc[raw.index<'2025-05-01']
    specs=SPECS_STABLE+POOLED_SPECS
    names=[s.name for s in specs]
    records=[]
    for fold in FOLDS:
        ix=pd.date_range(fold,periods=192,freq='15min',name='datetime')
        members={}
        for spec in specs:
            directory=args.base_dir if spec in SPECS_STABLE else args.pooled_dir
            members[spec.name]=joblib.load(directory/f'fold_{fold}_{spec.name}.joblib')['predictions']
        for t in TARGETS:
            for h in HORIZONS:
                times=future(ix,h)
                y=raw[t].reindex(times).to_numpy()
                y[times>ix.max()]=np.nan
                records.append(dict(fold=fold,target=t,h=h,y=y,
                    p=np.column_stack([members[n][t,h] for n in names])))
    base_names=[s.name for s in SPECS_STABLE]
    base_records=[{**r,'p':r['p'][:,:len(base_names)]} for r in records]
    base,bw=cv_metrics(base_records,base_names)
    expanded,ew=cv_metrics(records,names)
    weights,choices={},{}
    for t in TARGETS:
        a=base[(base.target==t)&(base.fold!=FOLDS[-1])].mape.mean()
        b=expanded[(expanded.target==t)&(expanded.fold!=FOLDS[-1])].mape.mean()
        use=b<a*.995
        choices[t]=dict(base_cv=float(a),pooled_cv=float(b),minimum_relative_gain=.005,use_pooled=bool(use))
        for band in ('15_30','45_120'):
            key=f'{t}/{band}'
            src=ew[key] if use else bw[key]
            weights[key]={n:float(src.get(n,0.)) for n in names}
    selected=pd.concat([expanded[expanded.target==t] if choices[t]['use_pooled'] else base[base.target==t] for t in TARGETS])
    table=pd.concat([base.assign(model='base_cv'),expanded.assign(model='pooled_cv'),selected.assign(model='selected')])
    table.to_csv(args.output_dir/'development_metrics.csv',index=False)
    dump_json(args.output_dir/'selection.json',dict(version=1,name='pooled_v4',train_only=True,
        names=names,specs=[asdict(s) for s in specs],weights=weights,calibrations={},
        transform=dict(kind='stable',active_days=7,robust_raw=True),family_selection=choices,
        tuning_cutoffs=FOLDS[:-1],holdout_cutoff=FOLDS[-1],
        protocol='April-only tuning, purged training targets, disjoint fold labels, frozen before May evaluation.'))
    print(table.groupby(['fold','target','model']).mape.mean().unstack().round(6).to_string(),flush=True)
    print(choices,flush=True)
    print({k:{n:round(v,4) for n,v in w.items() if v>1e-6} for k,w in weights.items()},flush=True)


if __name__=='__main__':
    main()
