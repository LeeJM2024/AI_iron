"""Freeze the v3 pipeline from April development predictions, never May labels."""
from dataclasses import asdict
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from develop_stable import FOLDS, SPECS_STABLE
from regime_ridge import REGIME_SPECS
from prelim import TARGETS,HORIZONS,read_data,future,score,fit_weights,band,dump_json


def cv_metrics(records,names):
    tuning=[r for r in records if r['fold']!=FOLDS[-1]]
    frozen=fit_weights(tuning,names)
    rows=[]
    for fold in FOLDS:
        w=frozen if fold==FOLDS[-1] else fit_weights([r for r in tuning if r['fold']!=fold],names)
        for r in [r for r in records if r['fold']==fold]:
            ww=np.array([w[f"{r['target']}/{band(r['h'])}"][n] for n in names])
            rows.append(dict(fold=fold,target=r['target'],horizon=r['h'],mape=score(r['y'],r['p']@ww)))
    return pd.DataFrame(rows),frozen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--base-dir',type=Path,default=Path('artifacts/stable_v3_robust_linear'))
    p.add_argument('--regime-dir',type=Path,default=Path('artifacts/stable_v3_regime'))
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/stable_v3_selected'))
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=read_data(args.train_dir)
    raw=raw.loc[raw.index<'2025-05-01']
    specs=SPECS_STABLE+REGIME_SPECS
    names=[s.name for s in specs]
    records=[]
    for fold in FOLDS:
        ix=pd.date_range(fold,periods=192,freq='15min',name='datetime')
        members={}
        for s in specs:
            directory=args.base_dir if s in SPECS_STABLE else args.regime_dir
            members[s.name]=joblib.load(directory/f'fold_{fold}_{s.name}.joblib')['predictions']
        for t in TARGETS:
            for h in HORIZONS:
                target_times = future(ix,h)
                y = raw[t].reindex(target_times).to_numpy()
                # A development fold's final horizons must not borrow labels
                # from the next fold (especially the April 29 monitor window).
                y[target_times > ix.max()] = np.nan
                records.append(dict(fold=fold,target=t,h=h,
                    y=y,
                    p=np.column_stack([members[n][t,h] for n in names])))
    base_names=[s.name for s in SPECS_STABLE]
    base_records=[{**r,'p':r['p'][:,:len(base_names)]} for r in records]
    base,base_w=cv_metrics(base_records,base_names)
    expanded,expanded_w=cv_metrics(records,names)
    # Guard against experts increasing variance: choose per target using only
    # leave-fold-out development errors, not the final April window or May.
    choice={}
    weights={}
    for t in TARGETS:
        a=base[(base.fold!=FOLDS[-1])&(base.target==t)].mape.mean()
        b=expanded[(expanded.fold!=FOLDS[-1])&(expanded.target==t)].mape.mean()
        use_expanded=b<a*.995
        choice[t]=dict(base_cv=float(a),regime_cv=float(b),use_regime=bool(use_expanded),minimum_gain=.005)
        for horizon_band in ('15_30','45_120'):
            key=f'{t}/{horizon_band}'
            ww=expanded_w[key] if use_expanded else base_w[key]
            weights[key]={n:float(ww.get(n,0.)) for n in names}
    selected=pd.concat([expanded[expanded.target==t] if choice[t]['use_regime'] else base[base.target==t]
                        for t in TARGETS])
    pd.concat([base.assign(model='base_cv'),expanded.assign(model='regime_cv'),
               selected.assign(model='selected')]).to_csv(args.output_dir/'development_metrics.csv',index=False)
    selection=dict(version=1,name='stable_v3',train_only=True,names=names,
        specs=[asdict(s) for s in specs],weights=weights,calibrations={},
        transform=dict(kind='stable',active_days=7,robust_raw=True),
        tuning_cutoffs=FOLDS[:-1],holdout_cutoff=FOLDS[-1],family_selection=choice,
        protocol='April-only selection. Rolling observations at/before origin. Active schema frozen before May.')
    dump_json(args.output_dir/'selection.json',selection)
    print(pd.concat([base.assign(model='base_cv'),expanded.assign(model='regime_cv'),
        selected.assign(model='selected')]).groupby(['fold','target','model']).mape.mean().unstack().round(6).to_string(),flush=True)
    print(choice,flush=True)
    print({k:{n:round(v,4) for n,v in w.items() if v>1e-6} for k,w in weights.items()},flush=True)


if __name__=='__main__':
    main()
