"""Pooled direct forecasting across eight horizons, with causal label purging.

The relative-residual member shares load dynamics between operating levels. The
absolute-residual member retains MW-level corrections. Horizon is a known query
parameter; derived covariates depend exclusively on the submitted input row.
"""
from __future__ import annotations
from dataclasses import asdict
import argparse
import hashlib
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from prelim import Spec, TARGETS, HORIZONS, UPPER, read_data, future, score
from develop_stable import FOLDS
from stable_inputs import StableInputTransform


POOLED_SPECS = [Spec('pooled60', 'pooled', days=60, trees=400, half_life=14.),
                Spec('pooled21', 'pooled', days=21, trees=400, half_life=0.),
                Spec('pooled_relative', 'pooled', days=90, trees=500, half_life=21.)]


def augment(x, h, target):
    """Deterministic transformation of input.csv plus requested target/horizon."""
    z = x.copy()
    z['feat_requested_horizon'] = float(h)
    now = x[target].clip(lower=1.)
    for n in (1,2,3,4,6,8,12,24,48,96):
        col = f'feat_{target}_lag_{n}'
        if col in x:
            z[f'feat_relative_lag_{n}'] = x[col]/now-1.
    for n in (2,4,8,16,32,96):
        col = f'feat_{target}_mean_{n}'
        if col in x:
            z[f'feat_relative_mean_{n}'] = x[col]/now-1.
    # Same feature manifest at fit/serve; these do not require future gas values.
    for gas in ('blast_furnace','coke','converter'):
        c = f'generator_use_{gas}_gas'
        if c in x:
            z[f'feat_gas_per_power_{gas}'] = x[c]/x.generator_all.clip(lower=1.)
    z['feat_group_power_ratio'] = x.generator_1/x.generator_all.clip(lower=1.)
    # Known target clock from origin Fourier pair, without accessing target rows.
    angle = 2*np.pi*h/96.
    s = 2*x.feat_clock_sin_1-1
    c = 2*x.feat_clock_cos_1-1
    z['feat_requested_clock_sin'] = s*np.cos(angle)+c*np.sin(angle)
    z['feat_requested_clock_cos'] = c*np.cos(angle)-s*np.sin(angle)
    return z.astype('float32')


class PooledShort:
    def __init__(self,spec,threads=4):
        self.spec,self.threads=spec,threads
        self.models,self.audit={},[]

    def fit(self,raw,x,cutoff):
        self.cutoff=pd.Timestamp(cutoff)
        self.relative=self.spec.name=='pooled_relative'
        self.columns=list(x.columns)
        for t in TARGETS:
            matrices,labels,weights=[],[],[]
            for h in HORIZONS:
                rows=x.index[(x.index>future([cutoff],-96*self.spec.days)[0])&
                             (x.index<=future([cutoff],-h)[0])]
                y=raw[t].reindex(future(rows,h)).to_numpy()
                ok=np.isfinite(y)&(y>0)&(y<=UPPER[t])&raw.loc[rows,t].notna().to_numpy()
                rows,y=rows[ok],y[ok]
                if len(rows)<96:
                    raise ValueError('Insufficient observed labels for pooled model')
                now=x.loc[rows,t].to_numpy()
                residual=y-now
                w=1/np.maximum(y,1.)
                if self.relative:
                    residual=residual/np.maximum(now,1.)
                    w=w*np.maximum(now,1.)
                if self.spec.half_life:
                    age=np.asarray((cutoff-rows).total_seconds())/86400
                    w*=np.exp2(-age/self.spec.half_life)
                # Every horizon is equally important despite different pair counts.
                w=w/w.sum()
                matrices.append(augment(x.loc[rows],h,t))
                labels.append(residual)
                weights.append(w)
                self.audit.append(dict(member=self.spec.name,target=t,horizon=h,rows=len(rows),
                    last_origin=str(rows[-1]),last_label=str(future(rows[-1:],h)[0]),cutoff=str(cutoff)))
            z=pd.concat(matrices,ignore_index=True)
            y=np.concatenate(labels)
            w=np.concatenate(weights)
            w/=w.mean()
            model=LGBMRegressor(objective='regression_l1',n_estimators=self.spec.trees,
                num_leaves=23,learning_rate=.03,min_child_samples=120,
                reg_lambda=10.,reg_alpha=.1,colsample_bytree=.9,
                n_jobs=self.threads,random_state=2026,deterministic=True,
                force_col_wise=True,verbosity=-1)
            model.fit(z,y,sample_weight=w)
            self.models[t]=model
        return self

    def predict(self,raw,x,origins):
        if min(origins)<self.cutoff:
            raise ValueError('Origins precede model cutoff')
        result={}
        for t in TARGETS:
            now=x.loc[origins,t].to_numpy()
            for h in HORIZONS:
                residual=self.models[t].predict(augment(x.loc[origins,self.columns],h,t))
                if self.relative:
                    residual*=np.maximum(now,1.)
                result[t,h]=np.clip(now+residual,0,UPPER[t])
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/pooled_v4'))
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=read_data(args.train_dir)
    raw=raw.loc[raw.index<'2025-05-01']
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+Path('stable_inputs.py').read_bytes()+
        pd.util.hash_pandas_object(raw).values.tobytes()).hexdigest()
    metrics=[]
    for fold in FOLDS:
        cutoff=future([pd.Timestamp(fold)],-1)[0]
        origins=pd.date_range(fold,periods=192,freq='15min',name='datetime')
        tx=StableInputTransform(robust_raw=True).fit(raw.loc[:cutoff])
        x=tx.transform(raw)
        for spec in POOLED_SPECS:
            path=args.output_dir/f'fold_{fold}_{spec.name}.joblib'
            saved=joblib.load(path) if path.exists() else None
            if saved and saved.get('fingerprint')==fingerprint:
                pred=saved['predictions']
            else:
                tic=time.perf_counter()
                model=PooledShort(spec,args.threads).fit(raw,x,cutoff)
                pred=model.predict(raw,x,origins)
                joblib.dump(dict(predictions=pred,spec=asdict(spec),audit=model.audit,
                                 fingerprint=fingerprint),path)
                print(f'{fold} {spec.name} {time.perf_counter()-tic:.1f}s',flush=True)
            for t in TARGETS:
                for h in HORIZONS:
                    times=future(origins,h)
                    y=raw[t].reindex(times).to_numpy()
                    y[times>origins.max()]=np.nan
                    metrics.append(dict(fold=fold,target=t,horizon=h,member=spec.name,mape=score(y,pred[t,h])))
        table=pd.DataFrame(metrics)
        print(table[table.fold==fold].groupby(['target','member']).mape.mean().unstack().round(5).to_string(),flush=True)
    pd.DataFrame(metrics).to_csv(args.output_dir/'development_metrics.csv',index=False)


if __name__=='__main__':
    main()
