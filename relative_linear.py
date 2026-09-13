"""Relative ARX regression, with optional IRLS approximation to MAPE."""
from pathlib import Path
import argparse
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from prelim import Spec,TARGETS,HORIZONS,UPPER,read_data,future,score
from compact_inputs import CompactInputTransform
from develop_stable import FOLDS

SPECS=[Spec('relative21','relative_linear',days=21,alpha=100.),
       Spec('relative60','relative_linear',days=60,alpha=300.,half_life=14.),
       Spec('relative60_mae','relative_linear',days=60,alpha=300.,half_life=14.),
       Spec('relative14_mae','relative_linear',days=14,alpha=100.)]


def features(x,target):
    z=x.astype('float64').copy()
    for t in TARGETS:
        level=x[t].clip(lower=1.)
        for c in x:
            if c.startswith(f'feat_{t}_'):
                z[c]=x[c]/level-1.
    z['relative_group_share']=x.generator_1/x.generator_all.clip(lower=1.)
    for c in x:
        if c.startswith('generator_use_'):
            z[c]=x[c]/x.generator_all.clip(lower=1.)
    z['relative_target_level']=x[target]/(200. if target==TARGETS[0] else 440.)
    return z


class RelativeLinear:
    def __init__(self,spec,threads=1):
        self.spec=spec

    def fit(self,raw,x,cutoff):
        self.cutoff=pd.Timestamp(cutoff)
        self.models,self.scalers,self.audit={},{},[]
        with threadpool_limits(limits=1):
            for t in TARGETS:
                f=features(x,t)
                for h in HORIZONS:
                    ix=x.index[(x.index>future([cutoff],-96*self.spec.days)[0])&(x.index<=future([cutoff],-h)[0])]
                    y=raw[t].reindex(future(ix,h)).to_numpy()
                    base=x.loc[ix,t].to_numpy()
                    valid=np.isfinite(y)&(y>0)&(y<=UPPER[t])&(base>0)&raw.loc[ix,t].notna().to_numpy()
                    ix,y,base=ix[valid],y[valid],base[valid]
                    r=y/base-1.
                    w=base/y
                    age=np.asarray((cutoff-ix).total_seconds())/86400
                    recency=np.exp2(-age/self.spec.half_life) if self.spec.half_life else np.ones(len(ix))
                    weights=recency*w**2
                    weights/=weights.mean()
                    scaler=StandardScaler().fit(f.loc[ix],sample_weight=weights)
                    z=scaler.transform(f.loc[ix])
                    model=Ridge(alpha=self.spec.alpha).fit(z,r,sample_weight=weights)
                    if self.spec.name.endswith('_mae'):
                        # Fixed training-derived smoothing keeps IRLS stable.
                        epsilon=max(.001,.1*np.median(np.abs(r-model.predict(z))))
                        for _ in range(8):
                            error=r-model.predict(z)
                            ww=recency*w/np.sqrt(error**2+epsilon**2)
                            ww/=ww.mean()
                            model=Ridge(alpha=self.spec.alpha).fit(z,r,sample_weight=ww)
                    self.models[t,h],self.scalers[t,h]=model,scaler
                    self.audit.append(dict(member=self.spec.name,target=t,horizon=h,rows=len(ix),
                        last_origin=str(ix[-1]),last_label=str(future(ix[-1:],h)[0]),cutoff=str(cutoff)))
        return self

    def predict(self,raw,x,origins):
        if min(origins)<self.cutoff:
            raise ValueError('Origins precede cutoff')
        out={}
        for t in TARGETS:
            f=features(x.loc[origins],t)
            base=x.loc[origins,t].to_numpy()
            for h in HORIZONS:
                r=self.models[t,h].predict(self.scalers[t,h].transform(f))
                out[t,h]=np.clip(base*(1+r),0,UPPER[t])
        return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/relative_linear_v10'))
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=read_data(args.train_dir)
    raw=raw.loc[raw.index<'2025-05-01']
    records=[]
    for fold in FOLDS:
        cutoff=future([pd.Timestamp(fold)],-1)[0]
        ix=pd.date_range(fold,periods=192,freq='15min',name='datetime')
        tx=CompactInputTransform().fit(raw.loc[:cutoff])
        x=tx.transform(raw)
        for spec in SPECS:
            tic=time.perf_counter()
            m=RelativeLinear(spec).fit(raw,x,cutoff)
            pred=m.predict(raw,x,ix)
            joblib.dump(dict(predictions=pred,audit=m.audit),args.output_dir/f'fold_{fold}_{spec.name}.joblib')
            for t in TARGETS:
                for h in HORIZONS:
                    times=future(ix,h)
                    y=raw[t].reindex(times).to_numpy()
                    y[times>ix.max()]=np.nan
                    records.append(dict(fold=fold,target=t,horizon=h,member=spec.name,mape=score(y,pred[t,h])))
            print(f'{fold} {spec.name}: {time.perf_counter()-tic:.1f}s',flush=True)
    table=pd.DataFrame(records)
    table.to_csv(args.output_dir/'development_metrics.csv',index=False)
    print(table.groupby(['fold','target','member']).mape.mean().unstack().round(5).to_string(),flush=True)

if __name__=='__main__':
    main()
