"""Smooth operating-regime ridge experts; training-only state centres.

Unlike hard date break detection, each current operating state gates predictions
from weighted local ARX experts. All inputs and learned centres are causal.
"""
from __future__ import annotations
from dataclasses import asdict
import argparse
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from prelim import (Spec, ShortModel, TARGETS, HORIZONS, UPPER, read_data, future,
                    fit_weights, band, dump_json, score)
from stable_inputs import StableInputTransform
from develop_stable import FOLDS


class LocalRidge:
    def __init__(self, alpha=300., state_index=0, bandwidth=1.):
        self.alpha, self.state_index, self.bandwidth = alpha, state_index, bandwidth

    def fit(self, x, y, sample_weight):
        state = x[:,self.state_index]
        self.centres = np.unique(np.quantile(state,[.10,.35,.65,.90]))
        self.global_model = Ridge(alpha=self.alpha).fit(x,y,sample_weight=sample_weight)
        self.models = []
        for centre in self.centres:
            local = np.exp(-.5*((state-centre)/self.bandwidth)**2)
            w = sample_weight*(.05+.95*local)
            # Normalize so alpha means the same amount of shrinkage per expert.
            w /= w.mean()
            self.models.append(Ridge(alpha=self.alpha).fit(x,y,sample_weight=w))
        return self

    def predict(self, x):
        state = x[:,self.state_index]
        distances = ((state[:,None]-self.centres[None,:])/self.bandwidth)**2
        weights = np.exp(-.5*(distances-distances.min(axis=1,keepdims=True)))
        weights /= weights.sum(axis=1,keepdims=True)
        experts = np.column_stack([m.predict(x) for m in self.models])
        return .2*self.global_model.predict(x)+.8*np.sum(weights*experts,axis=1)


REGIME_SPECS = [Spec('regime30','ridge',days=60,alpha=300.,half_life=14.),
                Spec('regime15','ridge',days=60,alpha=300.,half_life=14.)]


def fit_regime(spec, raw, x, cutoff, threads=4):
    model = ShortModel(spec,threads).fit(raw,x,cutoff)
    state_index = model.columns.index('generator_1')
    for t in TARGETS:
        for h in HORIZONS:
            rows = x.index[(x.index > future([cutoff],-96*spec.days)[0]) &
                           (x.index <= future([cutoff],-h)[0])]
            y = raw[t].reindex(future(rows,h)).to_numpy()
            ok = np.isfinite(y)&(y>0)&(y<=UPPER[t])&raw.loc[rows,t].notna().to_numpy()
            rows,y = rows[ok],y[ok]
            w = 1/np.maximum(y,1.)
            age = np.asarray((cutoff-rows).total_seconds())/86400
            w *= np.exp2(-age/spec.half_life)
            w /= w.mean()
            scaler = model.scalers[t,h]
            z = scaler.transform(x.loc[rows,model.columns].to_numpy(dtype=float))
            bandwidth_mw = 30. if spec.name == 'regime30' else 15.
            model.models[t,h] = LocalRidge(spec.alpha,state_index,
                bandwidth_mw/scaler.scale_[state_index]).fit(z,y-x.loc[rows,t].to_numpy(),w)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('artifacts/stable_v3_regime'))
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=read_data(args.train_dir)
    raw=raw.loc[raw.index<'2025-05-01']
    metrics=[]
    for fold in FOLDS:
        cutoff=future([pd.Timestamp(fold)],-1)[0]
        origins=pd.date_range(fold,periods=192,freq='15min',name='datetime')
        tx=StableInputTransform(robust_raw=True).fit(raw.loc[:cutoff])
        x=tx.transform(raw)
        for spec in REGIME_SPECS:
            tic=time.perf_counter()
            m=fit_regime(spec,raw,x,cutoff)
            pred=m.predict(raw,x,origins)
            joblib.dump(dict(predictions=pred,spec=asdict(spec),audit=m.audit),
                args.output_dir/f'fold_{fold}_{spec.name}.joblib')
            for t in TARGETS:
                for h in HORIZONS:
                    y=raw[t].reindex(future(origins,h)).to_numpy()
                    y[future(origins,h) > origins.max()] = np.nan
                    metrics.append(dict(fold=fold,target=t,horizon=h,member=spec.name,mape=score(y,pred[t,h])))
            print(f'{fold} {spec.name}: {time.perf_counter()-tic:.1f}s',flush=True)
    table=pd.DataFrame(metrics)
    table.to_csv(args.output_dir/'development_metrics.csv',index=False)
    print(table.groupby(['fold','target','member']).mape.mean().unstack().round(5).to_string(),flush=True)


if __name__=='__main__':
    main()
