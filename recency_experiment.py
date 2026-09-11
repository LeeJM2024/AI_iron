"""Train-only screen for causal time-decayed direct residual models."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import discover, read_observations
from forecasting import MEMBERS, TARGETS, ModelConfig, ResidualEnsemble, features, mape, shifted
from pipeline import write_json


HORIZONS=tuple(range(1,9))+(12,16,24,32,48,64,72,96)


def score(raw,x,cutoff,days,cfg,label):
    origins=raw.loc[pd.Timestamp(cutoff):pd.Timestamp(cutoff)+pd.Timedelta(int(days),unit='D')-pd.Timedelta(15,unit='min')].index
    model=ResidualEnsemble(cfg).fit(raw,x,cutoff,HORIZONS)
    rows=[]
    for target in TARGETS:
        member=np.stack([model.predict_members(raw,x,origins,h,target) for h in HORIZONS],axis=1)
        for j,h in enumerate(HORIZONS):
            y=raw[target].reindex(shifted(origins,h)).to_numpy()
            for k,name in enumerate(MEMBERS):
                rows.append(dict(cutoff=str(cutoff),candidate=label,target=target,horizon=h,
                    model=name,mape=mape(y,member[:,j,k])))
            rows.append(dict(cutoff=str(cutoff),candidate=label,target=target,horizon=h,
                model='learned_mean',mape=mape(y,member[:,j,3:5].mean(axis=1))))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/recency_screen'))
    parser.add_argument('--cutoffs',nargs='+',default=['2025-04-07','2025-04-18','2025-04-27'])
    parser.add_argument('--half-lives',nargs='+',type=float,default=[30.0])
    parser.add_argument('--trees',type=int,default=180)
    parser.add_argument('--days',type=int,default=2)
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    raw=read_observations(args.train_dir or discover('.')).loc[lambda d:d.index<'2025-05-01']
    x=features(raw); started=time.perf_counter(); rows=[]
    for half_life in args.half_lives:
        label=f'half_life_{half_life:g}d'
        cfg=ModelConfig(trees=args.trees,recency_half_life_days=half_life)
        for cutoff in map(pd.Timestamp,args.cutoffs):
            rows.extend(score(raw,x,cutoff,args.days,cfg,label))
    metrics=pd.DataFrame(rows); metrics.to_csv(args.output_dir/'metrics.csv',index=False)
    summary=metrics.assign(period=np.where(metrics.horizon<=8,'short','long_sampled')).groupby(
        ['cutoff','candidate','period','target','model']).mape.mean().reset_index()
    summary.to_csv(args.output_dir/'summary.csv',index=False)
    ranking=summary.groupby(['candidate','period','target','model']).mape.mean().reset_index().sort_values(
        ['candidate','period','target','mape'])
    ranking.to_csv(args.output_dir/'ranking.csv',index=False)
    write_json(args.output_dir/'run.json',dict(cutoffs=args.cutoffs,half_lives=args.half_lives,
        horizons=HORIZONS,seconds=time.perf_counter()-started,
        protocol='Training ends at each cutoff; no official May test data is read.'))
    print(ranking.to_string(index=False))


if __name__=='__main__':
    main()
