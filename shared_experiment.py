"""Train-only benchmark for the pooled shared-horizon challenger model."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import discover, read_observations
from forecasting import (MEMBERS, TARGETS, ModelConfig, ResidualEnsemble,
    SharedHorizonConfig, SharedHorizonResidual, bucket, features, mape, shifted)
from pipeline import write_json


SAMPLE_HORIZONS=tuple(range(1,9))+(12,16,24,32,48,64,72,96)


def origins_for(raw, cutoff, days):
    start=pd.Timestamp(cutoff)
    end=start+pd.Timedelta(int(days),unit='D')-pd.Timedelta(15,unit='min')
    return raw.loc[start:end].index


def score_predictions(raw, origins, predictor, model_name, horizons):
    records=[]
    for target in TARGETS:
        for h in horizons:
            pred=predictor(target,h)
            actual=raw[target].reindex(shifted(origins,h)).to_numpy()
            records.append(dict(target=target,horizon=h,model=model_name,
                mape=mape(actual,pred),scored=int(np.isfinite(actual).sum())))
    return records


def evaluate(raw, x, cutoff, days, baseline, pooled, horizons):
    origins=origins_for(raw,cutoff,days)
    records=[]
    clean=raw.ffill().fillna(0)
    records.extend(score_predictions(raw,origins,
        lambda target,h: clean.loc[origins,target].to_numpy(),'persistence',horizons))
    records.extend(score_predictions(raw,origins,
        lambda target,h: pooled.predict(raw,x,origins,h,target),'shared_horizon_lgb',horizons))
    for position,name in enumerate(MEMBERS):
        records.extend(score_predictions(raw,origins,
            lambda target,h,p=position: baseline.predict_members(raw,x,origins,h,target)[:,p],name,horizons))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/shared_horizon_screen'))
    parser.add_argument('--cutoffs',nargs='+',default=['2025-04-27'])
    parser.add_argument('--days',type=int,default=2)
    parser.add_argument('--trees',type=int,default=320)
    parser.add_argument('--baseline-trees',type=int,default=180)
    parser.add_argument('--stride',type=int,default=2)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    start=time.perf_counter(); args.output_dir.mkdir(parents=True,exist_ok=True)
    directory=args.train_dir or discover('.')
    raw=read_observations(directory).loc[lambda d:d.index<'2025-05-01']
    x=features(raw)
    all_rows=[]
    for cutoff in map(pd.Timestamp,args.cutoffs):
        # Both candidates are fit strictly before the validation origins.
        baseline=ResidualEnsemble(ModelConfig(trees=args.baseline_trees)).fit(raw,x,cutoff,SAMPLE_HORIZONS)
        pooled=SharedHorizonResidual(SharedHorizonConfig(trees=args.trees,origin_stride=args.stride)).fit(raw,x,cutoff,range(1,97))
        rows=evaluate(raw,x,cutoff,args.days,baseline,pooled,SAMPLE_HORIZONS)
        for row in rows:
            row['cutoff']=str(cutoff)
        all_rows.extend(rows)
    table=pd.DataFrame(all_rows)
    table.to_csv(args.output_dir/'metrics.csv',index=False)
    summary=table.assign(period=np.where(table.horizon<=8,'short','long_sampled')).groupby(
        ['cutoff','period','target','model']).mape.mean().reset_index()
    summary.to_csv(args.output_dir/'summary.csv',index=False)
    ranking=summary.groupby(['period','target','model']).mape.mean().reset_index().sort_values(['period','target','mape'])
    ranking.to_csv(args.output_dir/'ranking.csv',index=False)
    write_json(args.output_dir/'run.json',dict(cutoffs=args.cutoffs,days=args.days,horizons=SAMPLE_HORIZONS,
        pooled_config=dict(trees=args.trees,stride=args.stride),seconds=time.perf_counter()-start,
        protocol='Training ends at each cutoff; no official May test data is read.'))
    print(ranking.to_string(index=False))


if __name__=='__main__':
    main()
