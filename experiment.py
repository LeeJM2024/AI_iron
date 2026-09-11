"""Reproducible train-only model selection; official test labels never tune weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from forecasting import ModelConfig, ResidualEnsemble, features, select_weights, mape, MEMBERS, TARGETS, bucket, shifted
from pipeline import IndustrialDataPipeline, write_json


def discover(root):
    matches = list(Path(root).rglob('Pre_load.csv'))
    if len(matches) != 1:
        raise ValueError(f'Expected one Pre_load.csv, found {len(matches)}; set --train-dir')
    return matches[0].parent


def read_observations(directory):
    loader=IndustrialDataPipeline()
    loader.load(directory)
    return loader.observations


def data_profile(raw):
    return dict(rows=len(raw),start=str(raw.index.min()),end=str(raw.index.max()),
        missing=raw.isna().sum().to_dict(),zero_targets={t:int(raw[t].eq(0).sum()) for t in TARGETS})


def evaluate_fold(raw,x,cutoff,horizons,cfg,weights=None):
    cutoff=pd.Timestamp(cutoff)
    origins=x.loc[cutoff:cutoff+pd.Timedelta(2,unit='D')-pd.Timedelta(15,unit='min')].index
    model=ResidualEnsemble(cfg).fit(raw,x,cutoff,horizons)
    records=[]
    scores=[]
    for t in TARGETS:
        for h in horizons:
            p=model.predict_members(raw,x,origins,h,t)
            y=raw[t].reindex(shifted(origins,h)).to_numpy()
            records.append(dict(target=t,h=h,y=y,p=p))
            for k,name in enumerate(MEMBERS):
                scores.append(dict(cutoff=str(cutoff),target=t,horizon=h,model=name,mape=mape(y,p[:,k])))
            if weights is not None:
                pred=p@weights[f'{t}/{bucket(h)}']
                scores.append(dict(cutoff=str(cutoff),target=t,horizon=h,model='selected_ensemble',mape=mape(y,pred)))
    return records,scores


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('artifacts/development'))
    parser.add_argument('--trees',type=int,default=180)
    parser.add_argument('--train-days',type=int,default=90)
    parser.add_argument('--recency-half-life',type=float,default=None,
        help='Optional exponential age half-life in days for selected targets')
    parser.add_argument('--recency-targets',nargs='+',choices=TARGETS,default=None,
        help='Targets to receive --recency-half-life; default applies to both')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    start=time.perf_counter()
    directory=args.train_dir or discover('.')
    raw=read_observations(directory)
    # The distributed training file contains a May 1 endpoint. Keep all May out.
    raw=raw.loc[raw.index < '2025-05-01']
    x=features(raw)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*') if p.is_file()}
    write_json(args.output_dir/'data_profile.json',dict(**data_profile(raw),sha256=hashes))
    per_target=None
    if args.recency_targets is not None:
        if args.recency_half_life is None:
            parser.error('--recency-targets requires --recency-half-life')
        per_target={target:(args.recency_half_life if target in args.recency_targets else None)
            for target in TARGETS}
    cfg=ModelConfig(trees=args.trees,train_days=args.train_days,
        recency_half_life_days=args.recency_half_life if per_target is None else None,
        recency_half_life_by_target=per_target)
    horizons=list(range(1,9))+[16,24,48,96]
    records=[]
    scores=[]
    for cutoff in ('2025-04-07','2025-04-18'):
        r,s=evaluate_fold(raw,x,cutoff,horizons,cfg)
        records.extend(r); scores.extend(s)
    weights=select_weights(records)
    write_json(args.output_dir/'selection.json',dict(config=asdict(cfg),weights=weights,
        tuning_cutoffs=['2025-04-07','2025-04-18'],validation_cutoff='2025-04-27',
        excluded_from_selection='all official May test data'))
    # This fold is not used to fit ensemble weights or model hyperparameters.
    _,s=evaluate_fold(raw,x,'2025-04-27',horizons,cfg,weights)
    scores.extend(s)
    table=pd.DataFrame(scores)
    table.to_csv(args.output_dir/'fold_metrics.csv',index=False)
    summary=table.assign(period=np.where(table.horizon<=8,'short','long_sampled')).groupby(
        ['cutoff','period','target','model']).mape.mean().reset_index()
    summary.to_csv(args.output_dir/'summary.csv',index=False)
    print(summary[summary.cutoff.str.startswith('2025-04-27')].to_string(index=False))
    write_json(args.output_dir/'runtime.json',dict(seconds=time.perf_counter()-start))


if __name__=='__main__':
    main()
