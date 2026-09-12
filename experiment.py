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
from forecasting import ModelConfig, ResidualEnsemble, features, select_weights, weights_for, mape, MEMBERS, TARGETS, bucket, shifted
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
                pred=p@weights_for(weights,t,h)
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
    parser.add_argument('--smooth-learned',type=int,default=0,
        help='Average learned residuals over +/- N fitted horizons')
    parser.add_argument('--profile-days',type=int,default=7,
        help='Trailing days used by the regime-matched diurnal profile member')
    parser.add_argument('--no-profile-regime-lock',action='store_true',
        help='Disable the structural-break restriction on the profile window')
    parser.add_argument('--regime-lock-min-horizon',type=int,default=0,
        help='Train regime-locked only above this horizon (0 = lock all)')
    parser.add_argument('--shared-horizon',action='store_true',
        help='Wire the pooled SharedHorizonResidual in as an extra member')
    parser.add_argument('--shared-horizon-targets',nargs='+',choices=TARGETS,default=None,
        help='Targets admitting the pooled member (default: both)')
    parser.add_argument('--tuning-cutoffs',nargs='+',
        default=['2025-04-23','2025-04-25','2025-04-27'],
        help='Cutoffs used to fit ensemble weights; post-break dates with >=5 '
             'days of regime history so the profile member is mature. The '
             '2025-04-18 fold trained across the regime change and is retired')
    parser.add_argument('--validation-cutoff',default='2025-04-29',
        help='Fold reserved for validation; never used to fit weights')
    parser.add_argument('--horizon-grid',nargs='+',type=int,
        default=[1,2,3,4,5,6,7,8,12,16,20,24,28,32,40,48,56,64,80,96],
        help='Anchor horizons at which ensemble weights are fitted. The official '
             'run serves h=1..96, so the grid must cover every bucket with at '
             'least three anchors or the weights are extrapolated, not fitted.')
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
        recency_half_life_by_target=per_target,smooth_learned=args.smooth_learned,
        regime_lock_min_horizon=args.regime_lock_min_horizon,
        use_shared_horizon=args.shared_horizon,
        shared_horizon_targets=tuple(args.shared_horizon_targets) if args.shared_horizon_targets else TARGETS,
        profile_days=args.profile_days,
        profile_regime_lock=not args.no_profile_regime_lock)
    horizons=sorted(set(int(h) for h in args.horizon_grid))
    records=[]
    scores=[]
    tuning=list(args.tuning_cutoffs)
    for cutoff in tuning:
        r,s=evaluate_fold(raw,x,cutoff,horizons,cfg)
        records.extend(r); scores.extend(s)
    weights=select_weights(records)
    write_json(args.output_dir/'selection.json',dict(config=asdict(cfg),weights=weights,
        tuning_cutoffs=tuning,validation_cutoff=args.validation_cutoff,
        excluded_from_selection='all official May test data'))
    # This fold is not used to fit ensemble weights or model hyperparameters.
    _,s=evaluate_fold(raw,x,args.validation_cutoff,horizons,cfg,weights)
    scores.extend(s)
    table=pd.DataFrame(scores)
    table.to_csv(args.output_dir/'fold_metrics.csv',index=False)
    bands=np.where(table.horizon<=8,'short_8',np.where(table.horizon<=48,'mid','long_sampled'))
    summary=table.assign(period=bands).groupby(
        ['cutoff','period','target','model']).mape.mean().reset_index()
    summary.to_csv(args.output_dir/'summary.csv',index=False)
    print(summary[summary.cutoff.str.startswith(args.validation_cutoff)].to_string(index=False))
    write_json(args.output_dir/'runtime.json',dict(seconds=time.perf_counter()-start))


if __name__=='__main__':
    main()
