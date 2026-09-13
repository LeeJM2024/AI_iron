"""Fit a frozen development-selected model and evaluate the official held-out data."""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import replace
from experiment import discover, read_observations, data_profile
from forecasting import (ResidualEnsemble, ModelConfig, features, mape, TARGETS, MEMBERS,
                         bucket, shifted, detect_last_break)
from pipeline import PriceSchedule, DispatchConfig, MILPDispatcher, GasResourceForecaster, write_json
from dispatch import observed_surplus


def _tukey_fence(observed):
    """1.5-IQR fence of a 1-D sample, collapsing to the median if degenerate."""
    q1,q3=observed.quantile([.25,.75])
    iqr=float(q3-q1)
    if not np.isfinite(iqr) or iqr<=0:
        m=float(observed.median())
        return m,m
    return float(q1-1.5*iqr),float(q3+1.5*iqr)


def _regime_window(history, min_rows=288):
    """Post-break slice of the training history, or None if too short.

    ``min_rows`` of 288 is three days of 15-minute samples: enough for a stable
    IQR while still being available for every evaluation day in this project.
    """
    cols=[t for t in TARGETS if t in history.columns]
    if not cols:
        return None
    try:
        brk=detect_last_break(history[cols].mean(axis=1))
    except Exception:
        return None
    if brk is None:
        return None
    window=history.loc[brk:]
    return window if len(window)>=min_rows else None


def _quality_input_frame(x, cutoff, raw_history=None):
    """Return the delivered model-input table with train-fitted outlier repair.

    The competition scores explicit outlier handling in ``input.csv``.  Bounds
    are fitted only on observations available by the training cutoff, and each
    repair is exposed through a ``feat_*_outlier`` indicator.

    The fence is REGIME-AWARE, and that matters: a fence fitted on the whole
    history straddles the 2025-04-18 plant-state change, where post-break gas
    throughput sits +2.5 IQR above the long-run median.  Measured on the real
    window, the full-history fence repairs 312 delivered cells across 10 fields
    although ~300 of them are ordinary post-regime values -- e.g. every one of
    the 175 clipped ``generator_use_blast_furnace_gas`` cells is inside the
    post-regime distribution.  Clipping those would rewrite genuine observations
    in the delivered table.  A cell is therefore repaired only when it falls
    outside BOTH the full-history and the post-regime fence, i.e. only when the
    long-run and the current-regime distributions agree that it is anomalous.
    """
    history=(raw_history if raw_history is not None else x).loc[:pd.Timestamp(cutoff)]
    regime=_regime_window(history)
    delivered=x.copy()
    raw_columns=[c for c in delivered.columns if not c.startswith('feat_')]
    for column in raw_columns:
        observed=history[column].replace([np.inf,-np.inf],np.nan).dropna()
        if observed.empty:
            delivered[f'feat_{column}_outlier']=0.0
            continue
        lower,upper=_tukey_fence(observed)
        if regime is not None:
            post=regime[column].replace([np.inf,-np.inf],np.nan).dropna()
            if not post.empty:
                # Union of both fences: clip only where they agree.
                lo_post,hi_post=_tukey_fence(post)
                lower=min(lower,lo_post); upper=max(upper,hi_post)
        # CSV values are emitted from float32 features. Move the clipping
        # boundary one representable float inward so serialization cannot
        # round a repaired value back outside the fitted fence.
        lower32=float(np.nextafter(np.float32(lower),np.float32(np.inf)))
        upper32=float(np.nextafter(np.float32(upper),np.float32(-np.inf)))
        if not upper32>lower32:
            # Degenerate fence: a field that is constant across training (e.g.
            # converter_user1 == 0, IQR == 0) collapses to lower == upper.  The
            # inward nudges would then INVERT the bounds (nextafter(0, +inf) is
            # the smallest denormal, nextafter(0, -inf) its negative), making
            # clip() meaningless and the outlier flag permanently zero.  Use the
            # constant itself as both bounds instead.
            lower32=upper32=float(np.float32(upper))
        values=delivered[column].replace([np.inf,-np.inf],np.nan)
        repaired=values.clip(lower32,upper32)
        delivered[f'feat_{column}_outlier']=((values<lower)|(values>upper)).astype('float32')
        delivered[column]=repaired.fillna(float(observed.median()))
    return delivered.astype('float32')


def run(args):
    from main import _atomic_csv, _format_datetime, _result_frames
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    selection=json.loads(Path(args.selection).read_text(encoding='utf-8'))
    directory=discover(args.input_dir)
    test_paths=list(directory.parent.rglob('Pre_test_load.csv'))
    if len(test_paths)!=1:
        raise ValueError('Expected a single official Pre_test_load.csv')
    train=read_observations(directory)
    # Only its origin timestamps are used to choose the cutoff, not test values.
    test_times=pd.to_datetime(pd.read_csv(test_paths[0],usecols=['datetime']).datetime)
    cutoff=test_times.min()-pd.Timedelta(minutes=15)
    train=train.loc[:cutoff]
    weights=selection['weights']
    cfg=ModelConfig(**selection['config'])
    horizons=list(range(1,97))
    x_train=features(train)
    model=ResidualEnsemble(cfg).fit(train,x_train,cutoff,horizons,weights)
    trained_seconds=time.perf_counter()-start
    model.save(out/'forecast_model.joblib')
    pd.DataFrame(model.training_audit).to_csv(out/'training_audit.csv',index=False)
    write_json(out/'selection_frozen.json',selection)
    # Model/hyperparameters/weights are frozen before test observations are read.
    test=read_observations(test_paths[0].parent)
    combined=pd.concat([train,test]).sort_index()
    if combined.index.has_duplicates:
        raise ValueError('Train/test overlap after cutoff')
    combined=combined.reindex(pd.date_range(combined.index.min(),combined.index.max(),freq='15min',name='datetime'))
    x=features(combined)
    origins=pd.DatetimeIndex(test_times.drop_duplicates().sort_values(),name='datetime')
    tic=time.perf_counter()
    predictions=model.predict(combined,x,origins,weights)
    inference_seconds=time.perf_counter()-tic
    # Do not force total-group<=240 on noisy observations without validation:
    # group<=total is the only universally supported target consistency rule here.
    for h in horizons:
        group=f'generator_1_t+{15*h}_pred'; total=f'generator_all_t+{15*h}_pred'
        predictions[group]=np.minimum(predictions[group],predictions[total])
    ordered=[f'{t}_t+{15*h}_pred' for t in TARGETS for h in horizons]
    predictions=predictions[ordered]
    short,long=_result_frames(predictions)
    delivered_input=_format_datetime(_quality_input_frame(x,cutoff,train).loc[origins])
    for file,frame in [('s_result.csv',short),('l_result.csv',long),('input.csv',delivered_input)]:
        if frame.datetime.duplicated().any() or not np.isfinite(frame.drop(columns='datetime').to_numpy()).all():
            raise ValueError(f'Invalid submission: {file}')
        _atomic_csv(frame,out/file)
    # The delivered data dictionary additionally requests a split-orient JSON.
    (out/'s_result.json').write_text(short.to_json(orient='split',index=False),encoding='utf-8')
    metrics=[]
    clean=combined.ffill()
    for t in TARGETS:
        for h in horizons:
            actual=combined[t].reindex(shifted(origins,h)).to_numpy()
            pred=predictions[f'{t}_t+{15*h}_pred'].to_numpy()
            last=clean.loc[origins,t].to_numpy()
            for name,p in [('selected_ensemble',pred),('persistence',last)]:
                valid=np.isfinite(actual)&(np.abs(actual)>1e-8)
                metrics.append(dict(target=t,horizon=h,model=name,mape=mape(actual,p),
                    mae=float(np.mean(np.abs(actual[valid]-p[valid]))),scored=int(valid.sum()),
                    unavailable=int(np.isnan(actual).sum()),zeros=int(np.sum(actual==0))))
    scores=pd.DataFrame(metrics)
    _atomic_csv(scores,out/'test_metrics_by_horizon.csv')
    # Disjoint horizon bands (review_iter_2.md H4).  The old `long_96` averaged
    # h=1..96, which silently embedded the short-cycle score; these four are
    # reported independently and a "24h improvement" claim must hold on both
    # long_tail and h96_endpoint.
    def band(h):
        if h <= 8:
            return 'short_8'
        if h <= 48:
            return 'mid'
        return 'long_tail'
    summary=scores.assign(period=scores.horizon.map(band)).groupby(['period','target','model']).mape.mean().reset_index()
    endpoint=scores[scores.horizon==96].drop(columns='horizon').assign(period='h96_endpoint')
    summary=pd.concat([summary,endpoint],ignore_index=True)
    _atomic_csv(summary,out/'test_summary.csv')
    origin=origins.max()
    dates=pd.date_range(origin+pd.Timedelta(minutes=15),periods=96,freq='15min',name='datetime')
    tariff=PriceSchedule.from_excel(directory/'price.xlsx')
    net,caps,level,capacity,holder=observed_surplus(combined,origin,96)
    efficiency=GasResourceForecaster(combined.ffill().fillna(0)).estimate_efficiency(origin)
    cfg_dispatch=DispatchConfig(holder_capacity=capacity,solver_time_limit=args.solver_time_limit)
    plan=MILPDispatcher(cfg_dispatch).optimize(dates,tariff.prices(dates).to_numpy(),net,caps,efficiency,level)
    _atomic_csv(_format_datetime(plan.gas_plan),out/'opt_result.csv')
    _atomic_csv(_format_datetime(plan.audit),out/'dispatch_audit.csv')
    # Equal resource boundary and terminal inventory, flatter tariff benchmark.
    flat=MILPDispatcher(cfg_dispatch).optimize(dates,np.full(96,tariff.prices(dates).mean()),net,caps,efficiency,level)
    flat_revenue=float(np.sum(flat.power_path*tariff.prices(dates).to_numpy())*.25*1000)
    metadata=dict(training_cutoff=str(cutoff),origin_start=str(origins.min()),origin_end=str(origin),
        origins=len(origins),train=data_profile(train),test=data_profile(test),training_seconds=trained_seconds,
        inference_seconds=inference_seconds,seconds_per_origin=inference_seconds/len(origins),
        total_seconds=time.perf_counter()-start,dispatch_status=plan.status,dispatch=plan.diagnostics,
        flat_tariff_schedule_revenue=flat_revenue,selected_tariff_schedule_revenue=plan.diagnostics['revenue'],
        active_holder=holder,holder_capacity=capacity,
        dispatch_assumptions=['Generator gas flow assumed m3/h; dictionary omits units.',
            'Effective supply estimated from historical generator gas use plus measured BF inventory change.',
            'Missing holder 1 excluded; no unobserved coke/converter storage.',
            'Constraints hold for forecast resource boundary, not guaranteed realized production.'],
        score_definition='mean abs((y-p)/y), nonzero observed labels only; unknown future not scored',
        test_protocol='frozen model; rolling observations available only at/before each origin')
    write_json(out/'run_metadata.json',metadata)
    print(summary.to_string(index=False),flush=True)
    print(json.dumps(metadata,ensure_ascii=True,indent=2),flush=True)
    return metadata
