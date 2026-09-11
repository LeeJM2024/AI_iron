"""Causal residual ensembles with purged, out-of-time model selection."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

LOG = logging.getLogger(__name__)
TARGETS = ('generator_1', 'generator_all')
MEMBERS = ('persistence', 'ema', 'seasonal', 'lgb_residual', 'cat_residual')
BUCKETS = (2, 8, 24, 48, 96)


def shifted(index, steps):
    """Explicit ns arithmetic avoids pandas/NumPy 2.5 generic-timedelta warnings."""
    ix = pd.DatetimeIndex(index).as_unit('ns')
    return pd.DatetimeIndex(ix.asi8 + int(steps)*900_000_000_000, name=ix.name)


def bucket(h):
    return next(b for b in BUCKETS if h <= b)


def mape(actual, pred):
    """Unmodified observed labels; zero actuals are excluded and counted separately."""
    a, p = np.asarray(actual), np.asarray(pred)
    valid = np.isfinite(a) & np.isfinite(p) & (np.abs(a) > 1e-8)
    return float(np.mean(np.abs(a[valid]-p[valid])/np.abs(a[valid]))) if valid.any() else None


def features(raw):
    """All raw fields, explicit missingness, multi-scale state and change features."""
    clean = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    d = {c: clean[c] for c in clean}
    idx = clean.index
    for c in clean:
        s = clean[c]
        d['feat_'+c+'_missing'] = raw[c].isna().astype(float)
        # Do not suppress a shutdown as an outlier. A capped auxiliary view lets
        # the learner choose robust measurements without changing observed labels.
        past = s.shift(1).rolling(192, min_periods=24)
        lo, hi = past.quantile(.01), past.quantile(.99)
        d['feat_'+c+'_capped'] = s.clip(lower=lo, upper=hi).fillna(s)
        for n in (1, 4, 8, 24):
            d[f'feat_{c}_mean_{n}'] = s.rolling(n, min_periods=1).mean()
        d['feat_'+c+'_delta'] = s.diff().fillna(0)
    for target in TARGETS:
        s = clean[target]
        for n in (1,2,3,4,6,8,12,24,48,96,192,672):
            d[f'feat_{target}_lag_{n}'] = s.shift(n).fillna(0)
        for n in (4,8,24,96,672):
            w = s.rolling(n, min_periods=1)
            d[f'feat_{target}_std_{n}'] = w.std().fillna(0)
            d[f'feat_{target}_range_{n}'] = w.max()-w.min()
            d[f'feat_{target}_ema_{n}'] = s.ewm(span=n, adjust=False).mean()
        d[f'feat_{target}_regime_change'] = s-s.shift(96).fillna(s)
    for gas, prefix in [('blast_furnace','blast_furnace_'),('coke','coke_oven_'),('converter','converter_')]:
        prod = [c for c in clean if c.startswith(prefix) and c[len(prefix):].isdigit()]
        users = [c for c in clean if c.startswith(gas+'_user')]
        if gas == 'blast_furnace':
            users += [c for c in clean if c.startswith('air_heater_')]
        d[f'feat_{gas}_production'] = clean[prod].sum(axis=1)
        d[f'feat_{gas}_users'] = clean[users].sum(axis=1)
        # Separate gas types: the dictionary does not establish compatible units.
        d[f'feat_{gas}_balance_proxy'] = clean[prod].sum(axis=1)-clean[users].sum(axis=1)
    for n in (1,2):
        c = f'blast_furnace_gas_holder_{n}'
        if c in clean:
            cap = 200000 if n == 1 else 300000
            d[f'feat_holder_{n}_ratio'] = clean[c]/cap
            d[f'feat_holder_{n}_low_margin'] = clean[c]-.15*cap
            d[f'feat_holder_{n}_high_margin'] = .9*cap-clean[c]
    d['feat_other_power'] = clean.generator_all-clean.generator_1
    d['feat_hour_sin'] = np.sin(2*np.pi*(idx.hour*60+idx.minute)/1440)
    d['feat_hour_cos'] = np.cos(2*np.pi*(idx.hour*60+idx.minute)/1440)
    d['feat_dow'] = idx.dayofweek.astype(float)
    return pd.DataFrame(d, index=idx).astype('float32')


def horizon_features(x, raw, h):
    result = x.copy()
    idx = shifted(x.index, h)
    result['feat_target_hour_sin'] = np.sin(2*np.pi*(idx.hour*60+idx.minute)/1440)
    result['feat_target_hour_cos'] = np.cos(2*np.pi*(idx.hour*60+idx.minute)/1440)
    result['feat_target_dow'] = idx.dayofweek.astype(float)
    clean = raw.ffill().fillna(0)
    for t in TARGETS:
        for lag in (96,192,672):
            times = shifted(x.index, h-lag)
            result[f'feat_{t}_target_lag_{lag}'] = clean[t].reindex(times).to_numpy()
    return result.fillna(0).astype('float32')


@dataclass
class ModelConfig:
    trees: int = 180
    leaves: int = 15
    threads: int = 4
    train_days: int = 90
    recency_half_life_days: float | None = None
    recency_half_life_by_target: dict[str, float | None] | None = None
    seed: int = 2026


@dataclass
class SharedHorizonConfig:
    """Configuration for the pooled direct model used as a challenger member."""
    trees: int = 320
    leaves: int = 31
    threads: int = 4
    train_days: int = 90
    origin_stride: int = 2
    max_features: int = 160
    seed: int = 2026


class ResidualEnsemble:
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self.models = {}
        self.columns = []
        self.cutoff = None
        self.training_audit = []

    def fit(self, raw, x, cutoff, horizons, weights=None):
        cfg = self.config
        self.cutoff = pd.Timestamp(cutoff)
        # Schema/constant filtering is fitted on training time only.
        history_x = x.loc[:cutoff]
        self.columns = history_x.columns[history_x.nunique() > 1].tolist()
        start = self.cutoff-pd.Timedelta(int(cfg.train_days),unit='D')
        for h in horizons:
            h=int(h)
            z = horizon_features(x[self.columns], raw, h)
            label = raw[list(TARGETS)].shift(-h)
            end = self.cutoff-pd.Timedelta(h*15, unit='min')
            ix = x.index[(x.index >= start) & (x.index <= end)]
            for target in TARGETS:
                valid = label.loc[ix,target].notna() & raw.loc[ix,target].notna()
                rows = ix[valid]
                if len(rows) < 128:
                    raise ValueError('At least 128 observed training labels required')
                latest_label = shifted(rows[-1:], h)[0]
                if latest_label > self.cutoff:
                    raise AssertionError('Future target crossed training cutoff')
                self.training_audit.append(dict(target=target,horizon=int(h),rows=len(rows),
                    last_feature=str(rows[-1]),last_label=str(latest_label),cutoff=str(self.cutoff)))
                y = label.loc[rows,target].to_numpy()
                residual = y-raw.loc[rows,target].to_numpy()
                # Weighted MAE approximates MAPE, floor based solely on training.
                positive = y[y>0]
                floor = max(1., float(np.quantile(positive,.01))) if len(positive) else 1.
                w = 1/np.maximum(np.abs(y),floor)
                half_life=(cfg.recency_half_life_by_target.get(target)
                    if cfg.recency_half_life_by_target is not None else cfg.recency_half_life_days)
                if half_life is not None:
                    half_life=float(half_life)
                    if not np.isfinite(half_life) or half_life<=0:
                        raise ValueError('recency_half_life_days must be positive or None')
                    age_days=(rows[-1].value-rows.asi8)/(86_400*1_000_000_000)
                    w=w*np.exp2(-age_days/half_life)
                w = w/w.mean()
                lgb = LGBMRegressor(objective='regression_l1', n_estimators=cfg.trees,
                    num_leaves=cfg.leaves, learning_rate=.035, min_child_samples=60,
                    colsample_bytree=.85, reg_lambda=5., reg_alpha=.1,
                    random_state=cfg.seed, n_jobs=cfg.threads, verbosity=-1,
                    deterministic=True, force_col_wise=True)
                cat = CatBoostRegressor(loss_function='MAE', iterations=cfg.trees,
                    depth=5, learning_rate=.045, l2_leaf_reg=8,
                    thread_count=cfg.threads, random_seed=cfg.seed,
                    verbose=False, allow_writing_files=False)
                member_weights = weights.get(f'{target}/{bucket(h)}') if weights else None
                if member_weights is None or member_weights[3] > 1e-6:
                    lgb.fit(z.loc[rows],residual,sample_weight=w)
                else:
                    lgb = None
                if member_weights is None or member_weights[4] > 1e-6:
                    cat.fit(z.loc[rows],residual,sample_weight=w)
                else:
                    cat = None
                self.models[target,h] = (lgb,cat)
            LOG.info('Fitted residual members h=%s, cutoff=%s', h, cutoff)
        return self

    def predict_members(self, raw, x, origins, h, target):
        if pd.DatetimeIndex(origins).min() < self.cutoff:
            raise ValueError('Prediction origins precede training cutoff')
        clean = raw.ffill().fillna(0)
        now = clean.loc[origins,target].to_numpy()
        ema = clean[target].ewm(span=4,adjust=False).mean().loc[origins].to_numpy()
        seasonal = clean[target].reindex(shifted(origins,h-96)).to_numpy()
        seasonal = np.where(np.isfinite(seasonal),seasonal,now)
        z = horizon_features(x.loc[origins,self.columns],raw,h)
        learned = [now+m.predict(z) if m is not None else now for m in self.models[target,h]]
        return np.clip(np.column_stack([now,ema,seasonal,*learned]),0,200 if target==TARGETS[0] else 440)

    def predict(self, raw, x, origins, weights):
        d = {}
        for target,h in self.models:
            w = np.asarray(weights[f'{target}/{bucket(h)}'])
            d[f'{target}_t+{h*15}_pred'] = self.predict_members(raw,x,origins,h,target) @ w
        return pd.DataFrame(d,index=origins).sort_index(axis=1)

    def save(self,path):
        joblib.dump(self,path,compress=3)


class SharedHorizonResidual:
    """One direct residual model per target, pooled across forecast horizons.

    The horizon and target-time calendar are features.  This shares information
    between nearby horizons while every row still has its own causal origin and
    purged future label.  It is intentionally a separate challenger until it
    proves stable on out-of-time validation.
    """
    def __init__(self, config=None):
        self.config = config or SharedHorizonConfig()
        self.columns = []
        self.models = {}
        self.cutoff = None
        self.training_audit = []

    def _select_columns(self, x, cutoff):
        history=x.loc[:cutoff]
        nonconstant=set(history.columns[history.nunique()>1])
        priority=[]
        for c in x.columns:
            if (c in TARGETS or c == 'feat_other_power' or c.startswith('feat_generator_')
                    or c.startswith('feat_holder_') or c.startswith('feat_blast_furnace_')
                    or c.startswith('feat_coke_') or c.startswith('feat_converter_')):
                priority.append(c)
        # Keep measured current gas / holder / user state after the target series.
        priority += [c for c in x.columns if not c.startswith('feat_')]
        chosen=[]
        for c in priority:
            if c in nonconstant and c not in chosen:
                chosen.append(c)
            if len(chosen)>=self.config.max_features:
                break
        if not chosen:
            raise ValueError('No nonconstant pooled-model features in training history')
        return chosen

    @staticmethod
    def _augment(z, h):
        out=z.copy()
        fraction=float(h)/96.
        out['feat_horizon_steps']=float(h)
        out['feat_horizon_fraction']=fraction
        out['feat_horizon_sin']=np.sin(2*np.pi*fraction)
        out['feat_horizon_cos']=np.cos(2*np.pi*fraction)
        return out

    def _training_matrix(self, raw, x, horizons):
        cfg=self.config
        start=self.cutoff-pd.Timedelta(int(cfg.train_days),unit='D')
        matrices=[]
        labels={target:[] for target in TARGETS}
        audits=[]
        for h in horizons:
            h=int(h)
            end=self.cutoff-pd.Timedelta(h*15, unit='min')
            rows=x.index[(x.index>=start)&(x.index<=end)][::cfg.origin_stride]
            if len(rows)==0:
                continue
            future=shifted(rows,h)
            valid=raw.loc[rows,list(TARGETS)].notna().all(axis=1).to_numpy()
            valid &= raw[list(TARGETS)].reindex(future).notna().all(axis=1).to_numpy()
            rows=rows[valid]
            future=future[valid]
            if len(rows)<64:
                continue
            latest_label=future[-1]
            if latest_label>self.cutoff:
                raise AssertionError('Future target crossed pooled-model training cutoff')
            z=self._augment(horizon_features(x.loc[rows,self.columns],raw,h),h)
            matrices.append(z)
            for target in TARGETS:
                labels[target].append((raw[target].reindex(future).to_numpy()
                    -raw.loc[rows,target].to_numpy()).astype('float32'))
                audits.append(dict(target=target,horizon=int(h),rows=len(rows),
                    last_feature=str(rows[-1]),last_label=str(latest_label),cutoff=str(self.cutoff)))
        if not matrices:
            raise ValueError('No valid pooled-model training rows')
        self.training_audit=audits
        return pd.concat(matrices,axis=0,ignore_index=True), {t:np.concatenate(v) for t,v in labels.items()}

    def fit(self, raw, x, cutoff, horizons=range(1,97)):
        self.cutoff=pd.Timestamp(cutoff)
        horizons=tuple(int(h) for h in horizons)
        if not horizons or min(horizons)<1 or max(horizons)>96:
            raise ValueError('Pooled model horizons must be within 1..96')
        self.columns=self._select_columns(x,self.cutoff)
        matrix,residuals=self._training_matrix(raw,x,horizons)
        for target in TARGETS:
            y=residuals[target]
            future=y+0 # retain float32 before deriving target-scale weights below
            # Residual magnitude is not the MAPE denominator. Recover the actual
            # scale from the matching base target encoded in the feature matrix.
            base=matrix[target].to_numpy(dtype=float)
            actual=base+y
            positive=actual[actual>0]
            floor=max(1.,float(np.quantile(positive,.01))) if len(positive) else 1.
            weight=1/np.maximum(np.abs(actual),floor)
            weight=weight/weight.mean()
            model=LGBMRegressor(objective='regression_l1',n_estimators=self.config.trees,
                num_leaves=self.config.leaves,learning_rate=.03,min_child_samples=80,
                colsample_bytree=.8,reg_lambda=8.,reg_alpha=.2,
                random_state=self.config.seed,n_jobs=self.config.threads,
                verbosity=-1,deterministic=True,force_col_wise=True)
            model.fit(matrix,y,sample_weight=weight)
            self.models[target]=model
        LOG.info('Fitted pooled shared-horizon residual model, rows=%s, cutoff=%s',len(matrix),self.cutoff)
        return self

    def predict(self, raw, x, origins, h, target):
        origins=pd.DatetimeIndex(origins)
        if origins.min()<self.cutoff:
            raise ValueError('Prediction origins precede pooled-model training cutoff')
        if target not in TARGETS or h<1 or h>96:
            raise ValueError('Invalid pooled-model target or horizon')
        z=self._augment(horizon_features(x.loc[origins,self.columns],raw,h),h)
        now=raw.ffill().fillna(0).loc[origins,target].to_numpy()
        upper=200 if target==TARGETS[0] else 440
        return np.clip(now+self.models[target].predict(z),0,upper)

    def save(self,path):
        joblib.dump(self,path,compress=3)


def select_weights(records):
    weights = {}
    for target in TARGETS:
        for b in BUCKETS:
            rows = [r for r in records if r['target']==target and bucket(r['h'])==b]
            if not rows:
                continue
            y = np.concatenate([r['y'] for r in rows])
            p = np.concatenate([r['p'] for r in rows])
            ok = np.isfinite(y) & (np.abs(y)>1e-8)
            y,p = y[ok],p[ok]
            # Convex OOF stacking; tiny L2 stabilizes weights across regimes.
            def loss(w):
                return np.mean(np.abs(y-p@w)/np.abs(y)) + .0002*np.sum(w*w)
            choices=[]
            for initial in (np.ones(len(MEMBERS))/len(MEMBERS),np.eye(len(MEMBERS))[0]):
                res = minimize(loss,initial,method='SLSQP',bounds=[(0.,1.)]*len(MEMBERS),
                    constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':250,'ftol':1e-10})
                choices.append(res.x if res.success else initial)
            w = min(choices,key=loss)
            weights[f'{target}/{b}'] = (w/w.sum()).tolist()
    return weights
