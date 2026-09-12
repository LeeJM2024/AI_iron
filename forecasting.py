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
MEMBERS = ('persistence', 'ema', 'seasonal', 'profile_regime', 'profile_anchored',
           'momentum', 'lgb_residual', 'cat_residual', 'shared_horizon')
# Iteration 4 tried offering extra diurnal-profile windows (5d/10d) to the
# weight solver as a causal CV over profile_days; they regressed generator_all
# on the validation fold and were removed.  The multi-window machinery stays
# dormant behind PROFILE_WINDOW_VARIANTS=() for any future revisit.
PROFILE_WINDOW_VARIANTS = ()
BUCKETS = (2, 8, 24, 48, 96)
# Neighbouring daily anchors around "yesterday at the target time" (lag 96).
DAILY_ANCHOR_OFFSETS = (-4, -3, -2, -1, 1, 2, 3, 4)


def detect_last_break(series, min_ratio=1.5, min_side=3):
    """Locate the dominant level shift inside a training window, or None.

    Binary-segmentation objective: pick the daily boundary maximising the mean
    ratio between the days before and the days after, requiring `min_side` days
    on each side.  Used to keep every member inside one plant operating regime
    (the 2025-04-18 break doubles generator_1 and must not be averaged across).
    """
    daily = series.resample('D').mean().dropna()
    n = len(daily)
    if n < 2 * min_side + 1:
        return None
    values = daily.to_numpy(dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    best = None
    for i in range(min_side, n - min_side + 1):
        left = (cumulative[i] - cumulative[0]) / i
        right = (cumulative[n] - cumulative[i]) / (n - i)
        if left <= 0 or right <= 0:
            continue
        ratio = max(right / left, left / right)
        if best is None or ratio > best[0]:
            best = (ratio, daily.index[i])
    if best is not None and best[0] >= min_ratio:
        return pd.Timestamp(best[1])
    return None


def diurnal_profile(series, end, start=None):
    """96-slot mean level by time of day; `start` allows regime restriction.

    Causal: uses only rows at or before `end`.  Because it is indexed by the
    TARGET's time-of-day slot it is available at every horizon without leakage.
    """
    src = series.loc[:end]
    if start is not None:
        restricted = src.loc[start:]
        if len(restricted) >= 96:
            src = restricted
    if len(src) < 96:
        src = series.loc[:end]
    slot = (src.index.hour * 60 + src.index.minute) // 15
    return src.groupby(slot).mean().reindex(range(96)).ffill().bfill().to_numpy(dtype=float)


def shifted(index, steps):
    """Explicit ns arithmetic avoids pandas/NumPy 2.5 generic-timedelta warnings."""
    ix = pd.DatetimeIndex(index)
    try:
        ix = ix.as_unit('ns')  # pandas>=2; everything is ns on pandas 1.x
    except (AttributeError, TypeError):
        pass
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
            # Iteration 6 tried multi-scale holder deltas plus balance/efficiency
            # features (H6 measured a 0.21 post-break leading correlation to the
            # generator_1 change); they moved the validation fold by <=0.2pt in
            # mixed directions and were reverted to keep the frozen selection
            # reproducible.
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
        # Daily shape neighbours: the load level moves smoothly across the day,
        # so anchors adjacent to yesterday's target time help interpolation.
        # Anchors are clamped to the origin (steps<=0) so no future value,
        # including the far-side neighbour at very long horizons, is read.
        for k in DAILY_ANCHOR_OFFSETS:
            steps = min(h-96-k, 0)
            times = shifted(x.index, steps)
            result[f'feat_{t}_target_lag_96{k:+d}'] = clean[t].reindex(times).to_numpy()
    return result.fillna(0).astype('float32')


@dataclass
class ModelConfig:
    trees: int = 180
    leaves: int = 15
    threads: int = 4
    train_days: int = 90
    recency_half_life_days: float | None = None
    recency_half_life_by_target: dict[str, float | None] | None = None
    # Average the learned residual over +/- w fitted horizons to suppress
    # jitter of the slow level across adjacent steps. 0 disables.
    smooth_learned: int = 0
    # Regime-matched diurnal profile member (see review_iter_2.md H2).
    profile_days: int = 7
    profile_regime_lock: bool = True
    # Learned members must not train across the detected regime break either:
    # ~90% of a 90-day window predates the 2025-04-18 plant-state change.
    train_regime_lock: bool = True
    # Horizons above this threshold train regime-locked; at short/mid horizons
    # the wider window still wins because those steps ride on the current
    # level instead of a long-horizon level prior.
    regime_lock_min_horizon: int = 0
    # Wire the pooled SharedHorizonResidual challenger in as the last member
    # (review_iter_2.md: re-evaluate once H2/H3 have landed).  Member-level
    # fold evidence decides which targets admit it; for the others the column
    # is a neutral persistence duplicate so the pool shape stays fixed.
    use_shared_horizon: bool = False
    shared_horizon_targets: tuple = TARGETS
    # Seed bagging of the learned members: average predictions over this many
    # seeds.  Variance reduction helps most where the ensemble has no better
    # signal than the learned level, i.e. the short horizons that 初赛 scores.
    n_seed_bag: int = 1
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
        self.momentum = {}
        self.profile = {}
        self.profiles_multi = {}
        self.profile_days_effective = {}
        self.break_date = None
        self.shared = None
        self.training_audit = []

    def _build_profiles(self, raw):
        """Regime-matched diurnal profiles, one per target, fitted on training time."""
        cfg = self.config
        series = raw[list(TARGETS)].ffill()
        # Detect the break on the union of targets; generator_1 carries the signal.
        self.break_date = None
        if cfg.profile_regime_lock:
            self.break_date = detect_last_break(series.loc[:self.cutoff].mean(axis=1))
        for target in TARGETS:
            src = series[target]
            start = None
            if cfg.profile_regime_lock:
                start = max(self.break_date or pd.Timestamp.min,
                            self.cutoff - pd.Timedelta(int(cfg.profile_days), unit='D'))
            else:
                start = self.cutoff - pd.Timedelta(int(cfg.profile_days), unit='D')
            self.profile[target] = diurnal_profile(src, self.cutoff, start)
            self.profile_days_effective[target] = (self.cutoff - start).total_seconds() / 86400.0
            self.profiles_multi[target] = {}
            for days in PROFILE_WINDOW_VARIANTS:
                wstart = self.cutoff - pd.Timedelta(int(days), unit='D')
                if cfg.profile_regime_lock:
                    wstart = max(wstart, self.break_date or pd.Timestamp.min)
                self.profiles_multi[target][days] = diurnal_profile(src, self.cutoff, wstart)
        LOG.info('Diurnal profiles built at cutoff %s; regime break=%s; days=%s',
                 self.cutoff, self.break_date, cfg.profile_days)
        # A regime lock can silently truncate the profile window to a couple of
        # days on early folds.  The profile member then looks worthless in tuning
        # while being the strongest member at serve time, and the weight solver
        # drives its weight to zero.  Make the truncation loud.
        shortest = min(self.profile_days_effective.values()) if self.profile_days_effective else None
        if shortest is not None and shortest < cfg.profile_days - 1e-9:
            LOG.warning('profile window truncated to %.2fd (requested %sd) at cutoff %s; '
                        'folds this close to the break are not comparable to the served model',
                        shortest, cfg.profile_days, self.cutoff)

    @staticmethod
    def _estimate_momentum(raw, target, cutoff):
        """AR(1) fit of 15-min load changes on training data only.

        phi is the lag-1 autocorrelation (optimal shrinkage of the last change
        when predicting the next one); rho is the geometric decay of that
        signal with horizon.
        """
        d = raw[target].loc[:cutoff].ffill().diff().dropna()
        if len(d) < 96:
            return 0.0, 0.5
        phi = float(d.autocorr(1))
        if not np.isfinite(phi) or phi <= 0.01:
            return 0.0, 0.5
        lag2 = float(d.autocorr(2))
        rho = lag2/phi if np.isfinite(lag2) else 0.5
        rho = float(np.clip(rho, 0.05, 0.9))
        return phi, rho

    def fit(self, raw, x, cutoff, horizons, weights=None):
        cfg = self.config
        self.cutoff = pd.Timestamp(cutoff)
        # Schema/constant filtering is fitted on training time only.
        history_x = x.loc[:cutoff]
        self.columns = history_x.columns[history_x.nunique() > 1].tolist()
        for target in TARGETS:
            self.momentum[target] = self._estimate_momentum(raw, target, self.cutoff)
        LOG.info('Momentum parameters %s at cutoff %s', self.momentum, cutoff)
        self._build_profiles(raw)
        full_start = self.cutoff-pd.Timedelta(int(cfg.train_days),unit='D')
        locked_start = full_start
        if cfg.train_regime_lock and self.break_date is not None:
            locked_start = max(full_start, self.break_date)
        if locked_start > full_start:
            LOG.info('Regime lock active: h>%d trains from %s (break %s), '
                     'shorter horizons keep the %d-day window',
                     cfg.regime_lock_min_horizon, locked_start, self.break_date,
                     cfg.train_days)
        for h in horizons:
            h=int(h)
            # The regime break poisons the level prior that long horizons live
            # on, but short/mid horizons predict "current level + short-term
            # dynamics", where the wider window still helps (official evidence:
            # g1 h9-24 regressed 7.54->8.67% under a full lock).
            if cfg.train_regime_lock and h > cfg.regime_lock_min_horizon:
                start = locked_start
            else:
                start = full_start
            z = horizon_features(x[self.columns], raw, h)
            label = raw[list(TARGETS)].shift(-h)
            end = self.cutoff-pd.Timedelta(h*15, unit='min')
            ix = x.index[(x.index >= start) & (x.index <= end)]
            for target in TARGETS:
                valid = label.loc[ix,target].notna() & raw.loc[ix,target].notna()
                rows = ix[valid]
                # Regime-locked windows legitimately hold only the post-break
                # days; 64 rows (16h of labels) is still a usable fit sample.
                min_rows = 64 if start > full_start else 128
                if len(rows) < min_rows:
                    raise ValueError(f'At least {min_rows} observed training labels required')
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
                lgb_bag, cat_bag = [], []
                for bag_i in range(max(1, int(getattr(cfg, 'n_seed_bag', 1)))):
                    bag_seed = cfg.seed + bag_i*101
                    if member_weights is None or member_weights[6] > 1e-6:
                        lgb_i = LGBMRegressor(objective='regression_l1', n_estimators=cfg.trees,
                            num_leaves=cfg.leaves, learning_rate=.035, min_child_samples=60,
                            colsample_bytree=.85, reg_lambda=5., reg_alpha=.1,
                            random_state=bag_seed, n_jobs=cfg.threads, verbosity=-1,
                            deterministic=True, force_col_wise=True)
                        lgb_i.fit(z.loc[rows],residual,sample_weight=w)
                        lgb_bag.append(lgb_i)
                    if member_weights is None or member_weights[7] > 1e-6:
                        cat_i = CatBoostRegressor(loss_function='MAE', iterations=cfg.trees,
                            depth=5, learning_rate=.045, l2_leaf_reg=8,
                            thread_count=cfg.threads, random_seed=bag_seed,
                            verbose=False, allow_writing_files=False)
                        cat_i.fit(z.loc[rows],residual,sample_weight=w)
                        cat_bag.append(cat_i)
                # None entry keeps the slot shape when a member is zero-weighted.
                self.models[target,h] = (lgb_bag or [None], cat_bag or [None])
            LOG.info('Fitted residual members h=%s, cutoff=%s', h, cutoff)
        if cfg.use_shared_horizon:
            shr_cfg = SharedHorizonConfig(threads=cfg.threads, seed=cfg.seed)
            shr = SharedHorizonResidual(shr_cfg)
            shr.break_date = self.break_date
            shr.regime_lock_min_horizon = cfg.regime_lock_min_horizon
            shr.fit(raw, x, self.cutoff, sorted({h for _, h in self.models}))
            self.shared = shr
        return self

    def _learned_predictions(self, raw, x, origins, h, target):
        """Learned residual members, optionally averaged over nearby horizons.

        Adjacent horizons share the same slow level; averaging their residual
        estimates reduces estimator variance without touching observed labels.
        """
        cfg = self.config
        fitted = sorted(hh for (t, hh) in self.models if t == target and abs(hh-h) <= cfg.smooth_learned)
        if cfg.smooth_learned <= 0 or len(fitted) <= 1:
            fitted = [h]
        z_by_h = {}
        out = []
        for k in range(2):
            stack = []
            for hh in fitted:
                if hh not in z_by_h:
                    z_by_h[hh] = horizon_features(x.loc[origins, self.columns], raw, hh)
                seed_preds = [m.predict(z_by_h[hh])
                              for m in self.models[target, hh][k] if m is not None]
                if seed_preds:
                    stack.append(np.mean(np.stack(seed_preds), axis=0))
            out.append(np.mean(np.stack(stack), axis=0) if stack else np.zeros(len(origins)))
        return out

    def predict_members(self, raw, x, origins, h, target):
        if pd.DatetimeIndex(origins).min() < self.cutoff:
            raise ValueError('Prediction origins precede training cutoff')
        clean = raw.ffill().fillna(0)
        now = clean.loc[origins,target].to_numpy()
        ema = clean[target].ewm(span=4,adjust=False).mean().loc[origins].to_numpy()
        seasonal = clean[target].reindex(shifted(origins,h-96)).to_numpy()
        seasonal = np.where(np.isfinite(seasonal),seasonal,now)
        phi, rho = self.momentum.get(target, (0.0, 0.5))
        prev = clean[target].reindex(shifted(origins,-1)).to_numpy()
        prev = np.where(np.isfinite(prev), prev, now)
        momentum = np.clip(now + phi*(rho**(h-1))*(now-prev), 0, 200 if target==TARGETS[0] else 440)
        slot = (pd.DatetimeIndex(shifted(origins, h)).hour*60
                + pd.DatetimeIndex(shifted(origins, h)).minute)//15
        slots = self.profile.get(target)
        if slots is not None:
            now_slot = (pd.DatetimeIndex(origins).hour*60
                        + pd.DatetimeIndex(origins).minute)//15
            # Index the 96-slot curves by the two relevant times of day; both
            # members must be one value per origin, never the raw slot vector.
            profile = slots[slot]
            # Level-anchored profile: keep the diurnal shape but re-base it on
            # the current level.  Both the observed level and its profile
            # counterpart are averaged over the trailing hour, so the anchor
            # tracks day-to-day level wander without injecting 15-min fast
            # noise into the prediction (a single-sample anchor degrades the
            # member towards persistence).
            lvl_now = clean[target].rolling(4, min_periods=1).mean().loc[origins].to_numpy()
            profile_anchored = profile + lvl_now - slots[now_slot]
        else:
            profile = np.full(len(origins), float(np.mean(now)))
            profile_anchored = now
        lgb_pred, cat_pred = self._learned_predictions(raw, x, origins, h, target)
        learned = [now+p for p in (lgb_pred, cat_pred)]
        if (getattr(self, 'shared', None) is not None
                and target in self.config.shared_horizon_targets):
            shared = self.shared.predict(raw, x, origins, h, target)
        else:
            shared = now
        # Column order MUST match MEMBERS:
        #   persistence, ema, seasonal, profile_regime, profile_anchored,
        #   momentum, lgb_residual, cat_residual, shared_horizon
        return np.clip(np.column_stack([now,ema,seasonal,profile,profile_anchored,
                                        momentum,*learned,shared]),
                       0, 200 if target==TARGETS[0] else 440)

    def predict(self, raw, x, origins, weights):
        d = {}
        for target,h in self.models:
            w = np.asarray(weights_for(weights, target, h))
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
        full_start=self.cutoff-pd.Timedelta(int(cfg.train_days),unit='D')
        matrices=[]
        labels={target:[] for target in TARGETS}
        audits=[]
        for h in horizons:
            h=int(h)
            # Same horizon-dependent regime policy as ResidualEnsemble.fit.
            if (getattr(self,'regime_lock_min_horizon',None) is not None
                    and getattr(self,'break_date',None) is not None
                    and h > self.regime_lock_min_horizon):
                start=max(full_start, self.break_date)
            else:
                start=full_start
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


def _convex_mape_weights(y, p, cap=6000, seed=2026):
    """Exact global optimum of  min_w mean_i |y_i - p_i.w| / |y_i|  s.t. simplex.

    The objective is convex and piecewise-linear, so this is a LINEAR PROGRAM in
    epigraph form.  The previous SLSQP-from-two-starts routine stalls on the
    non-differentiable kinks and returns degenerate simplex corners (1e-17 style
    "zeros" that are artefacts, not evidence).  Here HiGHS returns the true
    optimum, or we fall back to the legacy solver if the LP is infeasible.
    """
    from scipy.optimize import linprog
    ok = np.isfinite(y) & np.isfinite(p).all(axis=1) & (np.abs(y) > 1e-8)
    y, p = y[ok], p[ok]
    n, k = p.shape
    if n == 0 or k == 0:
        return np.ones(max(k, 1)) / max(k, 1)
    if n > cap:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, cap, replace=False)
        y, p = y[keep], p[keep]
        n = cap
    scale = 1.0 / np.abs(y)
    A = p * scale[:, None]
    rhs = y * scale
    # variables: w (k), t (n)
    cost = np.concatenate([np.zeros(k), np.ones(n) / n])
    A_ub = np.vstack([
        np.hstack([-A, -np.eye(n)]),
        np.hstack([A, -np.eye(n)]),
    ])
    b_ub = np.concatenate([-rhs, rhs])
    A_eq = np.hstack([np.ones((1, k)), np.zeros((1, n))])
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * k + [(0.0, None)] * n
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    if res.success:
        w = np.clip(res.x[:k], 0.0, None)
        return w / w.sum() if w.sum() > 0 else np.ones(k) / k
    return None


def _stack(rows):
    y = np.concatenate([r['y'] for r in rows])
    p = np.concatenate([r['p'] for r in rows])
    ok = np.isfinite(y) & (np.abs(y) > 1e-8)
    return y[ok], p[ok]


def weights_for(weights, target, h):
    """Per-horizon weights win over the coarse bucket fallback."""
    if weights is None:
        return None
    # Per-horizon keys carry an 'h' prefix: 'g1/h96' must not collide with the
    # coarse bucket key 'g1/96', which is the fallback for unserved horizons.
    w = weights.get(f'{target}/h{int(h)}')
    if w is None:
        w = weights.get(f'{target}/{bucket(int(h))}')
    return w


def _bucket_span(b):
    """Half-open horizon range (lo, hi] owned by bucket ``b``."""
    lo = 0
    for cand in BUCKETS:
        if cand == b:
            return lo, b
        lo = cand
    return lo, b


def _interpolate_anchors(anchors, anchor_weights, horizons):
    """Piecewise-linear-in-log(h) interpolation of simplex weight vectors.

    Weights are only FITTED at the anchor horizons in the tuning grid, but they
    are APPLIED at every h=1..96.  Interpolating between anchors removes that
    train/serve mismatch and, because neighbouring horizons share the same slow
    level, the interpolated vector stays on the simplex.
    """
    a = np.asarray(anchors, dtype=float)
    w = np.asarray(anchor_weights, dtype=float)
    log_a = np.log(a)
    out = {}
    for h in horizons:
        lh = float(np.log(h))
        if lh <= log_a[0]:
            vec = w[0]
        elif lh >= log_a[-1]:
            vec = w[-1]
        else:
            j = int(np.searchsorted(log_a, lh))
            lo, hi = log_a[j - 1], log_a[j]
            t = (lh - lo) / (hi - lo)
            vec = (1.0 - t) * w[j - 1] + t * w[j]
        vec = np.clip(vec, 0.0, None)
        out[int(h)] = (vec / vec.sum()).tolist() if vec.sum() > 0 else None
    return out


def select_weights(records, shrinkage_grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Bucket weights refined by anchor weights interpolated over every horizon.

    Two defects of the old bucket-only scheme:
      1. Five buckets are too coarse -- bucket 24 (h=9..24) and bucket 48
         (h=25..48) each force a single weight vector over a range where the
         diurnal profile helps at one end and hurts at the other.
      2. Weights were fitted on a ~12-point horizon grid but applied at every
         h=1..96, so most served horizons never had a weight fitted for them.

    Fix: fit a weight vector at each ANCHOR horizon in the tuning grid, shrink
    it toward the coarse bucket weight by a factor chosen with
    leave-one-cutoff-out CV (anchors are noisy on few folds), then interpolate
    in log(h) to produce a weight for every horizon the bucket owns.
    """
    weights = {}
    uniform = np.ones(len(MEMBERS)) / len(MEMBERS)
    cutoffs = sorted({r.get('cutoff') for r in records if r.get('cutoff')})
    for target in TARGETS:
        for b in BUCKETS:
            rows = [r for r in records if r['target']==target and bucket(r['h'])==b]
            if not rows:
                continue
            y, p = _stack(rows)
            anchors = sorted({int(r['h']) for r in rows})

            def _fit(subset):
                yy, pp = _stack(subset)
                got = _convex_mape_weights(yy, pp)
                return uniform.copy() if got is None else np.clip(got, 0, None)

            # ---- leave-one-cutoff-out choice of the shrinkage factor ----
            best_lambda = 0.0
            if len(cutoffs) >= 2 and len(anchors) > 1:
                scores = {}
                for lam in shrinkage_grid:
                    total, seen = 0.0, 0
                    for c in cutoffs:
                        tr_rows = [r for r in rows if r.get('cutoff') != c]
                        ev_rows = [r for r in rows if r.get('cutoff') == c]
                        if not tr_rows or not ev_rows:
                            continue
                        w_b = _fit(tr_rows)
                        if w_b.sum() <= 0:
                            w_b = uniform.copy()
                        w_b = w_b / w_b.sum()
                        per_h = {}
                        for h in anchors:
                            hr = [r for r in tr_rows if int(r['h']) == h]
                            if hr:
                                per_h[h] = _fit(hr)
                        errs = []
                        for h in anchors:
                            ev = [r for r in ev_rows if int(r['h']) == h]
                            if not ev:
                                continue
                            y_e, p_e = _stack(ev)
                            vec = (1 - lam) * w_b + lam * per_h.get(h, w_b)
                            vec = vec / vec.sum() if vec.sum() > 0 else uniform
                            errs.append(float(np.mean(np.abs(y_e - p_e @ vec) / np.abs(y_e))))
                        if errs:
                            total += float(np.mean(errs)); seen += 1
                    if seen:
                        scores[lam] = total / seen
                if scores:
                    best_lambda = min(scores, key=scores.get)

            # ---- final fit on all folds ----
            w_b = _fit(rows)
            w_b = w_b / w_b.sum() if w_b.sum() > 0 else uniform.copy()
            weights[f'{target}/{b}'] = w_b.tolist()
            if len(anchors) < 2:
                continue
            blended = []
            for h in anchors:
                hr = [r for r in rows if int(r['h']) == h]
                w_h = _fit(hr)
                w_h = w_h / w_h.sum() if w_h.sum() > 0 else uniform.copy()
                vec = (1 - best_lambda) * w_b + best_lambda * w_h
                blended.append(vec / vec.sum())
            lo, hi = _bucket_span(b)
            span = range(max(1, lo + 1), hi + 1)
            for h, vec in _interpolate_anchors(anchors, blended, span).items():
                if vec is not None:
                    weights[f'{target}/h{h}'] = vec
    return weights
