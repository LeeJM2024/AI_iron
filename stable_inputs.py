"""Training-fitted active schema and causal, bounded industrial feature views.

This is model preprocessing, not a post-hoc edit of submission inputs. Raw labels
stay untouched. Sensor ranges, transformations, inactive columns and versions are
stored on the transformer and re-used identically at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prelim import InputTransform, TARGETS, UPPER


class StableInputTransform(InputTransform):
    def __init__(self, active_days=7, robust_raw=False):
        self.active_days = int(active_days)
        if self.active_days < 1:
            raise ValueError('active_days must be positive')
        self.robust_raw = bool(robust_raw)

    def fit(self, raw):
        for target in TARGETS:
            if target not in raw or not raw[target].notna().any():
                raise ValueError(f'Missing observed training target: {target}')
        # Never use the future submission interval to decide which columns exist.
        self.fit_cutoff = raw.index.max()
        recent = raw.tail(self.active_days*96)
        self.columns = [c for c in raw if c in TARGETS or
                        (raw[c].notna().any() and recent[c].nunique() > 1)]
        self.excluded = {c: ('all_missing' if raw[c].notna().sum() == 0 else
                             f'inactive_last_{self.active_days}_training_days')
                         for c in raw if c not in self.columns}
        self.bounds = {}
        for c in self.columns:
            s = raw[c].replace([np.inf,-np.inf],np.nan).dropna()
            q1,q3 = s.quantile([.25,.75])
            self.bounds[c] = (0., UPPER.get(c,max(float(s.quantile(.999)),float(q3+6*(q3-q1)),1.)))
        self.change_scales = {}
        for c in self.columns:
            s = raw[c].ffill()
            # Units stay in the raw fields. Change transforms are dimensionless.
            self.change_scales[c] = max(float(s.diff(4).abs().median()),
                                        float(s.abs().median())*.005, 1e-4)
        frame = self._build(raw)
        recent_features = frame.tail(self.active_days*96)
        keep = recent_features.columns[(recent_features.nunique() > 1) | recent_features.columns.isin(TARGETS)]
        # Preserve the first column in each identical series; no fake jitter.
        self.feature_columns = keep[(~recent_features[keep].T.duplicated()).to_numpy() | keep.isin(TARGETS)].tolist()
        self.excluded_features = [c for c in frame if c not in self.feature_columns]
        return self

    def _build(self, raw):
        working = raw.copy()
        if self.robust_raw:
            # Winsorize non-load inputs using previous 24h only. Output retains
            # original units. Load labels/current load never get quantile-clipped.
            for c in self.columns:
                if c in TARGETS:
                    continue
                s = working[c].replace([np.inf, -np.inf], np.nan)
                history = s.shift(1).rolling(96, min_periods=32)
                lo, hi = history.quantile(.025), history.quantile(.975)
                working[c] = s.clip(lower=lo, upper=hi)
        frame = super()._build(working)
        for c in frame:
            if c.startswith('feat_clock_'):
                # Same periodic signal, explicit nonnegative [0,1] encoding.
                frame[c] = (frame[c]+1.)*.5
            elif '_change_' in c:
                sensor = c[len('feat_'):].rsplit('_change_', 1)[0]
                scale = self.change_scales[sensor]
                frame[c] = .5+.5*np.tanh(frame[c]/(3.*scale))
            elif c.startswith('feat_') and '_std_' in c:
                # Heavy-tailed volatility: compress without erasing its ordering.
                frame[c] = np.log1p(frame[c])
        return frame.astype('float32')
