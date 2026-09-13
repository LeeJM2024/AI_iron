"""Compact ARX input: power history, fuel actually burned and measured inventory.

The explicit feature list is shared by training, inference and submission.
Upstream production/user telemetry is omitted in this ablation because generator
fuel consumption and holder inventory summarize the immediately relevant state.
Earlier rounds have already exposed May input-quality diagnostics; this is not a
blind architecture test. Fitted parameters use only training history, and no
observations or labels are edited on disk.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from prelim import TARGETS
from stable_inputs import StableInputTransform
from sklearn.preprocessing import QuantileTransformer


CORE = (*TARGETS, 'generator_use_blast_furnace_gas',
        'generator_use_coke_gas','generator_use_converter_gas',
        'blast_furnace_gas_holder_2')


class CompactInputTransform(StableInputTransform):
    def __init__(self,active_days=7,robust_raw=True,smooth_inputs=False,
                 dynamic_features=False,quantile_features=False):
        super().__init__(active_days,robust_raw)
        self.smooth_inputs=bool(smooth_inputs)
        self.dynamic_features=bool(dynamic_features)
        self.quantile_features=bool(quantile_features)

    def fit(self,raw):
        self.omitted = [c for c in raw if c not in CORE]
        result=super().fit(raw[[c for c in CORE if c in raw]])
        self.excluded.update({c:'core_signal_ablation' for c in self.omitted})
        self.source_feature_columns=list(self.feature_columns)
        if self.quantile_features:
            self.rank_columns=[c for c in self.feature_columns if c.startswith('feat_') and not c.startswith('feat_clock_')]
            frame=self._build(raw)[self.rank_columns].tail(14*96)
            self.quantile=QuantileTransformer(n_quantiles=min(512,len(frame)),
                output_distribution='uniform',random_state=2026)
            self.quantile.fit(frame)
            self.feature_columns=[c+'_quantile' if c in self.rank_columns else c for c in self.source_feature_columns]
        return result

    def _build(self,raw):
        frame=super()._build(raw)
        # Multi-scale load levels give the learner trends without separate,
        # heavy-tailed std/change fields. Lag 24 is not part of this fixed grid.
        keep=[c for c in frame if c in self.columns or c=='feat_other_power' or
              c.startswith('feat_clock_') or
              (any(c.startswith(f'feat_{t}_') for t in TARGETS) and
               (('_lag_' in c and not c.endswith('_lag_24')) or '_mean_' in c or '_ema_' in c))]
        if self.dynamic_features:
            keep=list(frame.columns)
        result=frame[keep].copy()
        if self.smooth_inputs:
            # Optional low-pass view of non-target sensors; load is untouched.
            for c in self.columns:
                if c not in TARGETS:
                    result[c]=result[c].ewm(span=4,adjust=False).mean()
        return result.astype('float32')

    def transform(self,raw):
        result=self._build(raw)[self.source_feature_columns]
        if self.quantile_features:
            result[self.rank_columns]=self.quantile.transform(result[self.rank_columns]).astype('float32')
            # Do not let downstream models mistake percentile features for MW.
            result=result.rename(columns={c:c+'_quantile' for c in self.rank_columns})
        return result
