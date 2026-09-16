"""Re-create the compact+trajectory transform (V16 schema) used by the rejected V16 family.

Kept minimal: the transform restores the already-engineered causal trajectory columns of the
four gas-side sensors that the compact schema drops. Used here only to generate additional
members for the prediction bank; the causal deployment route stays compact.
"""
from __future__ import annotations

from compact_inputs import CompactInputTransform
from prelim import TARGETS
from stable_inputs import StableInputTransform

TRAJECTORY_SENSORS = ('generator_use_blast_furnace_gas', 'generator_use_coke_gas',
                      'generator_use_converter_gas', 'blast_furnace_gas_holder_2')


class CompactPlusInputTransform(CompactInputTransform):
    def __init__(self, active_days=7, robust_raw=True, smooth_inputs=False,
                 dynamic_features=False, quantile_features=False,
                 trajectory_sensors=TRAJECTORY_SENSORS,
                 trajectory_kinds=('mean_4', 'mean_16', 'change_4')):
        super().__init__(active_days, robust_raw, smooth_inputs, dynamic_features, quantile_features)
        self.trajectory_sensors = tuple(trajectory_sensors)
        self.trajectory_kinds = tuple(trajectory_kinds)

    def trajectory_columns(self, frame):
        names = []
        for sensor in self.trajectory_sensors:
            names += [f'feat_{sensor}_{kind}' for kind in self.trajectory_kinds]
        return [c for c in names if c in frame]

    def _build(self, raw):
        full = StableInputTransform._build(self, raw)
        keep = [c for c in full if c in self.columns or c == 'feat_other_power' or
                c.startswith('feat_clock_') or
                (any(c.startswith(f'feat_{t}_') for t in TARGETS) and
                 (('_lag_' in c and not c.endswith('_lag_24')) or '_mean_' in c or '_ema_' in c)) or
                c in self.trajectory_columns(full)]
        result = full[keep].copy()
        if self.smooth_inputs:
            for c in self.columns:
                if c not in TARGETS:
                    result[c] = result[c].ewm(span=4, adjust=False).mean()
        return result.astype('float32')
