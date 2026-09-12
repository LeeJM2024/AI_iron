"""Regression tests for artifact compatibility and the input-quality repair.

Two failure modes this locks down, both of which actually happened in this repo:

1. ``forecast_model.joblib`` became unreadable ("'LGBMRegressor' object is not
   iterable") the moment the trainer started storing seed-bag lists, so
   ``validate_outputs.py`` could not reload the frozen model.  Inference must
   accept BOTH the legacy single-estimator layout and the list layout.

2. ``_quality_input_frame`` fitted its Tukey fence on a 120-day window that
   straddles the 2025-04-18 regime break, so it declared ~300 legitimate
   post-regime values anomalous and rewrote them -- and for a constant field the
   inward float32 nudge inverted the bounds entirely.

3. The delivered ``input.csv`` is float32 written with ``"%.6f"``, so a value
   clipped exactly onto a fence bound can read back half a CSV quantum on the
   wrong side of it.  A fence check needs exactly that much slack -- and must
   convert the bound to a Python float first, because ``np.float32 - tol`` is
   evaluated in float32 under NEP-50 and the tolerance vanishes.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecasting import (  # noqa: E402
    ModelConfig,
    ResidualEnsemble,
    estimator_slots,
    features,
)
from run_official import _quality_input_frame, _regime_window, _tukey_fence  # noqa: E402

# Half of the last "%.6f" digit, plus a hair for float representation error.
CSV_ROUNDTRIP_TOL = 5e-7 + 1e-9


def _fence_bounds(history, regime, column):
    """Mirror run_official._quality_input_frame's bound computation."""
    obs = history[column].replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = _tukey_fence(obs)
    if regime is not None:
        post = regime[column].replace([np.inf, -np.inf], np.nan).dropna()
        if not post.empty:
            lo2, hi2 = _tukey_fence(post)
            lo, hi = min(lo, lo2), max(hi, hi2)
    lo32 = float(np.nextafter(np.float32(lo), np.float32(np.inf)))
    hi32 = float(np.nextafter(np.float32(hi), np.float32(-np.inf)))
    if not hi32 > lo32:
        lo32 = hi32 = float(np.float32(hi))
    return lo32, hi32


def _synthetic_raw(days: int = 20, low: float = 100.0, high: float = 300.0,
                   switch_day: int = 16, seed: int = 3) -> pd.DataFrame:
    """Targets step from `low` to `high` on `switch_day`; aux field follows."""
    index = pd.date_range("2025-01-01", periods=days * 96, freq="15min")
    n = len(index)
    level = np.where(np.arange(n) // 96 >= switch_day, high, low)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "generator_1": level + rng.normal(0, 2.0, n),
            "generator_all": level * 2.0 + rng.normal(0, 3.0, n),
            "generator_use_blast_furnace_gas": level * 1000.0 + rng.normal(0, 100.0, n),
            "converter_user1": np.zeros(n),
        },
        index=index,
    )


class EstimatorSlotTests(unittest.TestCase):
    def test_legacy_single_estimator_is_wrapped(self):
        sentinel = object()
        self.assertEqual(estimator_slots(sentinel), (sentinel,))

    def test_list_layout_filters_none(self):
        a, b = object(), object()
        self.assertEqual(estimator_slots([a, None, b]), (a, b))
        self.assertEqual(estimator_slots(None), ())
        self.assertEqual(estimator_slots([]), ())


class LegacyModelLayoutTests(unittest.TestCase):
    """A model saved by the single-estimator trainer must still predict."""

    def test_predictions_match_after_downgrading_to_legacy_layout(self):
        raw = _synthetic_raw()
        x = features(raw)
        cutoff = raw.index[-400]
        cfg = ModelConfig(trees=20, leaves=7, threads=1, train_days=20,
                          use_shared_horizon=False)
        model = ResidualEnsemble(cfg).fit(raw, x, cutoff, [1, 2])
        origins = raw.index[-200:]
        target = "generator_1"

        bagged = model.predict_members(raw, x, origins, 1, target)
        self.assertEqual(bagged.shape, (len(origins), 9))

        # Downgrade every slot to the legacy single-estimator layout.
        for key, (lgb, cat) in list(model.models.items()):
            model.models[key] = (estimator_slots(lgb)[0], estimator_slots(cat)[0])
        legacy = model.predict_members(raw, x, origins, 1, target)

        self.assertEqual(legacy.shape, bagged.shape)
        self.assertTrue(np.isfinite(legacy).all())
        np.testing.assert_allclose(legacy, bagged, rtol=0, atol=0)


class QualityFrameTests(unittest.TestCase):
    def setUp(self):
        self.raw = _synthetic_raw()
        self.x = features(self.raw)
        self.cutoff = self.raw.index[-1]
        self.delivered = _quality_input_frame(self.x, self.cutoff, self.raw).loc[
            self.raw.index[-384:]
        ]
        self.raw_cols = [c for c in self.delivered.columns if not c.startswith("feat_")]

    def test_every_raw_field_gets_an_outlier_flag(self):
        for c in self.raw_cols:
            self.assertIn(f"feat_{c}_outlier", self.delivered.columns, c)
        self.assertTrue(np.isfinite(self.delivered.to_numpy(dtype=float)).all())

    def test_delivered_values_never_sit_outside_the_fence_used(self):
        regime = _regime_window(self.raw)
        self.assertIsNotNone(regime, "synthetic data must expose a regime window")
        for c in self.raw_cols:
            obs = self.raw[c].replace([np.inf, -np.inf], np.nan).dropna()
            lo, hi = _tukey_fence(obs)
            post = regime[c].replace([np.inf, -np.inf], np.nan).dropna()
            if not post.empty:
                lo2, hi2 = _tukey_fence(post)
                lo, hi = min(lo, lo2), max(hi, hi2)
            lo32 = np.nextafter(np.float32(lo), np.float32(np.inf))
            hi32 = np.nextafter(np.float32(hi), np.float32(-np.inf))
            if not hi32 > lo32:
                lo32 = hi32 = np.float32(hi)
            v = self.delivered[c].to_numpy(dtype=float)
            self.assertEqual(int(((v < lo32) | (v > hi32)).sum()), 0, c)

    def test_post_regime_normal_values_are_not_repaired(self):
        """The whole point: a level that is normal AFTER the break must survive."""
        field = "generator_use_blast_furnace_gas"
        post_level = _regime_window(self.raw)[field].median()
        self.assertAlmostEqual(post_level, 300_000.0, delta=5_000.0)
        # The full-history fence would reject it, which is the bug being fixed.
        lo_full, hi_full = _tukey_fence(self.raw[field])
        self.assertGreater(post_level, hi_full)

        tail = self.delivered[field].iloc[-96:]           # last day == post-regime
        self.assertTrue((tail > hi_full).all())
        flags = self.delivered[f"feat_{field}_outlier"].iloc[-96:]
        self.assertEqual(int(np.nansum(flags.to_numpy())), 0)

    def test_genuine_spike_is_flagged_and_clipped(self):
        field = "generator_use_blast_furnace_gas"
        raw = self.raw.copy()
        spike_index = raw.index[-10]
        raw.loc[spike_index, field] = 5_000_000.0
        x = features(raw)
        delivered = _quality_input_frame(x, raw.index[-1], raw).loc[[spike_index]]
        self.assertEqual(float(delivered[f"feat_{field}_outlier"].iloc[0]), 1.0)
        self.assertLess(float(delivered[field].iloc[0]), 5_000_000.0)

    def test_constant_field_keeps_its_value(self):
        """A constant-0 field must not collapse to a negative denormal."""
        vals = self.delivered["converter_user1"].to_numpy(dtype=float)
        self.assertEqual(np.unique(vals).tolist(), [0.0])
        self.assertFalse((vals < 0).any())


class SerializationBoundaryTests(unittest.TestCase):
    """The shipped CSV must survive its own float32/"%.6f" round trip.

    A value clipped exactly onto a fence bound reads back up to half a CSV
    quantum (5e-7) outside it.  Two traps are pinned here, both of which really
    fired in ``check_submission_package.py``: the missing tolerance, and the
    NEP-50 float32 promotion that silently eats the tolerance.
    """

    def test_npfloat32_subtraction_swallows_the_tolerance(self):
        # 47251.9140625 is float32-exact; the float32 spacing is 1/256 ~ 3.9e-3.
        bound = np.float32(47251.9140625)
        self.assertEqual(float(bound - 5e-7), float(bound))   # tolerance lost
        self.assertLess(float(bound) - 5e-7, float(bound))    # survives in float

    def test_constant_field_csv_round_trip_needs_the_quantum_tolerance(self):
        # A float32-exact constant whose "%.6f" form rounds to ...914062.
        value = 47251.9140625
        raw = _synthetic_raw()
        raw["converter_user1"] = value
        x = features(raw)
        delivered = _quality_input_frame(x, raw.index[-1], raw)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.csv"
            frame = delivered.copy()
            frame.index.name = "datetime"
            frame.to_csv(path, encoding="utf-8-sig", float_format="%.6f")
            back = pd.read_csv(path, encoding="utf-8-sig").set_index("datetime")
        back.index = pd.to_datetime(back.index)

        lo32, hi32 = _fence_bounds(raw, _regime_window(raw), "converter_user1")
        self.assertEqual(lo32, hi32, "a constant field must collapse to one bound")
        v = back["converter_user1"].to_numpy(dtype=float)
        self.assertNotEqual(float(v[0]), lo32, "the file must be quantized to 6 decimals")

        naive = int(((v < lo32) | (v > hi32)).sum())
        self.assertEqual(naive, len(v), "the un-toleranced check must flag every row")

        tolerated = int(((v < lo32 - CSV_ROUNDTRIP_TOL) |
                         (v > hi32 + CSV_ROUNDTRIP_TOL)).sum())
        self.assertEqual(tolerated, 0, "one CSV quantum of slack must clear it")


if __name__ == "__main__":
    unittest.main()
