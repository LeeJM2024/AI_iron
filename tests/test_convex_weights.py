"""Correctness of the MAPE-optimal convex weight LP.

``_convex_mape_weights`` was rewritten to feed HiGHS a SPARSE constraint matrix.
At n=6000 the dense form is 2*6000*6009*8 bytes (~1.15 GB) per solve, and one
12-fold leave-one-day-out pass solves the LP several hundred times -- enough to
exhaust memory.  Sparsity is meant to be a pure memory fix, so these tests pin
the thing that actually matters: the optimum must be unchanged, i.e. still the
exact minimiser of the simplex-constrained MAPE objective.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecasting import _convex_mape_weights  # noqa: E402
import forecasting  # noqa: E402


def _dense_reference(y, p, cap=6000, seed=2026):
    """The pre-sparse implementation, kept here as the oracle."""
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
    cost = np.concatenate([np.zeros(k), np.ones(n) / n])
    A_ub = np.vstack([np.hstack([-A, -np.eye(n)]), np.hstack([A, -np.eye(n)])])
    b_ub = np.concatenate([-rhs, rhs])
    A_eq = np.hstack([np.ones((1, k)), np.zeros((1, n))])
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * k + [(0.0, None)] * n
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.success:
        w = np.clip(res.x[:k], 0.0, None)
        return w / w.sum() if w.sum() > 0 else np.ones(k) / k
    return None


def _mape(y, pred):
    return float(np.mean(np.abs(y - pred) / np.abs(y)))


def _sample(n, k, seed, spread=3.0):
    rng = np.random.default_rng(seed)
    y = 100.0 + rng.normal(0, 5, n)
    p = np.column_stack([y + rng.normal(0, spread, n) for _ in range(k)])
    return y, p


class ConvexWeightTests(unittest.TestCase):
    def test_weights_are_a_simplex_vector(self):
        y, p = _sample(300, 9, seed=1)
        w = _convex_mape_weights(y, p)
        self.assertEqual(w.shape, (9,))
        self.assertTrue(np.all(w >= -1e-12))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)

    def test_matches_the_dense_reference_optimum(self):
        for n in (120, 500, 1500):
            with self.subTest(n=n):
                y, p = _sample(n, 9, seed=n)
                sparse_w = _convex_mape_weights(y, p)
                dense_w = _dense_reference(y, p)
                np.testing.assert_allclose(sparse_w, dense_w, rtol=0, atol=1e-9)

    def test_optimum_is_no_worse_than_any_single_member_or_uniform(self):
        """The LP must actually minimise: nothing on the simplex can beat it."""
        y, p = _sample(400, 9, seed=5)
        w = _convex_mape_weights(y, p)
        best = _mape(y, p @ w)
        for j in range(p.shape[1]):
            e = np.zeros(p.shape[1]); e[j] = 1.0
            self.assertLessEqual(best, _mape(y, p @ e) + 1e-9)
        uniform = np.ones(p.shape[1]) / p.shape[1]
        self.assertLessEqual(best, _mape(y, p @ uniform) + 1e-9)
        # random simplex points, as a sanity net
        rng = np.random.default_rng(11)
        for _ in range(50):
            v = rng.random(p.shape[1]); v /= v.sum()
            self.assertLessEqual(best, _mape(y, p @ v) + 1e-9)

    def test_degenerate_inputs_do_not_raise(self):
        # all-constant target: scale blows up, must fall back to a simplex vector
        y = np.full(50, 100.0)
        p = np.column_stack([y.copy() for _ in range(9)])
        w = _convex_mape_weights(y, p)
        self.assertEqual(w.shape, (9,))
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)
        # empty / single member
        self.assertEqual(_convex_mape_weights(np.array([]), np.empty((0, 3))).shape, (3,))
        w1 = _convex_mape_weights(np.array([1.0, 2.0]), np.array([[1.0], [2.0]]))
        self.assertAlmostEqual(float(w1[0]), 1.0, places=9)

    def test_large_problem_stays_cheap(self):
        """A 6000-sample solve must not need a gigabyte-scale dense matrix."""
        import time
        y, p = _sample(6000, 9, seed=99)
        t0 = time.perf_counter()
        w = _convex_mape_weights(y, p)
        elapsed = time.perf_counter() - t0
        self.assertEqual(w.shape, (9,))
        self.assertLess(elapsed, 30.0, f"n=6000 solve took {elapsed:.1f}s")


class ShrinkageSearchCostTests(unittest.TestCase):
    """The leave-one-cutoff-out shrinkage search must fit each LP once per
    cutoff, never once per (cutoff, shrinkage) pair.

    The default shrinkage grid holds five factors, and the per-cutoff fits do not
    depend on the factor at all, so the original nesting simply refitted
    everything five times.  That search is the hotspot of a leave-one-day-out
    pass that solves several hundred LPs, so the waste is not cosmetic.  This
    test pins the count against the grid size rather than against a wall-clock.
    """

    def _records(self):
        rng = np.random.default_rng(3)
        records = []
        for cutoff in ("d1", "d2"):
            for h in (1, 2, 4, 8):
                y = np.abs(rng.normal(100, 20, 120)) + 5
                p = np.column_stack([y + rng.normal(0, s, 120) for s in range(8)])
                records.append(dict(target="generator_1", h=h, y=y, p=p, cutoff=cutoff))
        return records

    def test_fit_count_does_not_grow_with_the_shrinkage_grid(self):
        records = self._records()
        counts = {}
        for grid in ((0.0,), (0.0, 0.25, 0.5, 0.75, 1.0)):
            seen = {"calls": 0}
            orig = forecasting._convex_mape_weights

            def wrapped(y, p, cap=6000, seed=2026, _seen=seen, _orig=orig):
                _seen["calls"] += 1
                return _orig(y, p, cap, seed)

            forecasting._convex_mape_weights = wrapped
            try:
                forecasting.select_weights(records, shrinkage_grid=grid)
            finally:
                forecasting._convex_mape_weights = orig
            counts[grid] = seen["calls"]

        self.assertEqual(counts[(0.0,)], counts[(0.0, 0.25, 0.5, 0.75, 1.0)],
                         f"fit count grew with the shrinkage grid: {counts}")


if __name__ == "__main__":
    unittest.main()
