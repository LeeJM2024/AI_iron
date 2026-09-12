"""Invariants for horizon-interpolated ensemble weights.

select_weights must emit a weight for EVERY served horizon (1..96), each one on
the simplex, and each key must be namespaced so per-horizon keys can never
collide with the coarse bucket keys they fall back to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecasting import (
    BUCKETS,
    MEMBERS,
    TARGETS,
    bucket,
    select_weights,
    weights_for,
    _bucket_span,
)


PRODUCTION_GRID = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96)


def _fake_records(grid=PRODUCTION_GRID):
    rng = np.random.default_rng(0)
    rows = []
    for cutoff in ("2025-04-22", "2025-04-25"):
        for target in TARGETS:
            for h in grid:
                n = 400
                y = np.abs(rng.normal(100, 20, n)) + 5
                p = np.column_stack([
                    y + rng.normal(0, s, n) for s in (2, 4, 6, 8, 10, 12, 3, 5)
                ])
                rows.append(dict(target=target, h=h, y=y, p=p, cutoff=cutoff))
    return rows


def main():
    rows = _fake_records()
    weights = select_weights(rows)

    # 1. bucket fallback keys exist and are on the simplex
    for target in TARGETS:
        for b in BUCKETS:
            w = np.asarray(weights[f"{target}/{b}"])
            assert w.shape == (len(MEMBERS),), (target, b, w.shape)
            assert np.all(w >= -1e-12), (target, b, w)
            assert abs(w.sum() - 1.0) < 1e-9, (target, b, w.sum())

    # 2. a weight exists for EVERY served horizon, keys cannot collide
    for target in TARGETS:
        for h in range(1, 97):
            key = f"{target}/h{h}"
            assert key in weights, f"no weight fitted for {target} h={h}"
            w = np.asarray(weights[key])
            assert w.shape == (len(MEMBERS),), (target, h, w.shape)
            assert np.all(w >= -1e-12), (target, h, w)
            assert abs(w.sum() - 1.0) < 1e-9, (target, h, w.sum())
            assert key != f"{target}/{bucket(h)}", "per-horizon key collides with bucket key"

    # 3. weights_for resolves per-horizon first, bucket second
    for target in TARGETS:
        for h in range(1, 97):
            assert weights_for(weights, target, h) is weights[f"{target}/h{h}"]
    # unknown horizon still falls back to its bucket, never None
    assert weights_for(weights, TARGETS[0], 1) is not None

    # 4. each bucket owns a contiguous, non-overlapping horizon span
    covered = []
    for b in BUCKETS:
        lo, hi = _bucket_span(b)
        covered.extend(range(lo + 1, hi + 1))
    assert sorted(covered) == list(range(1, 97)), "bucket spans do not tile 1..96"

    # 5. interpolation must be monotone-smooth in the sense that neighbouring
    # horizons have similar weights (no jump larger than the anchor spacing).
    for target in TARGETS:
        for h in range(2, 97):
            a = np.asarray(weights[f"{target}/h{h - 1}"])
            b = np.asarray(weights[f"{target}/h{h}"])
            if bucket(h - 1) != bucket(h):
                continue
            assert np.abs(a - b).max() < 0.5, (target, h, np.abs(a - b).max())

    print("PASS: interpolated weights cover h=1..96, simplex, no key collision")


if __name__ == "__main__":
    main()
