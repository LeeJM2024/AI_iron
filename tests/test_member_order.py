"""Assert that ResidualEnsemble column order matches MEMBERS.

A silent mismatch applies ensemble weights to the wrong members -- it produces
plausible-looking numbers that are entirely wrong. This guard must never fail.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from forecasting import MEMBERS, ResidualEnsemble, ModelConfig  # noqa: E402


def main():
    idx = pd.to_datetime(["2025-04-29 00:00", "2025-04-29 00:15", "2025-04-29 00:30"])
    raw = pd.DataFrame({t: np.linspace(100.0, 140.0, len(idx)) for t in ("generator_1", "generator_all")},
                       index=idx)
    # stretch the frame so the minimum-row guards are not hit in profile building
    long_idx = pd.date_range("2025-04-20", periods=1200, freq="15min")
    raw = pd.DataFrame({t: 100.0 + 20.0 * np.sin(np.arange(len(long_idx)) / 40.0)
                        for t in ("generator_1", "generator_all")}, index=long_idx)

    model = ResidualEnsemble(ModelConfig())
    model.cutoff = pd.Timestamp("2025-04-25 00:00")
    model.momentum = {"generator_1": (0.0, 0.5), "generator_all": (0.4, 0.6)}
    model._build_profiles(raw)
    model.models = {("generator_1", 1): (None, None), ("generator_all", 1): (None, None)}

    origins = pd.DatetimeIndex([pd.Timestamp("2025-04-25 00:00"), pd.Timestamp("2025-04-25 01:00")])
    p = model.predict_members(raw, raw, origins, 1, "generator_1")
    assert p.shape[1] == len(MEMBERS), f"columns={p.shape[1]} != MEMBERS={len(MEMBERS)}"

    now = raw.loc[origins, "generator_1"].to_numpy()
    ema = raw["generator_1"].ewm(span=4, adjust=False).mean().loc[origins].to_numpy()
    # phi = 0 collapses the momentum member onto persistence.
    assert np.allclose(p[:, MEMBERS.index("persistence")], now), "persistence column mismatch"
    assert np.allclose(p[:, MEMBERS.index("ema")], ema), "ema column mismatch"
    assert np.allclose(p[:, MEMBERS.index("momentum")], now), (
        "momentum column mismatch (expected persistence because phi=0)")
    print("column order OK ->", list(zip(range(len(MEMBERS)), MEMBERS)))
    print("PASS: predict_members column order matches MEMBERS")


if __name__ == "__main__":
    main()
