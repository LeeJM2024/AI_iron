"""End-to-end smoke test with dirty, multi-table industrial-style data."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Make direct invocation (`python tests/smoke_test.py`) behave like the
# documented module invocation from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import run
from pipeline import DispatchConfig, GAS_TYPES, MILPDispatcher, OPT_COLUMNS


def _dirty_csv(frame: pd.DataFrame, path: Path) -> None:
    dirty = frame.copy()
    numeric = [c for c in dirty if c != "datetime"]
    dirty.loc[17, numeric[0]] = np.nan
    dirty.loc[53, numeric[-1]] = float(dirty[numeric[-1]].median()) * 20.0
    dirty = pd.concat([dirty, dirty.iloc[[80]]], ignore_index=True)
    dirty = dirty.drop(index=[110]).sample(frac=1.0, random_state=7)
    dirty.to_csv(path, index=False, encoding="utf-8")


def make_inputs(directory: Path, days: int = 24) -> None:
    index = pd.date_range("2025-01-01", periods=days * 96, freq="15min")
    n = len(index)
    slot = np.arange(n)
    daily = np.sin(2.0 * np.pi * slot / 96.0)
    weekly = np.sin(2.0 * np.pi * slot / (96.0 * 7.0))
    rng = np.random.default_rng(2026)

    blast_supply = 1_050_000 + 75_000 * daily + 25_000 * weekly + rng.normal(0, 8_000, n)
    coke_supply = 90_000 + 7_000 * daily + rng.normal(0, 1_000, n)
    converter_supply = 145_000 + 22_000 * np.roll(daily, 12) + rng.normal(0, 2_000, n)
    gas = pd.DataFrame(
        {
            "datetime": index.strftime("%Y-%m-%d %H:%M:%S"),
            "blast_furnace_1": blast_supply,
            "air_heater_1": 330_000 + 20_000 * daily,
            "coke_oven_1": coke_supply,
            "converter_1": converter_supply,
            "into_gas_mixed_blast_furnace": 45_000 + 3_000 * daily,
            "into_gas_mixed_coke": 15_000 + 1_000 * daily,
            "into_gas_mixed_converter": 20_000 + 2_000 * daily,
        }
    )
    _dirty_csv(gas, directory / "gas.csv")

    holder = pd.DataFrame(
        {
            "datetime": index.strftime("%Y-%m-%d %H:%M:%S"),
            "blast_furnace_gas_holder_1": 70_000 + 12_000 * np.sin(2 * np.pi * slot / 192),
            "coke_gas_holder_1": 20_000 + 3_000 * np.sin(2 * np.pi * slot / 96),
            "converter_gas_holder_1": 22_000 + 4_000 * np.sin(2 * np.pi * slot / 144),
        }
    )
    _dirty_csv(holder, directory / "gas_holder.csv")

    users = pd.DataFrame(
        {
            "datetime": index.strftime("%Y-%m-%d %H:%M:%S"),
            "blast_furnace_user1": 320_000 + 15_000 * weekly,
            "coke_user1": 22_000 + 1_500 * daily,
            "converter_user1": 35_000 + 4_000 * daily,
            "mixed_gas_user1": 70_000 + 5_000 * daily,
        }
    )
    _dirty_csv(users, directory / "gas_user.csv")

    use_blast = 570_000 + 40_000 * daily + 18_000 * weekly
    use_coke = 35_000 + 3_000 * np.roll(daily, 4)
    use_converter = 55_000 + 5_000 * np.roll(daily, 8)
    total_load = (
        use_blast * (3.2 * 0.32 / 3600.0)
        + use_coke * (17.0 * 0.32 / 3600.0)
        + use_converter * (7.5 * 0.32 / 3600.0)
    )
    generator_1 = np.clip(total_load * 0.46 + 5.0 * daily, 0.0, 200.0)
    load = pd.DataFrame(
        {
            "datetime": index.strftime("%Y-%m-%d %H:%M:%S"),
            "generator_1": generator_1,
            "generator_all": np.clip(total_load, 0.0, 440.0),
            "generator_use_coke_gas": use_coke,
            "generator_use_converter_gas": use_converter,
            "generator_use_blast_furnace_gas": use_blast,
        }
    )
    _dirty_csv(load, directory / "load.csv")

    prices = np.where(
        ((index[:96].hour >= 8) & (index[:96].hour < 11))
        | ((index[:96].hour >= 18) & (index[:96].hour < 21)),
        1.25,
        np.where((index[:96].hour < 7) | (index[:96].hour >= 23), 0.35, 0.78),
    )
    tariff = pd.DataFrame({"time": index[:96].strftime("%H:%M")})
    for month in range(1, 13):
        tariff[f"{month}月"] = prices * (1.0 + 0.005 * month)
    tariff.to_excel(directory / "price.xlsx", index=False)


def test_end_to_end() -> None:
    with tempfile.TemporaryDirectory(prefix="ai_steel_smoke_") as temp:
        root = Path(temp)
        data = root / "data"
        output = root / "output"
        data.mkdir()
        make_inputs(data)
        args = argparse.Namespace(
            input_dir=data,
            output_dir=output,
            price_file=None,
            origins_file=None,
            forecast_start=None,
            forecast_end=None,
            origin_count=8,
            max_train_rows=2_000,
            max_features=96,
            n_estimators=30,
            seed=2026,
            solver_time_limit=10,
            fast=True,
            log_level="INFO",
        )
        metadata = run(args)
        assert metadata["forecast_origins"] == 8
        short = pd.read_csv(output / "s_result.csv")
        long = pd.read_csv(output / "l_result.csv")
        opt = pd.read_csv(output / "opt_result.csv")
        inputs = pd.read_csv(output / "input.csv")
        assert short.shape == (8, 17)
        assert long.shape == (8, 193)
        assert opt.shape == (96, 4)
        assert inputs.filter(regex=r"^feat_").shape[1] > 20
        assert np.isfinite(long.drop(columns="datetime").to_numpy()).all()
        assert (opt.drop(columns="datetime") >= 0.0).all().all()


def test_milp_constraints() -> None:
    timestamps = pd.date_range("2025-01-02", periods=32, freq="15min")
    prices = np.where((timestamps.hour >= 8) & (timestamps.hour < 20), 1.2, 0.4)
    net = {gas: np.full(32, value) for gas, value in zip(GAS_TYPES, (550_000.0, 30_000.0, 45_000.0))}
    caps = {gas: values * 1.5 for gas, values in net.items()}
    efficiency = {
        "blast_furnace": 3.2 * 0.32 / 3600.0,
        "coke": 17.0 * 0.32 / 3600.0,
        "converter": 7.5 * 0.32 / 3600.0,
    }
    cfg = DispatchConfig(solver_time_limit=10)
    result = MILPDispatcher(cfg).optimize(timestamps, prices, net, caps, efficiency, 100_000.0)
    assert result.status.startswith('MILP_HiGHS')
    assert result.holder_path.min() >= cfg.holder_min_fraction * cfg.holder_capacity - 1e-5
    assert result.holder_path.max() <= cfg.holder_max_fraction * cfg.holder_capacity + 1e-5
    assert result.power_path.min() >= -1e-5
    assert result.power_path.max() <= sum(cfg.unit_ratings_mw) + 1e-5
    assert list(result.gas_plan) == [OPT_COLUMNS[g] for g in GAS_TYPES]


if __name__ == "__main__":
    test_end_to_end()
    test_milp_constraints()
    print("smoke test passed")
