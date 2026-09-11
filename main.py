"""One-command entry point for forecasting and gas dispatch."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline import (
    FREQ,
    OPT_COLUMNS,
    STEP_MINUTES,
    TARGETS,
    DirectMultiHorizonForecaster,
    DispatchConfig,
    ForecastConfig,
    GasResourceForecaster,
    IndustrialDataPipeline,
    IndustrialFeatureBuilder,
    MILPDispatcher,
    PriceSchedule,
    validate_submission_frames,
    write_json,
)


LOGGER = logging.getLogger("ai_steel")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal 15-minute gas power forecasting and 24-hour MILP dispatch"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("."), help="Directory with the official input files")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for result CSV files")
    parser.add_argument("--price-file", type=Path, default=None, help="Optional explicit path to price.xlsx")
    parser.add_argument("--origins-file", type=Path, default=None, help="CSV containing forecast origin datetimes")
    parser.add_argument("--forecast-start", type=str, default=None, help="First rolling origin, inclusive")
    parser.add_argument("--forecast-end", type=str, default=None, help="Last rolling origin, inclusive")
    parser.add_argument("--origin-count", type=int, default=96, help="Default number of final rolling origins")
    parser.add_argument("--max-train-rows", type=int, default=50_000)
    parser.add_argument("--max-features", type=int, default=320)
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-ensemble", action="store_true", help="Disable optional CatBoost member")
    parser.add_argument("--max-model-threads", type=int, default=8)
    parser.add_argument("--solver-time-limit", type=int, default=25)
    parser.add_argument('--official', action='store_true', help='Run supplied Pre_/Pre_test_ package with frozen development selection')
    parser.add_argument('--selection', type=Path, default=Path('artifacts/development/selection.json'))
    parser.add_argument("--fast", action="store_true", help="Train anchor horizons and interpolate (for smoke tests)")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args(argv)


def _load_origins_file(path: Path) -> pd.DatetimeIndex:
    table = pd.read_csv(path, encoding="utf-8-sig")
    if table.empty:
        raise ValueError(f"Origins file is empty: {path}")
    column = "datetime" if "datetime" in table else table.columns[0]
    values = pd.to_datetime(table[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No valid datetime values in {path}")
    return pd.DatetimeIndex(values.drop_duplicates().sort_values(), name="datetime")


def select_origins(index: pd.DatetimeIndex, args: argparse.Namespace) -> pd.DatetimeIndex:
    if args.origins_file is not None:
        requested = _load_origins_file(args.origins_file)
        # Official timestamps should already be on the grid; flooring also handles second-level noise.
        requested = requested.floor(FREQ).drop_duplicates().sort_values()
        missing = requested.difference(index)
        if len(missing):
            raise ValueError(f"{len(missing)} requested origins are outside aligned data; first={missing[0]}")
        return requested

    start = pd.Timestamp(args.forecast_start).floor(FREQ) if args.forecast_start else None
    end = pd.Timestamp(args.forecast_end).floor(FREQ) if args.forecast_end else None
    if start is not None or end is not None:
        start = start or index.min()
        end = end or index.max()
        selected = index[(index >= start) & (index <= end)]
        if not len(selected):
            raise ValueError(f"No aligned timestamps in requested range [{start}, {end}]")
        return pd.DatetimeIndex(selected, name="datetime")

    count = max(1, min(int(args.origin_count), len(index)))
    return pd.DatetimeIndex(index[-count:], name="datetime")


def _format_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "datetime" not in result:
        result.insert(0, "datetime", result.index)
    result["datetime"] = pd.to_datetime(result["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    result = result.reset_index(drop=True)
    return result


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", float_format="%.6f")
    os.replace(temporary, path)


def _result_frames(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_result = _format_datetime(predictions)
    short_columns = ["datetime"]
    for target in TARGETS:
        short_columns.extend(
            f"{target}_t+{h * STEP_MINUTES}_pred" for h in range(1, 9)
        )
    return long_result.loc[:, short_columns], long_result


def run(args: argparse.Namespace) -> dict[str, object]:
    packaged_data = list(Path(args.input_dir).rglob('Pre_test_load.csv'))
    if getattr(args, 'official', False) or packaged_data:
        from run_official import run as official_run
        if not hasattr(args, 'selection'):
            args.selection = Path(__file__).parent/'artifacts/development/selection.json'
        if not args.selection.exists():
            raise FileNotFoundError('Missing frozen selection. First run: python experiment.py')
        metadata = official_run(args)
        from validate_outputs import validate
        validate(args.input_dir, args.output_dir)
        return metadata
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    price_file = args.price_file.resolve() if args.price_file else input_dir / "price.xlsx"

    LOGGER.info("Loading and aligning official tables from %s", input_dir)
    raw = IndustrialDataPipeline().load(input_dir)
    LOGGER.info("Aligned %d rows and %d raw numeric columns", len(raw), len(raw.columns))
    prices = PriceSchedule.from_excel(price_file)
    features = IndustrialFeatureBuilder(prices).build(raw)
    origins = select_origins(features.index, args)
    train_until = origins.min()
    LOGGER.info(
        "Forecasting %d rolling origins from %s to %s; training labels end at %s",
        len(origins),
        origins.min(),
        origins.max(),
        train_until,
    )

    forecast_config = ForecastConfig(
        max_horizon=96,
        max_train_rows=max(256, args.max_train_rows),
        max_features=max(32, args.max_features),
        n_estimators=max(20, args.n_estimators),
        random_state=args.seed,
        fast=args.fast,
        ensemble=not bool(getattr(args, "no_ensemble", False)),
        max_model_threads=max(1, int(getattr(args, "max_model_threads", 8))),
    )
    forecaster = DirectMultiHorizonForecaster(forecast_config).fit(features, train_until)
    predictions = forecaster.predict(features, origins)
    short_result, long_result = _result_frames(predictions)

    input_result = _format_datetime(features.loc[origins])
    model_dir = output_dir / "models"
    forecaster.save(model_dir)

    dispatch_origin = origins.max()
    dispatch_times = pd.date_range(
        dispatch_origin + pd.Timedelta(minutes=STEP_MINUTES),
        periods=96,
        freq=FREQ,
        name="datetime",
    )
    resource_model = GasResourceForecaster(raw)
    from dispatch import observed_surplus
    net_inflow, gas_caps, initial_holder, capacity, _ = observed_surplus(raw, dispatch_origin, 96)
    efficiency = resource_model.estimate_efficiency(dispatch_origin)
    dispatch_config = DispatchConfig(holder_capacity=capacity, solver_time_limit=max(1, args.solver_time_limit))
    dispatch = MILPDispatcher(dispatch_config).optimize(
        dispatch_times,
        prices.prices(dispatch_times).to_numpy(dtype=float),
        net_inflow,
        gas_caps,
        efficiency,
        initial_holder,
    )
    opt_result = _format_datetime(dispatch.gas_plan)
    opt_result = opt_result[["datetime", *(OPT_COLUMNS[g] for g in ("blast_furnace", "coke", "converter"))]]

    validate_submission_frames(input_result, short_result, long_result, opt_result)
    outputs = {
        "input.csv": input_result,
        "s_result.csv": short_result,
        "l_result.csv": long_result,
        "opt_result.csv": opt_result,
    }
    for filename, frame in outputs.items():
        _atomic_csv(frame, output_dir / filename)
        LOGGER.info("Wrote %s: %d rows x %d columns", filename, len(frame), len(frame.columns))

    metadata: dict[str, object] = {
        "frequency": FREQ,
        "raw_rows": len(raw),
        "raw_columns": list(raw.columns),
        "feature_count": len(features.columns) - len(raw.columns),
        "forecast_origins": len(origins),
        "forecast_origin_start": str(origins.min()),
        "forecast_origin_end": str(origins.max()),
        "train_label_cutoff": str(train_until),
        "model_kind": forecaster.model_kind,
        "trained_horizons": forecaster.trained_horizons,
        "dispatch_status": dispatch.status,
        "dispatch_diagnostics": dispatch.diagnostics,
        "gas_efficiency_mw_per_flow_unit": efficiency,
        "initial_holder": initial_holder,
        "seed": args.seed,
    }
    write_json(output_dir / "run_metadata.json", metadata)
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        metadata = run(args)
    except Exception:
        LOGGER.exception("Pipeline failed")
        return 1
    LOGGER.info("Completed successfully; dispatch=%s", metadata["dispatch_status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
