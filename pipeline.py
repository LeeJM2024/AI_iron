"""Industrial gas-to-power forecasting and dispatch pipeline.

The module intentionally keeps all transformations causal: an input row at time t
is built only from observations at or before t. Direct forecasting labels are also
cut so that every training target is available at the first forecast origin.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
FREQ = "15min"
STEP_MINUTES = 15
TARGETS = ("generator_1", "generator_all")
GAS_TYPES = ("blast_furnace", "coke", "converter")
OPT_COLUMNS = {
    "blast_furnace": "opt_generator_use_blast_furnace_gas",
    "coke": "opt_generator_use_coke_gas",
    "converter": "opt_generator_use_converter_gas",
}
CAPACITY_LIMITS = {"generator_1": (0.0, 200.0), "generator_all": (0.0, 440.0)}


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _gas_type(column: str) -> str | None:
    name = column.lower()
    if name.startswith('air_heater_'):
        return 'blast_furnace'
    if "blast" in name or re.search(r"(^|_)bf(g|_)", name):
        return "blast_furnace"
    if "coke" in name or re.search(r"(^|_)co(g|_)", name):
        return "coke"
    if "converter" in name or re.search(r"(^|_)ldg($|_)", name):
        return "converter"
    return None


def _datetime_column(columns: Sequence[Any]) -> Any:
    preferred = {
        "datetime",
        "timestamp",
        "time",
        "date",
        "datatime",
        "collecttime",
        "recordtime",
    }
    for column in columns:
        if _normalise_name(column) in preferred:
            return column
    return columns[0]


def _coerce_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().mean() >= 0.5:
        return parsed
    numeric = pd.to_numeric(values, errors="coerce")
    for unit in ("s", "ms", "us", "ns"):
        trial = pd.to_datetime(numeric, unit=unit, errors="coerce")
        valid = trial.between("2000-01-01", "2100-01-01").mean()
        if valid >= 0.5:
            return trial
    return parsed


def _read_csv_robust(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Unable to read {path}: {'; '.join(errors)}")


def _numeric_frame(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    if raw.empty:
        raise ValueError(f"{source} is empty")
    time_col = _datetime_column(list(raw.columns))
    dt = _coerce_datetime(raw[time_col])
    valid = dt.notna()
    if not valid.any():
        raise ValueError(f"No parseable datetime values in {source!r}")

    converted: dict[str, np.ndarray] = {}
    for original in raw.columns:
        if original == time_col:
            continue
        name = str(original).strip()
        if not name:
            continue
        values = pd.to_numeric(raw.loc[valid, original], errors="coerce")
        if values.notna().any() or raw[original].isna().all():
            # Use positional arrays: Series labels still refer to the CSV row numbers,
            # while the new frame is indexed by parsed timestamps.
            converted[name] = values.to_numpy(dtype=float)
        else:
            LOGGER.warning("Dropping non-numeric column %s from %s", name, source)

    frame = pd.DataFrame(converted, index=pd.DatetimeIndex(dt.loc[valid].values, name="datetime"))
    frame = frame.loc[~frame.index.isna()].sort_index()
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0, sort=True).mean()
    return frame


def _causal_clean_series(series: pd.Series, window: int = 96) -> pd.Series:
    """Hampel-like cleaning using past observations only."""

    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    past = values.shift(1)
    minimum = min(8, max(2, window // 8))
    median = past.rolling(window, min_periods=minimum).median()
    deviation = (past - median).abs()
    mad = deviation.rolling(window, min_periods=minimum).median()
    scale = 1.4826 * mad
    anomalous = scale.gt(0) & (values - median).abs().gt(8.0 * scale)
    values = values.mask(anomalous, median)
    return values.ffill().fillna(0.0).astype(float)


class IndustrialDataPipeline:
    """Load, resample, align, and causally clean the four official CSV files."""

    source_files = ("gas.csv", "gas_holder.csv", "gas_user.csv", "load.csv")

    def __init__(self, frequency: str = FREQ, clip_nonnegative: bool = True) -> None:
        self.frequency = frequency
        self.clip_nonnegative = clip_nonnegative
        self.source_columns: dict[str, list[str]] = {}
        self.observations: pd.DataFrame | None = None

    def load(self, input_dir: str | Path) -> pd.DataFrame:
        input_path = Path(input_dir)
        frames: list[pd.DataFrame] = []
        missing: list[str] = []
        for filename in self.source_files:
            candidates = [input_path / (prefix + filename) for prefix in ('', 'Pre_', 'Pre_test_')]
            existing = [p for p in candidates if p.exists()]
            if len(existing) > 1:
                raise ValueError(f'Ambiguous files for {filename}: {existing}')
            path = existing[0] if existing else candidates[0]
            if not path.exists():
                missing.append(filename)
                continue
            raw = _read_csv_robust(path)
            frame = _numeric_frame(raw, filename)
            # A row labelled t must never contain readings from (t,t+15min).
            sampler = frame.resample(self.frequency, label='right', closed='right')
            frame = sampler.last() if filename == 'gas_holder.csv' else sampler.mean()
            self.source_columns[filename] = list(frame.columns)
            frames.append(frame)

        if "load.csv" in missing:
            raise FileNotFoundError(f"Required file not found: {input_path / 'load.csv'}")
        if missing:
            LOGGER.warning("Optional source files missing: %s", ", ".join(missing))
        if not frames:
            raise ValueError(f"No usable CSV files found under {input_path}")

        merged = frames[0]
        for frame in frames[1:]:
            duplicate = merged.columns.intersection(frame.columns)
            if len(duplicate):
                LOGGER.warning("Combining duplicate raw columns: %s", list(duplicate))
                for column in duplicate:
                    merged[column] = merged[column].combine_first(frame[column])
                frame = frame.drop(columns=list(duplicate))
            merged = merged.join(frame, how="outer")

        full_index = pd.date_range(
            merged.index.min().floor(self.frequency),
            merged.index.max().floor(self.frequency),
            freq=self.frequency,
            name="datetime",
        )
        merged = merged.reindex(full_index)
        self.observations = merged.replace([np.inf, -np.inf], np.nan).copy()
        for column in merged.columns:
            # Preserve regime changes and true labels; clipping is feature-only.
            merged[column] = merged[column].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
            if self.clip_nonnegative and not re.search(r"temp|delta|diff", column, flags=re.I):
                merged[column] = merged[column].clip(lower=0.0)

        for target in TARGETS:
            if target not in merged:
                raise KeyError(f"load.csv must contain target column {target!r}")
        return merged.astype(float)


class PriceSchedule:
    """Map official price workbooks to arbitrary 15-minute timestamps."""

    def __init__(self, month_slots: Mapping[int, np.ndarray] | None = None) -> None:
        self.month_slots = {int(k): np.asarray(v, dtype=float) for k, v in (month_slots or {}).items()}

    @classmethod
    def from_excel(cls, path: str | Path | None) -> "PriceSchedule":
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f'Official tariff file required: {path}')
        try:
            sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        except Exception as exc:
            raise ValueError(f'Unable to parse tariff workbook {path}') from exc

        month_values: dict[int, list[float]] = {}
        for table in sheets.values():
            if table.empty:
                continue
            columns = list(table.columns)
            month_columns: dict[int, Any] = {}
            for column in columns:
                text = str(column).strip()
                match = re.search(r"(?<!\d)(1[0-2]|[1-9])\s*(?:month|月)?", text, flags=re.I)
                if match:
                    month_columns[int(match.group(1))] = column
            if len(month_columns) >= 2:
                for month, column in month_columns.items():
                    vals = pd.to_numeric(table[column], errors="coerce").dropna().tolist()
                    if vals:
                        month_values.setdefault(month, []).extend(float(v) for v in vals)
                continue

            normalised = {_normalise_name(c): c for c in columns}
            month_col = next((v for k, v in normalised.items() if k in {"month", "月份", "月"}), None)
            price_col = next((v for k, v in normalised.items() if "price" in k or "电价" in str(v)), None)
            if month_col is not None and price_col is not None:
                for month, group in table.groupby(month_col):
                    match = re.search(r"1[0-2]|[1-9]", str(month))
                    if not match:
                        continue
                    vals = pd.to_numeric(group[price_col], errors="coerce").dropna().tolist()
                    month_values.setdefault(int(match.group()), []).extend(float(v) for v in vals)

        slots: dict[int, np.ndarray] = {}
        for month, values in month_values.items():
            array = np.asarray(values, dtype=float)
            array = array[np.isfinite(array)]
            if not len(array):
                continue
            if len(array) == 24:
                array = np.repeat(array, 4)
            elif len(array) == 48:
                array = np.repeat(array, 2)
            elif len(array) != 96:
                raise ValueError(f'Tariff month {month}: expected 24/48/96 slots, got {len(array)}')
            slots[month] = array[:96]
        if not slots:
            raise ValueError(f'No monthly tariff matrix found in {path}')
        return cls(slots)

    @staticmethod
    def _fallback(timestamp: pd.Timestamp) -> float:
        hour = timestamp.hour + timestamp.minute / 60.0
        if 8.0 <= hour < 11.0 or 18.0 <= hour < 21.0:
            return 1.30
        if 7.0 <= hour < 8.0 or 11.0 <= hour < 18.0 or 21.0 <= hour < 23.0:
            return 0.85
        return 0.38

    def prices(self, index: Iterable[pd.Timestamp]) -> pd.Series:
        dt_index = pd.DatetimeIndex(index)
        result = np.empty(len(dt_index), dtype=float)
        for i, timestamp in enumerate(dt_index):
            values = self.month_slots.get(timestamp.month)
            if values is None:
                raise ValueError(f'No official tariff for month {timestamp.month}')
            else:
                slot = timestamp.hour * 4 + timestamp.minute // 15
                result[i] = float(values[min(slot, len(values) - 1)])
        return pd.Series(result, index=dt_index, name="feat_price")


def classify_gas_columns(columns: Iterable[str]) -> dict[str, dict[str, list[str]]]:
    groups = {
        gas: {"production": [], "user": [], "mixed_in": [], "generator": [], "holder": []}
        for gas in GAS_TYPES
    }
    for column in columns:
        gas = _gas_type(column)
        if gas is None:
            continue
        name = column.lower()
        if "generator_use" in name or ("generator" in name and "gas" in name):
            group = "generator"
        elif "holder" in name or "gasometer" in name:
            group = "holder"
        elif "into_gas_mixed" in name or ("mixed" in name and "into" in name):
            group = "mixed_in"
        elif "user" in name or "heater" in name or "consumer" in name or "consume" in name:
            group = "user"
        elif re.search(r"blast_furnace_\d+$|coke_oven_\d+$|converter_\d+$", name):
            group = "production"
        elif any(token in name for token in ("produce", "generation", "output", "supply")):
            group = "production"
        else:
            continue
        groups[gas][group].append(column)
    return groups


class IndustrialFeatureBuilder:
    """Create causal, physically meaningful features with the required prefix."""

    def __init__(
        self,
        price_schedule: PriceSchedule,
        holder_capacity: float = 200_000.0,
        max_lag_sources: int = 32,
    ) -> None:
        self.price_schedule = price_schedule
        self.holder_capacity = holder_capacity
        self.max_lag_sources = max_lag_sources

    @staticmethod
    def _sum(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
        if not columns:
            return pd.Series(0.0, index=frame.index)
        return frame[list(columns)].sum(axis=1)

    def build(self, raw: pd.DataFrame) -> pd.DataFrame:
        result = raw.copy()
        index = result.index
        groups = classify_gas_columns(raw.columns)

        minute = index.hour * 60 + index.minute
        result["feat_cal_hour_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
        result["feat_cal_hour_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
        result["feat_cal_dow_sin"] = np.sin(2.0 * np.pi * index.dayofweek / 7.0)
        result["feat_cal_dow_cos"] = np.cos(2.0 * np.pi * index.dayofweek / 7.0)
        result["feat_cal_month_sin"] = np.sin(2.0 * np.pi * (index.month - 1) / 12.0)
        result["feat_cal_month_cos"] = np.cos(2.0 * np.pi * (index.month - 1) / 12.0)
        result["feat_cal_is_weekend"] = (index.dayofweek >= 5).astype(float)
        result["feat_price"] = self.price_schedule.prices(index).values

        lhv = {"blast_furnace": 3.2, "coke": 17.0, "converter": 7.5}
        supply_total = pd.Series(0.0, index=index)
        user_total = pd.Series(0.0, index=index)
        holder_total = pd.Series(0.0, index=index)
        energy_supply = pd.Series(0.0, index=index)
        energy_demand = pd.Series(0.0, index=index)
        mixed_users = [c for c in raw.columns if "mixed" in c.lower() and "user" in c.lower()]
        mixed_demand = self._sum(raw, mixed_users)

        mixed_flows = {
            gas: self._sum(raw, groups[gas]["mixed_in"]) for gas in GAS_TYPES
        }
        mixed_flow_total = sum(mixed_flows.values(), start=pd.Series(0.0, index=index))
        for gas in GAS_TYPES:
            supply = self._sum(raw, groups[gas]["production"])
            users = self._sum(raw, groups[gas]["user"])
            share = mixed_flows[gas] / mixed_flow_total.replace(0.0, np.nan)
            share = share.ffill().fillna(1.0 / len(GAS_TYPES)).clip(0.0, 1.0)
            allocated_mixed = mixed_demand * share
            holder = self._sum(raw, groups[gas]["holder"])
            available = supply - users - allocated_mixed

            result[f"feat_{gas}_supply"] = supply
            result[f"feat_{gas}_priority_demand"] = users + allocated_mixed
            result[f"feat_{gas}_available"] = available
            result[f"feat_{gas}_mixed_share"] = share
            result[f"feat_{gas}_holder"] = holder
            supply_total += supply
            user_total += users + allocated_mixed
            holder_total += holder
            energy_supply += supply * lhv[gas]
            energy_demand += (users + allocated_mixed) * lhv[gas]

        result["feat_gas_supply_total"] = supply_total
        result["feat_gas_priority_demand_total"] = user_total
        result["feat_gas_available_total"] = supply_total - user_total
        result["feat_gas_energy_supply_mj"] = energy_supply
        result["feat_gas_energy_demand_mj"] = energy_demand
        result["feat_gas_energy_available_mj"] = energy_supply - energy_demand
        result["feat_holder_total"] = holder_total
        result["feat_holder_ratio"] = (holder_total / self.holder_capacity).clip(0.0, 2.0)
        result["feat_holder_margin_low"] = holder_total - 0.15 * self.holder_capacity
        result["feat_holder_margin_high"] = 0.90 * self.holder_capacity - holder_total
        result["feat_holder_delta_1"] = holder_total.diff().fillna(0.0)
        result["feat_holder_delta_4"] = holder_total.diff(4).fillna(0.0)

        if all(target in raw for target in TARGETS):
            result["feat_generator_other"] = (raw["generator_all"] - raw["generator_1"]).clip(lower=0.0)
            result["feat_generator_1_share"] = (
                raw["generator_1"] / raw["generator_all"].replace(0.0, np.nan)
            ).ffill().fillna(200.0 / 440.0).clip(0.0, 1.0)

        priority = list(TARGETS)
        priority += [c for c in raw.columns if any(token in c.lower() for token in ("gas", "furnace", "coke", "converter", "holder", "user"))]
        priority += list(raw.columns)
        lag_sources = list(dict.fromkeys(c for c in priority if c in raw))[: self.max_lag_sources]
        lag_steps = (1, 2, 4, 8, 12, 24, 48, 96, 192, 672)
        lag_features: dict[str, pd.Series] = {}
        for column in lag_sources:
            for lag in lag_steps:
                lag_features[f"feat_{column}_lag_{lag}"] = raw[column].shift(lag)
            past = raw[column].shift(1)
            for window in (4, 8, 24, 96):
                rolling = past.rolling(window, min_periods=1)
                lag_features[f"feat_{column}_roll_mean_{window}"] = rolling.mean()
                lag_features[f"feat_{column}_roll_std_{window}"] = rolling.std().fillna(0.0)

        if lag_features:
            result = pd.concat([result, pd.DataFrame(lag_features, index=index)], axis=1)

        engineered = [c for c in result.columns if c.startswith("feat_")]
        invalid = [c for c in engineered if not c.startswith("feat_")]
        if invalid:
            raise AssertionError(f"Engineered columns without feat_ prefix: {invalid}")
        result[engineered] = result[engineered].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return result.astype(float)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    positive = np.abs(y_true[np.isfinite(y_true) & (np.abs(y_true) > 1e-9)])
    scale = float(np.median(positive)) if len(positive) else 1.0
    denominator = np.maximum(np.abs(y_true), max(1e-6, 0.01 * scale))
    return float(np.mean(np.abs(y_true - y_pred) / denominator))


def rolling_time_splits(
    index: pd.DatetimeIndex,
    n_splits: int = 3,
    validation_size: int = 96,
    min_train_size: int = 512,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return chronological expanding-window train/validation index positions."""

    n = len(index)
    if n < min_train_size + validation_size:
        return []
    latest_train_end = n - validation_size
    starts = np.linspace(
        min_train_size,
        latest_train_end,
        num=max(1, min(n_splits, latest_train_end - min_train_size + 1)),
        dtype=int,
    )
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_end in sorted(set(int(value) for value in starts)):
        validation_end = min(n, train_end + validation_size)
        if validation_end > train_end:
            splits.append((np.arange(train_end), np.arange(train_end, validation_end)))
    return splits


@dataclass
class ForecastConfig:
    max_horizon: int = 96
    max_train_rows: int = 50_000
    max_features: int = 320
    n_estimators: int = 220
    learning_rate: float = 0.035
    num_leaves: int = 31
    min_child_samples: int = 30
    validation_fraction: float = 0.15
    random_state: int = 2026
    fast: bool = False
    ensemble: bool = True
    max_model_threads: int = 8


class DirectMultiHorizonForecaster:
    """Chronological direct LightGBM models blended with causal seasonal forecasts."""

    anchor_horizons = (1, 2, 4, 8, 12, 24, 36, 48, 72, 96)

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.models: dict[tuple[str, int], list[Any]] = {}
        self.blend_weights: dict[tuple[str, int], float] = {}
        self.feature_columns: list[str] = []
        self.trained_horizons: list[int] = []
        self.validation_metrics: list[dict[str, Any]] = []
        self.model_kind = "seasonal"
        self._history: pd.DataFrame | None = None

    def _select_features(self, frame: pd.DataFrame) -> list[str]:
        numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        preferred = list(TARGETS)
        preferred += [c for c in numeric if c.startswith("feat_generator")]
        preferred += [c for c in numeric if c.startswith("feat_gas_") or c.startswith("feat_holder")]
        preferred += [c for c in numeric if c.startswith("feat_cal_") or c == "feat_price"]
        preferred += numeric
        return list(dict.fromkeys(preferred))[: self.config.max_features]

    @staticmethod
    def _with_target_calendar(x: pd.DataFrame, horizon: int) -> pd.DataFrame:
        result = x.copy()
        target_time = result.index + pd.to_timedelta(horizon * STEP_MINUTES, unit="min")
        minute = target_time.hour * 60 + target_time.minute
        result["feat_target_hour_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
        result["feat_target_hour_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
        result["feat_target_dow_sin"] = np.sin(2.0 * np.pi * target_time.dayofweek / 7.0)
        result["feat_target_dow_cos"] = np.cos(2.0 * np.pi * target_time.dayofweek / 7.0)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _seasonal_prediction(series: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
        values: list[float] = []
        recent_default = float(series.loc[: origins.min()].tail(8).median()) if len(series) else 0.0
        for origin in origins:
            available = series.loc[:origin]
            current = float(available.iloc[-1]) if len(available) else recent_default
            daily_time = origin + pd.Timedelta(minutes=STEP_MINUTES * (horizon - 96))
            weekly_time = origin + pd.Timedelta(minutes=STEP_MINUTES * (horizon - 672))
            daily = float(series.get(daily_time, current)) if daily_time <= origin else current
            weekly = float(series.get(weekly_time, daily)) if weekly_time <= origin else daily
            recent = float(available.tail(8).median()) if len(available) else current
            values.append(0.72 * daily + 0.18 * weekly + 0.10 * recent)
        return np.asarray(values, dtype=float)

    def _make_model(self, n_estimators: int | None = None) -> Any | None:
        try:
            from lightgbm import LGBMRegressor

            self.model_kind = "lightgbm"
            return LGBMRegressor(
                objective="regression_l1",
                n_estimators=n_estimators or self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                max_depth=-1,
                min_child_samples=self.config.min_child_samples,
                subsample=0.90,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=0.20,
                random_state=self.config.random_state,
                n_jobs=max(1, min(self.config.max_model_threads, os.cpu_count() or 2)),
                verbosity=-1,
            )
        except ImportError:
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor

                self.model_kind = "hist_gradient_boosting"
                return HistGradientBoostingRegressor(
                    loss="absolute_error",
                    learning_rate=0.06,
                    max_iter=min(n_estimators or self.config.n_estimators, 140),
                    max_leaf_nodes=self.config.num_leaves,
                    min_samples_leaf=self.config.min_child_samples,
                    l2_regularization=0.2,
                    random_state=self.config.random_state,
                )
            except ImportError:
                LOGGER.warning("Neither LightGBM nor scikit-learn is installed; using seasonal forecasts")
                return None

    def _make_ensemble(self, n_estimators: int | None = None) -> list[Any]:
        """Build a small heterogeneous ensemble without requiring every package."""

        models: list[Any] = []
        primary = self._make_model(n_estimators)
        if primary is not None:
            models.append(primary)
        if self.config.ensemble:
            try:
                from catboost import CatBoostRegressor

                models.append(
                    CatBoostRegressor(
                        loss_function="MAE",
                        iterations=min(n_estimators or self.config.n_estimators, 260),
                        learning_rate=0.045,
                        depth=8,
                        l2_leaf_reg=6.0,
                        random_seed=self.config.random_state,
                        thread_count=max(1, min(self.config.max_model_threads, os.cpu_count() or 2)),
                        verbose=False,
                        allow_writing_files=False,
                    )
                )
                self.model_kind = "lightgbm+catboost" if primary is not None else "catboost"
            except ImportError:
                pass
        return models

    def _training_data(
        self,
        frame: pd.DataFrame,
        target: str,
        horizon: int,
        train_until: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.Series]:
        label = frame[target].shift(-horizon)
        latest_feature_time = train_until - pd.Timedelta(minutes=STEP_MINUTES * horizon)
        mask = (frame.index <= latest_feature_time) & label.notna()
        rows = frame.index[mask]
        if len(rows) > self.config.max_train_rows:
            rows = rows[-self.config.max_train_rows :]
        x = self._with_target_calendar(frame.loc[rows, self.feature_columns], horizon)
        return x, label.loc[rows].astype(float)

    def _calibrate_weight(
        self,
        frame: pd.DataFrame,
        target: str,
        horizon: int,
        train_until: pd.Timestamp,
    ) -> float:
        x, y = self._training_data(frame, target, horizon, train_until)
        if len(y) < 240:
            return 0.72
        validation_size = max(32, int(len(y) * self.config.validation_fraction))
        splits = rolling_time_splits(
            pd.DatetimeIndex(x.index), n_splits=3, validation_size=validation_size, min_train_size=max(120, len(y) // 3)
        )
        if not splits:
            return 0.72
        model_errors: list[float] = []
        naive_errors: list[float] = []
        # The two most recent folds represent the deployment regime while keeping
        # calibration bounded for the 192-model full-horizon fit.
        for train_positions, validation_positions in splits[-2:]:
            # Purge labels whose target timestamp is later than validation origin.
            valid_start = x.index[validation_positions[0]]
            train_positions = train_positions[x.index[train_positions] + pd.Timedelta(minutes=15*horizon) <= valid_start]
            model = self._make_model(max(80, self.config.n_estimators // 2))
            if model is None:
                return 0.0
            model.fit(x.iloc[train_positions], y.iloc[train_positions])
            model_pred = np.asarray(model.predict(x.iloc[validation_positions]), dtype=float)
            origins = pd.DatetimeIndex(x.index[validation_positions])
            naive_pred = self._seasonal_prediction(frame[target], origins, horizon)
            model_errors.append(safe_mape(y.iloc[validation_positions].to_numpy(), model_pred))
            naive_errors.append(safe_mape(y.iloc[validation_positions].to_numpy(), naive_pred))
        model_error = float(np.mean(model_errors))
        naive_error = float(np.mean(naive_errors))
        weight = float(np.clip(naive_error / max(model_error + naive_error, 1e-9), 0.15, 0.95))
        self.validation_metrics.append(
            {
                "target": target,
                "horizon": horizon,
                "model_mape": model_error,
                "seasonal_mape": naive_error,
                "model_weight": weight,
                "validation_rows": int(sum(len(split[1]) for split in splits[-2:])),
                "validation_folds": len(splits[-2:]),
            }
        )
        return weight

    def fit(self, frame: pd.DataFrame, train_until: pd.Timestamp) -> "DirectMultiHorizonForecaster":
        self._history = frame[list(TARGETS)].copy()
        self.feature_columns = self._select_features(frame)
        anchors = sorted(set(h for h in self.anchor_horizons if h <= self.config.max_horizon))
        if self.config.max_horizon not in anchors:
            anchors.append(self.config.max_horizon)
        self.trained_horizons = anchors if self.config.fast else list(range(1, self.config.max_horizon + 1))

        LOGGER.info("Calibrating chronological blend weights on %d anchor horizons", len(anchors))
        for target in TARGETS:
            anchor_weights = {
                h: self._calibrate_weight(frame, target, h, train_until) for h in anchors
            }
            x_anchor = np.asarray(sorted(anchor_weights), dtype=float)
            y_anchor = np.asarray([anchor_weights[int(h)] for h in x_anchor], dtype=float)
            interpolated = np.interp(
                np.arange(1, self.config.max_horizon + 1), x_anchor, y_anchor
            )
            for h, value in enumerate(interpolated, start=1):
                self.blend_weights[(target, h)] = float(value)

        total = len(TARGETS) * len(self.trained_horizons)
        completed = 0
        for target in TARGETS:
            for horizon in self.trained_horizons:
                x, y = self._training_data(frame, target, horizon, train_until)
                models = self._make_ensemble()
                if models and len(y) >= 64:
                    fitted: list[Any] = []
                    for model in models:
                        model.fit(x, y)
                        fitted.append(model)
                    self.models[(target, horizon)] = fitted
                completed += 1
                if completed == total or completed % 12 == 0:
                    LOGGER.info("Trained %d/%d direct models", completed, total)
        return self

    def _predict_trained(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        target: str,
        horizon: int,
    ) -> np.ndarray:
        naive = self._seasonal_prediction(frame[target], origins, horizon)
        models = self.models.get((target, horizon), [])
        if not models:
            prediction = naive
        else:
            x = self._with_target_calendar(frame.loc[origins, self.feature_columns], horizon)
            model_prediction = np.mean(
                [np.asarray(model.predict(x), dtype=float) for model in models], axis=0
            )
            weight = self.blend_weights.get((target, horizon), 0.72)
            prediction = weight * model_prediction + (1.0 - weight) * naive
        lower, upper = CAPACITY_LIMITS[target]
        return np.clip(prediction, lower, upper)

    def predict(self, frame: pd.DataFrame, origins: Iterable[pd.Timestamp]) -> pd.DataFrame:
        origin_index = pd.DatetimeIndex(origins, name="datetime")
        missing = origin_index.difference(frame.index)
        if len(missing):
            raise KeyError(f"Forecast origins are absent from aligned data: {list(missing[:5])}")
        prediction_columns: dict[str, np.ndarray] = {}
        trained = np.asarray(self.trained_horizons, dtype=int)
        all_horizons = np.arange(1, self.config.max_horizon + 1, dtype=int)
        for target in TARGETS:
            matrix = np.column_stack(
                [self._predict_trained(frame, origin_index, target, int(h)) for h in trained]
            )
            if not np.array_equal(trained, all_horizons):
                expanded = np.vstack(
                    [np.interp(all_horizons, trained, row) for row in matrix]
                )
            else:
                expanded = matrix
            for j, horizon in enumerate(all_horizons):
                minutes = int(horizon * STEP_MINUTES)
                prediction_columns[f"{target}_t+{minutes}_pred"] = expanded[:, j]
        result = pd.DataFrame(prediction_columns, index=origin_index)
        # Reconcile the separately fitted targets. generator_1 is the 4x50MW
        # group and the remaining two units have at most 2x120MW output.
        for horizon in all_horizons:
            minutes = int(horizon * STEP_MINUTES)
            group_column = f"generator_1_t+{minutes}_pred"
            total_column = f"generator_all_t+{minutes}_pred"
            total = result[total_column].clip(0.0, 440.0)
            lower = (total - 240.0).clip(lower=0.0)
            upper = total.clip(upper=200.0)
            result[group_column] = result[group_column].clip(lower=lower, upper=upper)
            result[total_column] = total
        return result

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": self.models,
                "blend_weights": self.blend_weights,
                "feature_columns": self.feature_columns,
                "trained_horizons": self.trained_horizons,
                "config": asdict(self.config),
                "model_kind": self.model_kind,
            },
            path / "direct_forecaster.joblib",
            compress=3,
        )
        pd.DataFrame(self.validation_metrics).to_csv(
            path / "validation_metrics.csv", index=False, encoding="utf-8"
        )


def _seasonal_future(series: pd.Series, origin: pd.Timestamp, horizon: int) -> np.ndarray:
    clean = series.loc[:origin].astype(float)
    if clean.empty:
        return np.zeros(horizon, dtype=float)
    recent = float(clean.tail(8).median())
    predictions = np.empty(horizon, dtype=float)
    for h in range(1, horizon + 1):
        daily_time = origin + pd.Timedelta(minutes=STEP_MINUTES * (h - 96))
        weekly_time = origin + pd.Timedelta(minutes=STEP_MINUTES * (h - 672))
        daily = float(clean.get(daily_time, recent)) if daily_time <= origin else recent
        weekly = float(clean.get(weekly_time, daily)) if weekly_time <= origin else daily
        predictions[h - 1] = max(0.0, 0.75 * daily + 0.15 * weekly + 0.10 * recent)
    return predictions


class GasResourceForecaster:
    """Forecast gas supply left after priority production users are served."""

    def __init__(self, raw: pd.DataFrame) -> None:
        self.raw = raw
        self.groups = classify_gas_columns(raw.columns)

    def _aggregate(self, columns: Sequence[str]) -> pd.Series:
        if not columns:
            return pd.Series(0.0, index=self.raw.index)
        return self.raw[list(columns)].sum(axis=1)

    def forecast(self, origin: pd.Timestamp, horizon: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        from dispatch import observed_surplus
        net, caps, _, _, _ = observed_surplus(self.raw, origin, horizon)
        return net, caps

    def estimate_efficiency(self, origin: pd.Timestamp) -> dict[str, float]:
        # MW produced per gas-flow unit. Defaults assume gas columns are Nm3/h.
        defaults = {
            "blast_furnace": 3.2 * 0.32 / 3600.0,
            "coke": 17.0 * 0.32 / 3600.0,
            "converter": 7.5 * 0.32 / 3600.0,
        }
        history = self.raw.loc[:origin]
        gas_series = [self._aggregate(self.groups[gas]["generator"]).loc[:origin] for gas in GAS_TYPES]
        if "generator_all" not in history or any(not self.groups[g]["generator"] for g in GAS_TYPES):
            return defaults
        x = np.column_stack([s.to_numpy(dtype=float) for s in gas_series])
        y = history["generator_all"].to_numpy(dtype=float)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y) & (y >= 0.0) & (x.sum(axis=1) > 0.0)
        if valid.sum() < 64:
            return defaults
        try:
            from scipy.optimize import nnls

            coefficients, _ = nnls(x[valid], y[valid])
        except Exception as exc:
            LOGGER.warning("Gas efficiency estimation failed (%s); using physical defaults", exc)
            return defaults
        result: dict[str, float] = {}
        for i, gas in enumerate(GAS_TYPES):
            lower = defaults[gas] / 20.0
            upper = defaults[gas] * 20.0
            result[gas] = float(np.clip(coefficients[i], lower, upper))
        return result

    def initial_holder(self, origin: pd.Timestamp, capacity: float = 200_000.0) -> float:
        from dispatch import observed_surplus
        _, _, level, measured_capacity, _ = observed_surplus(self.raw, origin, 1)
        if capacity != measured_capacity:
            raise ValueError(f'Configured capacity {capacity} differs from observed holder {measured_capacity}')
        if not .15*capacity <= level <= .9*capacity:
            raise ValueError('Observed holder outside safety band')
        return level


@dataclass
class DispatchConfig:
    holder_capacity: float = 200_000.0
    holder_min_fraction: float = 0.15
    holder_max_fraction: float = 0.90
    unit_ratings_mw: tuple[float, ...] = (50.0, 50.0, 50.0, 50.0, 120.0, 120.0)
    minimum_load_fraction: float = 0.60
    ramp_fraction_per_minute: float = 0.10
    interval_minutes: int = STEP_MINUTES
    revenue_scale: float = 1000.0
    flare_penalty: float = 0.20
    emergency_gas_penalty: float = 50.0
    ramp_penalty: float = 0.25
    startup_penalty: float = 30.0
    terminal_storage_penalty: float = 0.05
    solver_time_limit: int = 25


@dataclass
class DispatchResult:
    gas_plan: pd.DataFrame
    status: str
    holder_path: np.ndarray
    power_path: np.ndarray
    diagnostics: dict[str, float] = field(default_factory=dict)


class MILPDispatcher:
    """Verified integer dispatch for every horizon; never relax unit constraints."""

    def __init__(self, config: DispatchConfig | None = None) -> None:
        self.config = config or DispatchConfig()

    def optimize(self, timestamps, prices, net_inflow, gas_caps, efficiency, initial_holder):
        from dispatch import solve_dispatch
        return solve_dispatch(self.config, timestamps, prices, net_inflow,
                              gas_caps, efficiency, initial_holder)


def validate_submission_frames(
    input_frame: pd.DataFrame,
    short_result: pd.DataFrame,
    long_result: pd.DataFrame,
    opt_result: pd.DataFrame,
) -> None:
    for name, frame in {
        "input.csv": input_frame,
        "s_result.csv": short_result,
        "l_result.csv": long_result,
        "opt_result.csv": opt_result,
    }.items():
        if "datetime" not in frame.columns:
            raise ValueError(f"{name} is missing datetime")
        if frame["datetime"].duplicated().any():
            raise ValueError(f"{name} contains duplicate datetime values")
        numeric = frame.drop(columns="datetime")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains non-finite predictions")
    engineered = [c for c in input_frame.columns if c not in {"datetime"} and c.startswith("feat")]
    if any(not c.startswith("feat_") for c in engineered):
        raise ValueError("Every engineered input column must start with feat_")
    if any(not c.startswith("opt_") for c in opt_result.columns if c != "datetime"):
        raise ValueError("Every optimized gas column must start with opt_")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(destination)
