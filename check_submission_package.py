"""Pre-flight check for the submission package (HANDOFF stage A, steps 2-4).

The first scored submission returned ``quality=40/50`` with ``out=0`` and
``missing_s_result``, and the cause was not the model: the delivered
``input.csv`` was a stale 308-column table carrying no outlier evidence, and the
``results_prebaked/`` fallback held the same stale file.  A wasted submission is
expensive (5 per day, and each must change exactly one thing), so validate the
whole package locally before sending it.

Run with no arguments.  Exit code 0 means the package is internally consistent
and safe to submit.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiment import discover, read_observations  # noqa: E402
from forecasting import features  # noqa: E402
from run_official import _quality_input_frame, _regime_window, _tukey_fence  # noqa: E402

ROOT_FILES = ["input.csv", "s_result.csv", "l_result.csv", "opt_result.csv",
              "result.csv", "s_result.json"]
PREBAKED_FILES = ROOT_FILES
SHORT_COLUMNS = 1 + 2 * 8      # datetime + 2 targets x h1..8
LONG_COLUMNS = 1 + 2 * 96      # datetime + 2 targets x h1..96

# input.csv is written as float32 with "%.6f", so reading it back can move a cell
# by at most half the last decimal (5e-7).  Comparisons against the in-memory
# float32 table must allow exactly that much slack -- a tighter tolerance only
# reports the file format back to us.
CSV_ROUNDTRIP_TOL = 5e-7 + 1e-9


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []

    # ---- 1. required files present -------------------------------------
    for name in ROOT_FILES:
        if not (ROOT / name).exists():
            problems.append(f"missing root file: {name}")
    if not (ROOT / "run.sh").exists() or not (ROOT / "run.ps1").exists():
        problems.append("missing run.sh / run.ps1 evaluation entry point")
    if not (ROOT / "artifacts" / "development" / "selection.json").exists():
        problems.append("missing artifacts/development/selection.json (run.sh needs it)")

    if problems:
        _report(problems, notes)
        return 1

    # ---- 2. shapes and internal consistency ----------------------------
    short = pd.read_csv(ROOT / "s_result.csv", encoding="utf-8-sig")
    long = pd.read_csv(ROOT / "l_result.csv", encoding="utf-8-sig")
    opt = pd.read_csv(ROOT / "opt_result.csv", encoding="utf-8-sig")
    result = pd.read_csv(ROOT / "result.csv", encoding="utf-8-sig")
    inputs = pd.read_csv(ROOT / "input.csv", encoding="utf-8-sig")

    if short.shape[1] != SHORT_COLUMNS:
        problems.append(f"s_result.csv has {short.shape[1]} columns, expected {SHORT_COLUMNS}")
    if long.shape[1] != LONG_COLUMNS:
        problems.append(f"l_result.csv has {long.shape[1]} columns, expected {LONG_COLUMNS}")
    if opt.shape != (96, 4):
        problems.append(f"opt_result.csv shape {opt.shape}, expected (96, 4)")
    if not (short.columns.equals(result.columns) and short.equals(result)):
        problems.append("result.csv is not identical to s_result.csv")
    for frame, name in ((short, "s_result.csv"), (long, "l_result.csv"), (opt, "opt_result.csv")):
        if frame["datetime"].duplicated().any():
            problems.append(f"{name}: duplicate datetimes")
        if not np.isfinite(frame.drop(columns="datetime").to_numpy(dtype=float)).all():
            problems.append(f"{name}: non-finite values")
    opt_cols = [c for c in opt.columns if c != "datetime"]
    if not all(c.startswith("opt_") for c in opt_cols):
        problems.append(f"opt_result.csv columns not all opt_-prefixed: {opt_cols}")

    # ---- 3. s_result.json must describe s_result.csv --------------------
    sjson = json.loads((ROOT / "s_result.json").read_text(encoding="utf-8"))
    if "columns" not in sjson or "data" not in sjson:
        problems.append("s_result.json lacks the columns/data structure")
    else:
        if list(sjson["columns"]) != list(short.columns):
            problems.append("s_result.json columns do not match s_result.csv")
        if len(sjson["data"]) != len(short):
            problems.append(f"s_result.json has {len(sjson['data'])} rows, "
                            f"s_result.csv has {len(short)}")

    # Raw official fields legitimately keep their给定 names (PROJECT_RULES):
    # only ENGINEERED columns must be feat_-prefixed.  Read the authoritative
    # field list from the data rather than assuming every unprefixed name is
    # a violation.
    directory = discover(ROOT)
    test_paths = list(directory.parent.rglob("Pre_test_load.csv"))
    official_fields: set[str] = set()
    if len(test_paths) == 1:
        official_fields = set(read_observations(test_paths[0].parent).columns)

    # ---- 4. input.csv quality contract ---------------------------------
    body = inputs.drop(columns=["datetime"])
    raw_cols = [c for c in body.columns if not c.startswith("feat_")]
    unknown = [c for c in raw_cols if official_fields and c not in official_fields]
    if unknown:
        problems.append(f"input.csv: columns that are neither official raw fields "
                        f"nor feat_-prefixed: {unknown}")
    if len(inputs) != len(short):
        problems.append(f"input.csv has {len(inputs)} rows, s_result.csv has {len(short)}")
    dt = pd.to_datetime(inputs["datetime"])
    if not dt.is_unique:
        problems.append("input.csv: duplicate datetimes")
    elif len(dt) > 1 and (dt.diff().dropna().dt.total_seconds() != 900.0).any():
        problems.append("input.csv: timestamps are not a strict 15-minute grid")
    if not np.isfinite(body.to_numpy(dtype=float)).all():
        problems.append("input.csv: non-finite values")
    missing = [c for c in raw_cols if f"feat_{c}_outlier" not in body.columns]
    if missing:
        problems.append(f"input.csv: raw fields without an outlier flag: {missing}")
    flag_cols = [f"feat_{c}_outlier" for c in raw_cols if f"feat_{c}_outlier" in body.columns]
    flagged_cells = int(sum(float(body[c].to_numpy(dtype=float).sum()) for c in flag_cols))
    notes.append(f"input.csv: {len(raw_cols)} raw fields, "
                 f"{sum(c.startswith('feat_') for c in body.columns)} feat_ columns, "
                 f"{len(flag_cols)} outlier flags, {flagged_cells} cells flagged")

    # ---- 5. the delivered table must match what the pipeline would build -
    directory = discover(ROOT)
    test_paths = list(directory.parent.rglob("Pre_test_load.csv"))
    if len(test_paths) == 1:
        origin = pd.to_datetime(pd.read_csv(test_paths[0], usecols=["datetime"]).datetime).min()
        cutoff = origin - pd.Timedelta(minutes=15)
        train = read_observations(directory).loc[:cutoff]
        test = read_observations(test_paths[0].parent)
        combined = pd.concat([train, test]).sort_index()
        x = features(combined)
        origins = pd.DatetimeIndex(sorted(pd.to_datetime(
            pd.read_csv(test_paths[0], usecols=["datetime"]).datetime).drop_duplicates()))
        rebuilt = _quality_input_frame(x, cutoff, train).loc[origins]
        common = [c for c in rebuilt.columns if c in body.columns]
        if len(common) != len(body.columns):
            notes.append(f"rebuilt table has {len(rebuilt.columns)} columns, delivered "
                         f"{len(body.columns)}; comparing {len(common)}")
        rebuilt_values = rebuilt[common].to_numpy(dtype=float)
        delivered_values = body[common].to_numpy(dtype=float)
        max_diff = float(np.abs(rebuilt_values - delivered_values).max())
        if max_diff > CSV_ROUNDTRIP_TOL:
            problems.append(f"delivered input.csv differs from what the current pipeline "
                            f"would build (max |diff|={max_diff:.3e} > {CSV_ROUNDTRIP_TOL:.0e})")
        else:
            notes.append(f"delivered input.csv reproduces the current pipeline output "
                         f"(max |diff|={max_diff:.1e}, within the %.6f CSV quantum)")

        # values must never sit outside the fence that produced them
        regime = _regime_window(train)
        outside = 0
        for c in raw_cols:
            obs = train[c].replace([np.inf, -np.inf], np.nan).dropna()
            if obs.empty:
                continue
            lo, hi = _tukey_fence(obs)
            if regime is not None:
                post = regime[c].replace([np.inf, -np.inf], np.nan).dropna()
                if not post.empty:
                    lo2, hi2 = _tukey_fence(post)
                    lo, hi = min(lo, lo2), max(hi, hi2)
            # float() is essential: lo32 is an np.float32, and under NEP-50 weak
            # scalar promotion `np.float32 - 5e-7` is evaluated IN float32, where
            # the spacing near 5e4 (~4e-3) swallows the tolerance entirely.
            lo32 = float(np.nextafter(np.float32(lo), np.float32(np.inf)))
            hi32 = float(np.nextafter(np.float32(hi), np.float32(-np.inf)))
            if not hi32 > lo32:
                lo32 = hi32 = float(np.float32(hi))
            v = body[c].to_numpy(dtype=float)
            # Repair clips to [lo32, hi32] in float32; serializing at "%.6f" can
            # then round a boundary value out by <=5e-7.  Allow exactly that.
            outside += int(((v < lo32 - CSV_ROUNDTRIP_TOL) |
                            (v > hi32 + CSV_ROUNDTRIP_TOL)).sum())
        if outside:
            problems.append(f"input.csv: {outside} cells outside the fence that produced them")
        else:
            notes.append("input.csv: 0 cells outside the fence used for repair "
                         "(allowing the %.6f CSV quantum)")
    else:
        problems.append("could not locate exactly one Pre_test_load.csv for regeneration check")

    # ---- 6. fallback copies must be equivalent -------------------------
    for name in PREBAKED_FILES:
        p = ROOT / "results_prebaked" / name
        if not p.exists():
            problems.append(f"results_prebaked/{name} missing (run.sh fallback path)")
        elif sha(p) != sha(ROOT / name):
            problems.append(f"results_prebaked/{name} differs from the root copy "
                            f"({sha(p)[:12]} vs {sha(ROOT / name)[:12]})")

    _report(problems, notes, extra=_hash_table())
    return 1 if problems else 0


def _hash_table() -> list[str]:
    lines = ["", "SHA256 (first 12):"]
    for name in ROOT_FILES + ["run.sh", "run.ps1"]:
        p = ROOT / name
        if p.exists():
            lines.append(f"  {name:<20} {sha(p)[:12]}")
    for name in PREBAKED_FILES:
        p = ROOT / "results_prebaked" / name
        if p.exists():
            lines.append(f"  prebaked/{name:<11} {sha(p)[:12]}")
    return lines


def _report(problems, notes, extra=None):
    print("SUBMISSION PACKAGE CHECK")
    print("-" * 60)
    for n in notes:
        print(f"  ok   {n}")
    if problems:
        print("\nBLOCKERS:")
        for p in problems:
            print(f"  FAIL {p}")
    else:
        print("\nAll checks passed.")
    if extra:
        print("\n".join(extra))


if __name__ == "__main__":
    sys.exit(main())
