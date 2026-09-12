"""Regenerate the delivered ``input.csv`` with the current quality repair.

Why this exists
---------------
The competition awards quality points for explicit outlier handling in
``input.csv``, and the delivered table must exist in TWO places:

  * the package root ``input.csv`` (shipped answer file);
  * ``results_prebaked/input.csv`` (the fallback ``run.sh`` copies in when the
    pipeline fails or no data package is found).

Those two copies drifted.  The zip that produced ``quality=40/50`` with
``out=0`` carried a 308-column table with zero outlier flags and 312 values
outside the training fence, while ``results_prebaked/`` carried the same stale
file.  A failed pipeline therefore re-served the very table that lost the
points.  This script rebuilds both copies from the current code and refuses to
write anything unless the audit passes.

It does NOT retrain the model: only the delivered input table changes, so a
controlled submission can attribute any score movement to outlier handling
alone while ``s_result.csv`` stays byte-identical.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiment import discover, read_observations  # noqa: E402
from forecasting import features  # noqa: E402
from main import _atomic_csv, _format_datetime  # noqa: E402
from run_official import _quality_input_frame, _regime_window, _tukey_fence  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(delivered: pd.DataFrame, history: pd.DataFrame, origins: pd.DatetimeIndex,
          original: pd.DataFrame | None = None):
    """Return (problems, report) for the delivered table.

    ``original`` is the pre-repair slice.  The "was a normal value repaired"
    check MUST use it rather than the delivered values: clipping to the union
    upper bound can land a repaired value back inside the post-regime fence,
    so testing the delivered table would report false positives.
    """
    problems, report = [], []
    dt = pd.to_datetime(delivered["datetime"])
    body = delivered.drop(columns=["datetime"])
    raw_cols = [c for c in body.columns if not c.startswith("feat_")]
    report.append(f"rows={len(delivered)}  columns={len(body.columns)}")
    if len(delivered) != len(origins):
        problems.append(f"row count {len(delivered)} != origins {len(origins)}")
    if not dt.is_unique:
        problems.append("duplicate datetimes")
    elif len(dt) > 1 and (dt.diff().dropna().dt.total_seconds() != 900.0).any():
        problems.append("timestamps are not a strict 15-minute grid")
    arr = body.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        problems.append(f"non-finite values present ({int((~np.isfinite(arr)).sum())})")

    missing_flags = [c for c in raw_cols if f"feat_{c}_outlier" not in body.columns]
    if missing_flags:
        problems.append(f"raw fields without an outlier flag: {missing_flags}")
    report.append(f"raw columns={len(raw_cols)}  feat_={sum(c.startswith('feat_') for c in body.columns)}"
                  f"  outlier flags={sum(c.endswith('_outlier') for c in body.columns)}")

    regime = _regime_window(history)
    repaired, outside = 0, 0
    for c in raw_cols:
        obs = history[c].replace([np.inf, -np.inf], np.nan).dropna()
        if obs.empty:
            continue
        lo, hi = _tukey_fence(obs)
        if regime is not None:
            post = regime[c].replace([np.inf, -np.inf], np.nan).dropna()
            if not post.empty:
                lo2, hi2 = _tukey_fence(post)
                lo, hi = min(lo, lo2), max(hi, hi2)
        lo32 = np.nextafter(np.float32(lo), np.float32(np.inf))
        hi32 = np.nextafter(np.float32(hi), np.float32(-np.inf))
        if not hi32 > lo32:
            lo32 = hi32 = np.float32(hi)  # degenerate fence, matches production
        v = body[c].to_numpy(dtype=float)
        outside += int(((v < lo32) | (v > hi32)).sum())
        flag = body.get(f"feat_{c}_outlier")
        repaired += int(np.nansum(flag.to_numpy())) if flag is not None else 0
    report.append(f"repaired cells={repaired}  cells outside the union fence={outside}")
    if outside:
        problems.append(f"{outside} delivered cells sit outside the fence that produced them")
    if regime is not None:
        normal_but_repaired = 0
        source = original.drop(columns=["datetime"]) if original is not None else body
        for c in raw_cols:
            post = regime[c].replace([np.inf, -np.inf], np.nan).dropna()
            flag = body.get(f"feat_{c}_outlier")
            if post.empty or flag is None:
                continue
            lo, hi = _tukey_fence(post)
            v = source[c].to_numpy(dtype=float)
            normal_but_repaired += int(((v >= lo) & (v <= hi) & (flag.to_numpy() == 1)).sum())
        if normal_but_repaired:
            problems.append(f"{normal_but_repaired} post-regime-normal cells were repaired")
        report.append(f"post-regime-normal cells repaired={normal_but_repaired}")
    return problems, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Write the files (default is a dry run)")
    ap.add_argument("--prebaked-dir", type=Path, default=ROOT / "results_prebaked")
    args = ap.parse_args()

    directory = discover(ROOT)
    test_paths = list(directory.parent.rglob("Pre_test_load.csv"))
    if len(test_paths) != 1:
        raise ValueError("Expected exactly one official Pre_test_load.csv")
    origin = pd.to_datetime(pd.read_csv(test_paths[0], usecols=["datetime"]).datetime).min()
    cutoff = origin - pd.Timedelta(minutes=15)
    train = read_observations(directory).loc[:cutoff]
    test = read_observations(test_paths[0].parent)
    combined = pd.concat([train, test]).sort_index()
    x = features(combined)
    origins = pd.DatetimeIndex(
        sorted(pd.to_datetime(pd.read_csv(test_paths[0], usecols=["datetime"]).datetime
                              ).drop_duplicates()))

    delivered = _format_datetime(_quality_input_frame(x, cutoff, train).loc[origins])
    original = _format_datetime(x.loc[origins])
    problems, report = audit(delivered, train, origins, original)

    print(f"cutoff={cutoff}  origins={len(origins)}")
    for line in report:
        print("  " + line)
    if problems:
        print("\nAUDIT FAILED:")
        for p in problems:
            print("  - " + p)
        return 1

    targets = [ROOT / "input.csv", args.prebaked_dir / "input.csv"]
    for path in targets:
        old = _sha(path) if path.exists() else "(absent)"
        if args.apply:
            _atomic_csv(delivered, path)
            print(f"wrote {path.relative_to(ROOT)}  {old[:12]} -> {_sha(path)[:12]}")
        else:
            print(f"[dry-run] would write {path.relative_to(ROOT)} (currently {old[:12]})")
    print("\nAUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
