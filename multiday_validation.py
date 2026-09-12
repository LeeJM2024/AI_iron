"""Multi-day post-regime rolling validation (HANDOFF_99_PLUS.md stage C).

Why this exists
---------------
Every model choice in this project was previously judged on ONE two-day window
(Apr 23/25/27 tuning + Apr 29 validation).  A single 48-hour window has large
variance, and the team's own log records four consecutive improvements that were
real on that window and absent on the official one.  Stage C of the handoff asks
for >=8 post-regime daily folds with per-day statistics.

Protocol
--------
For each validation day D in the post-regime period (default 2025-04-19..04-30):

  * training labels end strictly before D starts (cutoff = D 00:00 - 15 min);
  * the validation block is the 96 origins of D itself, scored on h=1..8;
  * each fold trains its own model, so no fold sees its own labels.

Ensemble weights are then refitted LEAVE-ONE-DAY-OUT: the weights applied to day
D are fitted on the member predictions of every OTHER day.  That is what makes
the ensemble column honest -- otherwise April 23..29 would be scored with weights
fitted on themselves.

Two stages
----------
``--stage collect`` trains the folds and caches member predictions to an npz, so
weight/ensemble experiments cost seconds instead of half an hour.
``--stage report`` loads a cache and prints the per-day table plus the gated
verdict required by stage C6.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import discover, read_observations
from forecasting import (
    MEMBERS,
    TARGETS,
    ModelConfig,
    ResidualEnsemble,
    features,
    mape,
    select_weights,
    shifted,
    weights_for,
)

ROOT = Path(__file__).resolve().parent
DAYS = [f"2025-04-{d:02d}" for d in range(19, 31)]
HORIZONS = list(range(1, 9))


def build_cache(tag: str, cfg: ModelConfig, days, horizons, out_dir: Path, log=print):
    """Train one model per validation day and cache member predictions."""
    directory = discover(ROOT)
    raw = read_observations(directory)
    raw = raw.loc[raw.index < "2025-05-01"]
    x = features(raw)
    records = []
    meta = []
    t0 = time.perf_counter()
    for day in days:
        start = pd.Timestamp(day)
        cutoff = start - pd.Timedelta(minutes=15)
        if cutoff < raw.index.min():
            log(f"skip {day}: cutoff {cutoff} precedes data")
            continue
        origins = pd.DatetimeIndex(
            [start + pd.Timedelta(minutes=15 * i) for i in range(96)]
        ).intersection(raw.index)
        if len(origins) < 8:
            log(f"skip {day}: only {len(origins)} origins available")
            continue
        fold = ResidualEnsemble(cfg).fit(raw, x, cutoff, horizons)
        for t in TARGETS:
            for h in horizons:
                p = fold.predict_members(raw, x, origins, h, t)
                y = raw[t].reindex(shifted(origins, h)).to_numpy()
                ok = np.isfinite(y) & (np.abs(y) > 1e-8)
                records.append((day, t, h, y[ok], p[ok]))
        per_day = {}
        for t in TARGETS:
            per_day[t] = float(np.mean([
                mape(raw[t].reindex(shifted(origins, h)).to_numpy(),
                     fold.predict_members(raw, x, origins, h, t)[:, 0])
                for h in horizons
            ]))
        meta.append(dict(day=day, cutoff=str(cutoff), origins=int(len(origins)),
                         persistence_short8=per_day))
        log(f"{day}: origins={len(origins)} persistence g1={per_day['generator_1']*100:.3f}% "
            f"gall={per_day['generator_all']*100:.3f}%  ({time.perf_counter()-t0:.0f}s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "days": np.array([r[0] for r in records]),
        "targets": np.array([r[1] for r in records]),
        "horizons": np.array([r[2] for r in records], dtype=int),
    }
    for i, name in enumerate(MEMBERS):
        payload[f"y_{i}"] = np.array([r[3] for r in records], dtype=object)
        payload[f"p_{i}"] = np.array([r[4][:, i] for r in records], dtype=object)
    np.savez_compressed(out_dir / f"records_{tag}.npz", **payload)
    (out_dir / f"meta_{tag}.json").write_text(json.dumps(
        dict(tag=tag, config=asdict(cfg), days=list(days), horizons=list(horizons),
             folds=meta, seconds=time.perf_counter() - t0), indent=2), encoding="utf-8")
    log(f"cached {len(records)} (day,target,h) blocks -> records_{tag}.npz "
        f"in {time.perf_counter()-t0:.0f}s")
    return out_dir / f"records_{tag}.npz"


def load_cache(path: Path):
    data = np.load(path, allow_pickle=True)
    n = len(data["days"])
    rows = []
    for i in range(n):
        rows.append(dict(day=str(data["days"][i]), target=str(data["targets"][i]),
                         h=int(data["horizons"][i]), y=data[f"y_0"][i],
                         p=np.column_stack([data[f"p_{k}"][i] for k in range(len(MEMBERS))])))
    return rows


def _to_records(rows):
    """Adapt cache rows to select_weights' expected schema."""
    return [dict(target=r["target"], h=r["h"], y=r["y"], p=r["p"], cutoff=r["day"])
            for r in rows]


def report(path: Path, out_dir: Path, tag: str, gate_ref: float | None = None):
    rows = load_cache(path)
    days = sorted({r["day"] for r in rows})
    lines = []
    per_day = {}
    for day in days:
        held = [r for r in rows if r["day"] != day]
        test = [r for r in rows if r["day"] == day]
        wts = select_weights(_to_records(held))
        entry = {}
        for target in TARGETS:
            sel = [r for r in test if r["target"] == target]
            if not sel:
                continue
            # Match the official convention exactly: MAPE per horizon, then the
            # mean over horizons (validate_outputs.py), NOT a pooled cell MAPE.
            pers_h, ens_h = [], []
            for h in sorted({r["h"] for r in sel}):
                sub = [r for r in sel if r["h"] == h]
                y = np.concatenate([r["y"] for r in sub])
                pers_h.append(mape(y, np.concatenate([r["p"][:, 0] for r in sub])))
                ens_h.append(mape(y, np.concatenate([
                    r["p"] @ np.asarray(weights_for(wts, target, r["h"])) for r in sub])))
            entry[target] = dict(persistence=float(np.mean(pers_h)),
                                 ensemble=float(np.mean(ens_h)))
        per_day[day] = entry

    lines.append(f"{'day':<12}{'g1 persist':>12}{'g1 ens':>10}{'g1 gain':>10}"
                 f"{'gall persist':>14}{'gall ens':>11}{'gall gain':>11}")
    for day in days:
        e = per_day[day]
        g1, ga = e.get("generator_1"), e.get("generator_all")
        lines.append(
            f"{day:<12}{g1['persistence']*100:>11.3f}%{g1['ensemble']*100:>9.3f}%"
            f"{(g1['persistence']-g1['ensemble'])*100:>9.3f}"
            f"{ga['persistence']*100:>13.3f}%{ga['ensemble']*100:>10.3f}%"
            f"{(ga['persistence']-ga['ensemble'])*100:>10.3f}")

    lines.append("")
    summary = {}
    for target in TARGETS:
        gains = np.array([(per_day[d][target]["persistence"] - per_day[d][target]["ensemble"]) * 100
                          for d in days])
        ens = np.array([per_day[d][target]["ensemble"] * 100 for d in days])
        summary[target] = dict(
            pooled_ensemble=float(ens.mean()), gain_mean=float(gains.mean()),
            gain_std=float(gains.std(ddof=1)), gain_min=float(gains.min()),
            gain_p90=float(np.percentile(gains, 10)),
            days_improved=int(np.sum(gains > 0)), days=len(days))
        s = summary[target]
        lines.append(
            f"{target}: ensemble mean MAPE {s['pooled_ensemble']:.3f}%  "
            f"gain vs persistence mean {s['gain_mean']:+.3f}pp "
            f"(sd {s['gain_std']:.3f}, p10 {s['gain_p90']:+.3f}, worst {s['gain_min']:+.3f})  "
            f"improved on {s['days_improved']}/{s['days']} days")

    lines.append("")
    verdict = []
    for target in TARGETS:
        s = summary[target]
        majority = s["days_improved"] >= (2 * s["days"] + 2) // 3
        worst_ok = s["gain_min"] > -0.5
        ok = majority and s["gain_mean"] > 0 and worst_ok
        verdict.append(f"{target}: {'PASS' if ok else 'FAIL'} "
                       f"(majority={majority} worst-day-ok={worst_ok})")
    lines.extend(verdict)
    if gate_ref is not None:
        lines.append(f"(reference ensemble mean MAPE {gate_ref:.3f}% -- improvement must beat this "
                     f"on the majority of days, not just in the mean)")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"multiday_{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / f"multiday_{tag}.json").write_text(json.dumps(
        dict(tag=tag, per_day=per_day, summary=summary), indent=2), encoding="utf-8")
    print("\n".join(lines))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("collect", "report", "both"), default="both")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("artifacts/multiday"))
    ap.add_argument("--days", nargs="+", default=DAYS)
    ap.add_argument("--train-days", type=int, default=90)
    ap.add_argument("--trees", type=int, default=180)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--recency-half-life", type=float, default=None)
    ap.add_argument("--recency-targets", nargs="+", choices=TARGETS, default=None)
    ap.add_argument("--profile-days", type=int, default=7)
    ap.add_argument("--regime-lock-min-horizon", type=int, default=0)
    ap.add_argument("--use-shared-horizon", action="store_true",
                    help="Wire SharedHorizonResidual in (frozen iter5c config sets this)")
    ap.add_argument("--shared-horizon-targets", nargs="+", choices=TARGETS,
                    default=["generator_1"])
    args = ap.parse_args()

    cache = args.cache_dir / f"records_{args.tag}.npz"
    if args.stage in ("collect", "both"):
        per_target = None
        if args.recency_targets is not None:
            per_target = {t: (args.recency_half_life if t in args.recency_targets else None)
                          for t in TARGETS}
        cfg = ModelConfig(
            trees=args.trees, leaves=args.leaves, threads=args.threads,
            train_days=args.train_days,
            recency_half_life_days=args.recency_half_life if per_target is None else None,
            recency_half_life_by_target=per_target,
            profile_days=args.profile_days,
            regime_lock_min_horizon=args.regime_lock_min_horizon,
            use_shared_horizon=args.use_shared_horizon,
            shared_horizon_targets=tuple(args.shared_horizon_targets))
        build_cache(args.tag, cfg, args.days, HORIZONS, args.cache_dir)
    if args.stage in ("report", "both"):
        report(cache, args.cache_dir, args.tag)


if __name__ == "__main__":
    main()
