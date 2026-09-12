"""Build the stage-A controlled submission zip (HANDOFF_99_PLUS.md stage A step 5).

Stage A is a single-variable experiment: ship the repaired ``input.csv`` while
holding every other delivered file byte-identical, so a change in the platform's
``out`` component cannot be mistaken for a change in the model.  The previous
package is therefore treated as the control, and this script refuses to build if
any file that must stay constant has drifted since.

The previous archive (``teamname_gas_predict_prelim.zip``) held, in order::

    s_result.csv    786a83204b87
    input.csv       4425d6030171   <- the UNREPAIRED 308-column table; replaced
    result.csv      786a83204b87
    l_result.csv    a08e589976f8
    opt_result.csv  a60d183230a1

Run ``python -X utf8 build_submission.py``; the zip lands in ``submission/``
with deterministic (fixed) entry timestamps so the archive is reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Entry order and the hashes that MUST NOT change for this to be a controlled
# experiment.  ``input.csv`` carries the repaired hash instead.
CONTROL_ENTRIES = [
    ("s_result.csv", "786a83204b87"),
    ("input.csv", None),                 # replaced by the repaired table
    ("result.csv", "786a83204b87"),
    ("l_result.csv", "a08e589976f8"),
    ("opt_result.csv", "a60d183230a1"),
]
REPAIRED_INPUT_SHA = "18596665b528"
FIXED_STAMP = (1980, 1, 1, 0, 0, 0)      # zip epoch: reproducible archives


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _zip_add(zf: zipfile.ZipFile, relpath: str, src: Path) -> None:
    info = zipfile.ZipInfo(relpath, date_time=FIXED_STAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    zf.writestr(info, src.read_bytes())


def build_stageA(args) -> int:
    problems = []
    for name, expected in CONTROL_ENTRIES:
        p = ROOT / name
        if not p.exists():
            problems.append(f"missing {name}")
            continue
        got = sha12(p)
        if name == "input.csv":
            if got != REPAIRED_INPUT_SHA:
                problems.append(
                    f"input.csv is {got}, expected the repaired {REPAIRED_INPUT_SHA} "
                    f"-- run refresh_input_quality.py first")
        elif got != expected:
            problems.append(f"{name} drifted: {got} != control {expected}")
    if problems:
        print("REFUSING TO BUILD:")
        for p in problems:
            print(f"  {p}")
        return 1

    # The repaired table must also pass the package check's quality contract.
    import subprocess
    import sys
    check = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "check_submission_package.py")],
                           cwd=ROOT, capture_output=True, text=True)
    if check.returncode != 0:
        print("REFUSING TO BUILD: check_submission_package.py reported blockers")
        print(check.stdout[-2000:])
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, _ in CONTROL_ENTRIES:
            _zip_add(z, name, ROOT / name)

    print(f"wrote {args.out.relative_to(ROOT)}")
    with zipfile.ZipFile(args.out) as z:
        for i in z.infolist():
            print(f"  {i.filename:<16} {i.file_size:>9}  sha={sha12_of(z, i.filename)}")
    print()
    print("Changed vs the previous submission: input.csv only.")
    print(f"  input.csv  {CONTROL_ENTRIES[1][1] or '4425d6030171'} -> {REPAIRED_INPUT_SHA}")
    for name, expected in CONTROL_ENTRIES:
        if name != "input.csv":
            print(f"  {name:<16} unchanged ({expected})")
    return 0


# Files that must be present at the package root in the self-contained "full" build.
FULL_ROOT_FILES = ["s_result.csv", "l_result.csv", "input.csv", "opt_result.csv",
                   "result.csv", "s_result.json", "run.sh", "run.ps1",
                   "requirements.txt", "README.md"]
FULL_PREBAKED = ["s_result.csv", "l_result.csv", "input.csv", "opt_result.csv",
                 "result.csv", "s_result.json"]


def build_full(args) -> int:
    # 1) every source must exist before we assemble anything
    missing = [f for f in FULL_ROOT_FILES if not (ROOT / f).exists()]
    for f in FULL_PREBAKED:
        if not (ROOT / "results_prebaked" / f).exists():
            missing.append(f"results_prebaked/{f}")
    sel = ROOT / "artifacts" / "development" / "selection.json"
    if not sel.exists():
        missing.append("artifacts/development/selection.json")
    code_mods = [p.name for p in ROOT.glob("*.py") if not p.name.startswith("_probe")]
    missing += [m for m in code_mods if not (ROOT / m).exists()]
    if missing:
        print("REFUSING FULL BUILD: missing sources:")
        for m in missing:
            print(f"  {m}")
        return 1

    # 2) the deliverable result files must pass the same contract the platform sees
    import subprocess
    import sys
    check = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "check_submission_package.py")],
                           cwd=ROOT, capture_output=True, text=True)
    if check.returncode != 0:
        print("REFUSING FULL BUILD: check_submission_package.py reported blockers")
        print(check.stdout[-2000:])
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FULL_ROOT_FILES:
            _zip_add(z, f, ROOT / f)
        for f in FULL_PREBAKED:
            _zip_add(z, f"results_prebaked/{f}", ROOT / "results_prebaked" / f)
        _zip_add(z, "artifacts/development/selection.json", sel)
        for m in code_mods:
            _zip_add(z, m, ROOT / m)

    print(f"wrote {args.out.relative_to(ROOT)}")
    print("Contents:")
    with zipfile.ZipFile(args.out) as z:
        entries = sorted(z.infolist(), key=lambda i: i.filename)
        for i in entries:
            print(f"  {i.filename:<40} {i.file_size:>9}  sha={sha12_of(z, i.filename)}")
    print()
    print("Self-contained package: root results + run.sh/run.ps1 + results_prebaked/ "
          "+ artifacts/development/selection.json + code modules + requirements.txt + README.md")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stageA", "full"], default="stageA",
                    help="stageA = controlled single-variable (default); "
                         "full = self-contained README-compliant package")
    ap.add_argument("--out", type=Path, default=None,
                    help="output zip path (default: submission/teamname_gas_predict_"
                         "<mode>.zip)")
    args = ap.parse_args()
    if args.out is None:
        suffix = "stageA" if args.mode == "stageA" else "full"
        args.out = ROOT / "submission" / f"teamname_gas_predict_prelim_{suffix}.zip"
    if args.mode == "stageA":
        return build_stageA(args)
    return build_full(args)


def sha12_of(zf: zipfile.ZipFile, name: str) -> str:
    return hashlib.sha256(zf.read(name)).hexdigest()[:12]


if __name__ == "__main__":
    raise SystemExit(main())
