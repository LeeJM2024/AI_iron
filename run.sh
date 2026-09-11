#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [[ ! -f artifacts/development/selection.json ]]; then
  python3 experiment.py
fi
python3 main.py --official --input-dir "$SCRIPT_DIR" --output-dir "$SCRIPT_DIR/output/official" "$@"
