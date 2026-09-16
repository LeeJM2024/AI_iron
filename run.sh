#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
    py=python3
else
    py=python
fi
"$py" -X utf8 current_release.py --output-dir "${1:-output/current}"
