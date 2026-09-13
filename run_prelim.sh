#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -lt 2 ]; then
    echo 'Usage: bash run_prelim.sh TRAIN_DIRECTORY TEST_DIRECTORY [OUTPUT_DIRECTORY]' >&2
    exit 2
fi
python -X utf8 -W ignore::DeprecationWarning run_prelim.py \
  --train-dir "$1" --test-dir "$2" --output-dir "${3:-output/prelim_v2}"
