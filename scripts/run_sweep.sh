#!/usr/bin/env bash
# Learning-rate / weight-decay sweep for the Swin V2-T species classifier (config `sweep` section).
#   scripts/run_sweep.sh                 run every grid combination; finished ones are skipped on re-run
#   scripts/run_sweep.sh --dry-run       list the planned runs
#   scripts/run_sweep.sh --epochs 3 --limit 64    smoke test
# Results: runs/sweep_species/summary.csv, ranked by validation macro-AP.
set -uo pipefail
cd "$(dirname "$0")/.."; . scripts/env.sh
$PY src/dataset.py species || exit 1
exec $PY src/sweep_species.py "$@"
