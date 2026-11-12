#!/usr/bin/env bash
# Train models listed under run.tasks in config/config_training.yaml, or only the ones you name.
#   scripts/run_training.sh                    all tasks from the config
#   scripts/run_training.sh full               only Task A (YOLOv8 full-body HBB)
#   scripts/run_training.sh obb species        several chosen tasks
#   scripts/run_training.sh full -- --epochs 2 --device 0     flags after "--" go to the training script(s)
#   scripts/run_training.sh --no-telegram      skip the Telegram monitor (see telegram_monitor.py)
#   scripts/run_training.sh --no-compare       skip the new-vs-old classifier comparison after the species task (compare_models.py)
#   scripts/run_training.sh --sequential       run the chosen tasks one after another instead of in parallel
# Tasks: full | obb | species. Checks the input data first; logs go to <runs>/<task>.log.
set -uo pipefail
cd "$(dirname "$0")/.."; . scripts/env.sh

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; }

CHOSEN=(); EXTRA=(); SEQ=""; NOTIFY=true; COMPARE=true
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --sequential) SEQ=true ;;
    --parallel) SEQ=false ;;
    --no-telegram) NOTIFY=false ;;
    --no-compare) COMPARE=false ;;
    --) shift; EXTRA=("$@"); break ;;
    full|obb|species) CHOSEN+=("$1") ;;
    *) echo "unknown task or option '$1' (tasks: full, obb, species)"; usage; exit 1 ;;
  esac
  shift
done

read -r RUNS PARALLEL TASKS < <($PY -c "
import common
r = common.CFG['run']
print(common.RUNS, str(r.get('parallel', True)).lower(), ' '.join(r['tasks']))") || exit 1
[ ${#CHOSEN[@]} -gt 0 ] && TASKS="${CHOSEN[*]}"
[ -n "$SEQ" ] && PARALLEL=$([ "$SEQ" = true ] && echo false || echo true)
[ -n "${TASKS:-}" ] || { echo "config: run.tasks is empty"; exit 1; }
mkdir -p "$RUNS"
$PY src/dataset.py $TASKS || exit 1   # validates input/datasets and prints the image / box counts

CMP_ARGS=()
for ((i = 0; i < ${#EXTRA[@]}; i++)); do   # a --name / --device override for the species run applies to the comparison too
  case "${EXTRA[i]}" in --name) CMP_ARGS+=(--run "${EXTRA[i+1]}") ;; --device) CMP_ARGS+=(--device "${EXTRA[i+1]}") ;; esac
done
rc=0
for t in $TASKS; do echo pending > "$RUNS/$t.status"; done
MON=""
if [ "$NOTIFY" = "true" ] && [ -f config/telegram.env ]; then
  $PY src/telegram_monitor.py watch $TASKS > "$RUNS/telegram.log" 2>&1 &
  MON=$!
fi
compare_species() {  # new species classifier vs the old weights; never fails the training run
  echo "compare species -> $RUNS/species_compare.log"
  $PY src/compare_models.py "${CMP_ARGS[@]}" > "$RUNS/species_compare.log" 2>&1 || echo "compare FAILED (see $RUNS/species_compare.log)"
  [ "$NOTIFY" = "true" ] && $PY src/telegram_monitor.py compare "${CMP_ARGS[1]:-$($PY -c "import common;print(common.CFG['species']['name'])")}" || true
}
report() {
  if wait "$2"; then echo "OK     $1"; echo ok > "$RUNS/$1.status"
    [ "$1" = species ] && [ "$COMPARE" = true ] && compare_species
  else echo "FAILED $1 (see $RUNS/$1.log)"; echo failed > "$RUNS/$1.status"; rc=1; fi
}
jobs_started=()
for t in $TASKS; do
  case "$t" in
    full|obb) cmd=(src/train_yolo.py --task "$t") ;;
    species)  cmd=(src/train_classifier.py) ;;
    *) echo "config: unknown task '$t' (use full, obb, species)"; exit 1 ;;
  esac
  echo "start  $t -> $RUNS/$t.log"; echo running > "$RUNS/$t.status"
  $PY "${cmd[@]}" "${EXTRA[@]}" > "$RUNS/$t.log" 2>&1 &
  if [ "$PARALLEL" = "true" ]; then jobs_started+=("$t:$!"); else report "$t" "$!"; fi
done
for j in "${jobs_started[@]:-}"; do [ -n "$j" ] && report "${j%%:*}" "${j#*:}"; done
[ -n "$MON" ] && wait "$MON"
exit $rc
