#!/usr/bin/env bash
# Terminal UI for the mosquito detector (YOLO full / obb) and species classifier: train, sweep, compare, infer, test.
#   ./run.sh            interactive menu (arrows or j/k + Enter, digits jump, q quits)
#   ./run.sh test       run the unit tests;  ./run.sh status  show GPUs and runs;  ./run.sh -h  this help
#   ./run.sh setup      one-click machine setup (uv, zsh, Oh My Zsh, git, venv, packages, tests); add --check to only report
#   ./run.sh check      verify the Python packages
# Training data: input/datasets (see docs/how_to_train.md). Settings left empty use config/config_training.yaml; set them under "Settings" and they apply to every training run.
set -uo pipefail
cd "$(dirname "$0")"; . scripts/env.sh   # ROOT, VENV, PY (uv environment), PYTHONPATH
RUNS=outputs/runs
OLD_DIR=${OLD_DIR:-$ROOT/../mosquitto_msislab}   # old app pipeline (compare / infer)
OLD_CLS=("$OLD_DIR/models/mosquito_cls.onnx" "$OLD_DIR/models/new_models/Models_extracted/Models/best_accuracy_top1_epoch_96.pth"
         "$OLD_DIR/models/new_models/Models_extracted/Models/epoch_100.pth")
DEVICE="" EPOCHS="" BATCH="" NAME="" FRACTION="" IMGSZ="" DATA="" ARCH="" TELEGRAM=on
B=$'\e[1m' D=$'\e[2m' R=$'\e[0m' C=$'\e[36m' G=$'\e[32m' Y=$'\e[33m' RED=$'\e[31m' INV=$'\e[7m'

[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
[ -t 0 ] && [ -t 1 ] || [ $# -gt 0 ] || { echo "run.sh needs a terminal (try ./run.sh -h)"; exit 1; }
trap 'printf "\e[?25h"' EXIT

pause() { printf "\n${D}press any key...${R}"; read -rsn1 _ || true; }
ask() { local v; read -er -p "$1 [${2:-}]: " -i "${2:-}" v || v="${2:-}"; REPLY="$v"; }
yes_no() { local a; read -rsn1 -p "$1 [y/N] " a; echo; [ "$a" = y ] || [ "$a" = Y ]; }

menu() {  # menu "title" "sub" item...  -> MENU_SEL index (-1 = back / q)
  local title=$1 sub=$2; shift 2; local items=("$@") n=$# cur=0 k rest i
  printf '\e[?25l'
  while true; do
    printf '\e[2J\e[H'
    printf "${B}${C}  Mosquito training console${R}  ${D}%s${R}\n" "$title"
    printf "${D}  device=%s epochs=%s batch=%s name=%s data=%s arch=%s imgsz=%s fraction=%s telegram=%s${R}\n" \
      "${DEVICE:-cfg}" "${EPOCHS:-cfg}" "${BATCH:-cfg}" "${NAME:-cfg}" "${DATA:-cfg}" "${ARCH:-cfg}" "${IMGSZ:-cfg}" "${FRACTION:-cfg}" "$TELEGRAM"
    [ -n "$sub" ] && printf "  ${D}%s${R}\n" "$sub"; echo
    for ((i = 0; i < n; i++)); do
      if [ $i -eq $cur ]; then printf "  ${INV} %d) %s ${R}\n" $((i + 1)) "${items[i]}"; else printf "   %d) %s\n" $((i + 1)) "${items[i]}"; fi
    done
    printf "\n  ${D}↑/↓ move · Enter select · q back${R}\n"
    IFS= read -rsn1 k || { MENU_SEL=-1; break; }
    case "$k" in
      $'\e') read -rsn2 -t 0.01 rest || true
        case "$rest" in '[A') ((cur = (cur + n - 1) % n)) ;; '[B') ((cur = (cur + 1) % n)) ;; esac ;;
      k) ((cur = (cur + n - 1) % n)) ;; j) ((cur = (cur + 1) % n)) ;;
      "") MENU_SEL=$cur; break ;;
      q|Q) MENU_SEL=-1; break ;;
      [1-9]) [ "$k" -le "$n" ] && { MENU_SEL=$((k - 1)); break; } ;;
    esac
  done
  printf '\e[?25h'
}

run() {  # run <cmd...>: show, confirm, execute
  printf '\e[2J\e[H'; printf "${B}\$ %s${R}\n\n" "$*"
  yes_no "Run this?" || return 0
  "$@"; local rc=$?
  [ $rc -eq 0 ] && printf "\n${G}done${R}\n" || printf "\n${RED}exit code %d${R}\n" $rc
  pause
}

common_flags() { FL=(); [ -n "$DEVICE" ] && FL+=(--device "$DEVICE"); [ -n "$EPOCHS" ] && FL+=(--epochs "$EPOCHS")
  [ -n "$BATCH" ] && FL+=(--batch "$BATCH"); [ -n "$NAME" ] && FL+=(--name "$NAME"); return 0; }
train() {  # train <task...>; classifier-only flags are added when species is among them
  common_flags
  [ -n "$FRACTION" ] && [[ " $* " != *" species "* ]] && FL+=(--fraction "$FRACTION")
  if [[ " $* " == *" species "* ]]; then [ -n "$DATA" ] && FL+=(--data "$DATA"); [ -n "$IMGSZ" ] && FL+=(--imgsz "$IMGSZ"); [ -n "$ARCH" ] && FL+=(--arch "$ARCH"); fi
  local opts=(); [ "$TELEGRAM" = off ] && opts+=(--no-telegram)
  run scripts/run_training.sh "$@" "${opts[@]}" -- "${FL[@]}"
}

train_menu() {
  while menu "Train" "Logs go to outputs/runs/<task>.log; Telegram gets start/finish messages and the old-vs-new comparison" \
      "Detector: whole-mosquito boxes (YOLOv8s, 14 species)" "Detector: oriented body boxes (YOLOv8s-obb)" \
      "Classifier: whole images (Swin V2-T, multi-label)" "Classifier: per-mosquito crops (the pipeline input)" \
      "All three in parallel (tasks from config run.tasks)" "Back"; [ "$MENU_SEL" -ge 0 ] && [ "$MENU_SEL" -ne 5 ]; do
    case $MENU_SEL in
      0) train full ;; 1) train obb ;;
      2) local d=$DATA; DATA=images; train species; DATA=$d ;;
      3) local d=$DATA z=$IMGSZ; DATA=crops; IMGSZ=${IMGSZ:-384}; NAME=${NAME:-swin_v2_t_crops}; train species; DATA=$d IMGSZ=$z NAME="" ;;
      4) train ;;
    esac
  done
}

settings_menu() {
  local labels=(device epochs batch name fraction data imgsz arch) vars=(DEVICE EPOCHS BATCH NAME FRACTION DATA IMGSZ ARCH)
  local hints=("GPU, e.g. cuda:3 or 0" "training epochs" "batch size" "run name (output folder under runs/)"
               "YOLO only: share of the train set, 0-1" "classifier data: images | crops" "classifier input size, e.g. 384" "classifier model: swin_v2_t | swin_v2_s | swin_v2_b")
  while true; do
    local it=() i
    for i in "${!vars[@]}"; do it+=("$(printf '%-9s = %-12s %s' "${labels[i]}" "${!vars[i]:-(config)}" "(${hints[i]})")"); done
    it+=("telegram notifications: $TELEGRAM" "Reset all to config defaults" "Edit config/config_training.yaml" "Back")
    menu "Settings" "Empty = value from config/config_training.yaml" "${it[@]}"
    local s=$MENU_SEL; [ "$s" -lt 0 ] && return
    if [ "$s" -lt ${#vars[@]} ]; then ask "${labels[s]}" "${!vars[s]}"; printf -v "${vars[s]}" '%s' "$REPLY"
    elif [ "$s" -eq 8 ]; then [ "$TELEGRAM" = on ] && TELEGRAM=off || TELEGRAM=on
    elif [ "$s" -eq 9 ]; then DEVICE="" EPOCHS="" BATCH="" NAME="" FRACTION="" DATA="" IMGSZ="" ARCH=""
    elif [ "$s" -eq 10 ]; then "${EDITOR:-nano}" config/config_training.yaml
    else return; fi
  done
}

runs_with_best() { for f in $RUNS/*/best.pt; do [ -f "$f" ] && basename "$(dirname "$f")"; done; }
pick_run() {  # pick_run "title" -> PICK = run name
  local r=(); while read -r x; do r+=("$x"); done < <(runs_with_best)
  [ ${#r[@]} -eq 0 ] && { echo "no trained classifier in runs/"; pause; PICK=""; return; }
  menu "$1" "Trained classifiers with a best.pt" "${r[@]}"; [ "$MENU_SEL" -ge 0 ] && PICK=${r[MENU_SEL]} || PICK=""
}

compare_menu() {
  pick_run "Compare: which new classifier?"; [ -n "$PICK" ] || return
  local dev=${DEVICE:-$($PY -c "import common;print(common.CFG['species']['device'])")}
  run $PY src/compare_models.py --run "$PICK" --device "$dev" --old-cls-model "${OLD_CLS[@]}" &&
    { [ "$TELEGRAM" = on ] && yes_no "Send the result to Telegram?" && $PY src/telegram_monitor.py compare "$PICK"; }
  return 0
}

infer_menu() {
  pick_run "Pipeline inference: which classifier?"; [ -n "$PICK" ] || return
  ask "input image or folder" "$OLD_DIR/inference_images/samples"; local in=$REPLY
  ask "output folder" "/tmp/mosquito_infer_$PICK"; local out=$REPLY
  ask "GPU index" "${DEVICE#cuda:}"; local dev=${REPLY:-3}
  local ROOT=$PWD; (cd "$OLD_DIR" && run $PY infer.py --input "$in" --output "$out" --cls-model "$ROOT/$RUNS/$PICK/best.pt" --device "$dev")
}

test_menu() {
  while menu "Tests" "pytest in tests/" "All unit tests" "Fast fail (-x, quiet)" "Real-data tests only" "Pick by keyword (-k)" "Back"; [ "$MENU_SEL" -ge 0 ] && [ "$MENU_SEL" -ne 4 ]; do
    case $MENU_SEL in
      0) run $PY -m pytest tests -q ;; 1) run $PY -m pytest tests -x -q ;;
      2) run $PY -m pytest tests/test_real_data.py -q ;;
      3) ask "keyword" ""; run $PY -m pytest tests -q -k "$REPLY" ;;
    esac
  done
}

setup_env() { scripts/bootstrap.sh "$@"; }
check_env() { $PY src/check_env.py; }
env_menu() {
  menu "Environment" "uv venv: $VENV" "One-click setup: uv, zsh/Oh My Zsh, git, venv, packages, tests" "Report only (change nothing)" "Check Python packages and CUDA" "Back"
  case $MENU_SEL in 0) run setup_env ;; 1) run setup_env --check ;; 2) run check_env ;; esac
}

dataset_menu() {
  menu "Dataset" "Training data lives in $(grep -m1 'input:' config/config_training.yaml | awk '{print $2}') (edit the files there, then train)" \
    "Check: counts per split and class, label errors" "Import from the Label Studio export (needs --force if data exists)" "Back"
  case $MENU_SEL in 0) run $PY src/dataset.py ;; 1) run $PY src/import_labelstudio.py ;; esac
}

status() {
  printf '\e[2J\e[H'; printf "${B}GPUs${R}\n"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  nvidia-smi not available"
  printf "\n${B}Task status${R}\n"; for t in full obb species; do printf "  %-8s %s\n" "$t" "$(cat $RUNS/$t.status 2>/dev/null || echo -)"; done
  printf "\n${B}Last log lines${R}\n"; for t in full obb species; do [ -f $RUNS/$t.log ] && { printf "  ${Y}%s${R}: " "$t"; tail -c 300 $RUNS/$t.log | tr '\r' '\n' | tail -1; }; done
  printf "\n${B}Trained classifiers${R}\n"
  for r in $(runs_with_best); do printf "  %-26s %s\n" "$r" "$($PY -c "import json;m=json.load(open('$RUNS/$r/metrics.json'));print('val',round(m['val_macro_ap'],3),'test',round(m['test_macro_ap_plain'],3))" 2>/dev/null)"; done
  printf "\n${B}YOLO runs${R}\n"; ls -d $RUNS/yolov8s_* 2>/dev/null | sed 's/^/  /'
}

main_menu() {
  while menu "Main menu" "Detector + classifier training for the mosquito pipeline" \
      "Train" "Settings (device, epochs, batch, ...)" "Hyper-parameter sweep (classifier)" "Compare new vs old classifiers" \
      "Run the full pipeline on images (infer.py)" "Tests" "Status (GPUs, runs, logs)" "Dataset: check counts / import from Label Studio" \
      "Telegram: send latest comparison" "Setup environment (uv, packages)" "Quit (q)"; [ "$MENU_SEL" -ge 0 ] && [ "$MENU_SEL" -ne 10 ]; do
    case $MENU_SEL in
      0) train_menu ;; 1) settings_menu ;;
      2) menu "Sweep" "Grid from config sweep:" "Dry run (list planned runs)" "Run the sweep" "Back"
         case $MENU_SEL in 0) run scripts/run_sweep.sh --dry-run ;; 1) run scripts/run_sweep.sh ;; esac ;;
      3) compare_menu ;; 4) infer_menu ;; 5) test_menu ;; 6) status; pause ;;
      7) dataset_menu ;; 8) run $PY src/telegram_monitor.py compare ;; 9) env_menu ;;
    esac
  done
  printf '\e[2J\e[H'
}

case "${1:-}" in
  test) exec $PY -m pytest tests -q ;;
  setup) shift; setup_env "$@" ;;
  check) check_env ;;
  status) status ;;
  "") if [ ! -x "$PY" ]; then echo "Python environment not found at $VENV."; yes_no "Run the one-click setup now?" && { setup_env; pause; }; fi
     main_menu ;;
  *) echo "unknown command '$1' (try ./run.sh -h)"; exit 1 ;;
esac
