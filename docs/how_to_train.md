# How to use the mosquito training code

Three models are trained from one config, `config/config_training.yaml`:

| Task | Model | What it learns | Output folder (in `outputs/runs/`) |
|---|---|---|---|
| `full` | YOLOv8s | whole-mosquito boxes, 14 species classes (Task A) | `yolov8s_full/` |
| `obb` | YOLOv8s-OBB | oriented body boxes, one class `body` (Task B) | `yolov8s_obb_body/` |
| `species` | Swin Transformer V2-T | species classifier (Task C): `data: crops` = one sample per box (what the pipeline feeds it), `data: images` = whole frames, multi-label | `swin_v2_t_species/` |

Use the project's uv venv (the scripts pick it automatically; see `scripts/env.sh`: `$VENV`, else the shared
`/projects/mosquitto_msis/code/.venv`, else `<project>/.venv`).

## 0. Setting up a new machine (one command)

```bash
git clone <repo> && cd mosquito_misslab_training
./run.sh setup            # checks everything, installs what is missing, runs the unit tests
./run.sh setup --check    # report only
```

`scripts/bootstrap.sh` (safe to re-run; each step is skipped when already done): system tools (git, curl, zsh, gcc, make,
unzip, OpenCV libs via apt / dnf / brew, asks for the sudo password) -> uv -> Oh My Zsh with the autosuggestions and
syntax-highlighting plugins, and optionally zsh as login shell -> git identity and ssh key -> NVIDIA driver (reported;
drivers cannot be installed automatically) -> Python 3.12, the venv and `config/requirements.txt` (torch wheels matched to the
GPU driver, CPU wheels without a GPU) -> package check -> unit tests -> training data / Telegram / old-pipeline checks.
Options: `--yes`, `--no-system`, `--no-shell`. `./run.sh check` only verifies the Python packages. Machine-specific paths
live in `config/config_training.yaml` (`paths.labelstudio_export`, `compare.old_infer_dir`, relative to this folder) and
in `config/telegram.env` (copy `config/telegram.env.example`).

## Training data (`input/datasets/`)

All trainers read one folder, `input/datasets/` (config `paths.input`). To change the data, edit the files there and start
training; nothing has to be rebuilt (crop caches refresh themselves when a file changes).

```
input/datasets/
  full_body_detection/            whole-mosquito boxes -> `full` detector AND the `species` classifier
    data.yaml                     names: {0: Culex tritaeniorhynchus, 1: Culex spp., ...}   <- the class list
    images/{train,val,test}/*.jpg|png
    labels/{train,val,test}/<same file name>.txt     one box per line: class cx cy w h  (normalised 0-1)
  body_oriented_detection/        oriented body boxes -> `obb` detector
    data.yaml                     names: {0: body}
    images/..., labels/...        one box per line: class x1 y1 x2 y2 x3 y3 x4 y4 (normalised)
```

- Add an image: put it in `images/<split>/` and its label file in `labels/<split>/` (same name, `.txt`). No label file = no objects.
- New or renamed class: edit `names` in `data.yaml`; label class ids are the position in that list.
- Keep a photo (and its copies) in one split only, or val / test scores become too optimistic.
- Images can be symlinks. Train, val and test must each contain images.
- `./run.sh` -> Dataset (or `python src/dataset.py`) prints the images and boxes per split and class and stops with a message on
  errors (wrong class id, wrong number of values in a label line, an empty split, a missing `data.yaml`).
- `python src/import_labelstudio.py` rebuilds this folder from the Label Studio export (`paths.labelstudio_export`) with the
  70/15/15 photo-grouped split (`split` in the config); it refuses to overwrite existing data unless `--force`.
- Species classifier: the class list and the box labels of `full_body_detection` are its only source. The whole-image mode
  (`data: images`) therefore uses the images of this folder, not the larger Label Studio species-classification set.

## 1. Quick start

```bash
cd /projects/mosquitto_msis/code/mosquito_misslab_training
scripts/run_training.sh                 # check input/datasets, train every task in run.tasks
```

## 2. Train one model (or a subset)

```bash
scripts/run_training.sh full            # Task A only
scripts/run_training.sh obb             # Task B only
scripts/run_training.sh species         # Task C only
scripts/run_training.sh obb species     # any subset
scripts/run_training.sh --sequential    # one after another (default: all at once, one GPU each)
scripts/run_training.sh --no-telegram   # skip the Telegram monitor
scripts/run_training.sh --help
```

Flags after `--` go to the training script(s), overriding the config for that run:

```bash
scripts/run_training.sh full -- --epochs 2 --device 0 --fraction 0.1    # smoke test
scripts/run_training.sh species -- --epochs 2 --limit 50
```

| Flag | Applies to |
|---|---|
| `--epochs --batch --device --name` | all tasks |
| `--fraction` | `full`, `obb` |
| `--limit` | `species` |

With several tasks selected, only pass flags that every selected task accepts.

Calling the scripts directly also works:

```bash
python src/dataset.py [full|obb|species]        # check / summarise the input data
python src/import_labelstudio.py [--force]     # rebuild input/datasets from a Label Studio export
python src/train_yolo.py --task full|obb [--epochs ...]
python src/train_classifier.py [--epochs ...]
```

## 3. Configuration (`config/config_training.yaml`)

- `paths`: `input` (training data), `runs`, generated YOLO yaml folder, and the Label Studio export + `splits` (importer only). Relative paths are inside this folder.
- `run.tasks`: tasks trained when none are named on the command line. `run.parallel`: run them at once or one by one.
- `split`: seed and train/val/test ratios used only by `import_labelstudio.py` (grouped by photo so a photo and its labeled copy never straddle splits).
- `full`, `obb`, `species`: per-model settings (GPU `device`, epochs, batch, image size, early-stopping patience, learning rates, augmentation).
- `telegram.interval_min`: progress message interval in minutes (0 = only start/finish/fail).

GPUs are set per task (`full: 0`, `obb: 1`, `species: cuda:2`). Check `nvidia-smi` and change `device` if one is busy.

## 3b. Learning-rate sweep (species classifier)

```bash
scripts/run_sweep.sh --dry-run          # list the planned runs
scripts/run_sweep.sh                    # train every combination of the sweep.grid in the config
scripts/run_sweep.sh --epochs 3 --limit 64    # quick smoke test
```

- The `sweep` section of the config sets the GPUs (`devices`, one job each at a time), the epochs per run (40) and the `grid` (default: `lr_head` x `lr_backbone` = 9 runs, `weight_decay` fixed at 0.05). Add more values or the other `species` keys to widen it.
- Runs go to `outputs/runs/sweep_species/<tag>/`, logs to `outputs/runs/sweep_species/<tag>.log`. Finished runs are skipped if you start it again.
- The result is `outputs/runs/sweep_species/summary.csv`, ranked by **validation** macro-AP. Pick the winner from the validation score only; the test score is shown for information. Copy the best values into the `species` section, then train as normal.

## 3c. Compare the new species classifier with the old weights

`scripts/run_training.sh` runs `src/compare_models.py` automatically once the `species` task finishes (skip with `--no-compare`;
a failed comparison never fails the training). Run it by hand any time:

```bash
python src/compare_models.py                                   # runs/<species.name>/best.pt vs the old app's classifier
python src/compare_models.py --run sweep_species/h0.0003_b3e-05_wd0.05
python src/compare_models.py --old-cls-model /path/to/epoch_100.pth   # old = .onnx or an mmpretrain .pth
```

The old model is the 4-class SwinV2-B run through the full old pipeline (`mosquitto_msislab/infer.py`: detector,
body detector, Voronoi split, classifier). The new model is the 14-species Swin V2-T on the whole image. Both are scored on
the same held-out images (`compare.split`, default `test`) on the old model's classes: Anopheles, Culex tritaeniorhynchus,
Non, Others (the new species are collapsed onto them; Unknown counts as Others). Per image a class is present if any
mosquito of it is, scored by its highest probability; an image with no detection counts as Non for the old model.
Metrics: per-class AP, precision/recall/F1 at `compare.threshold`, macro AP / F1 and exact-match accuracy.

Outputs: `<run>/compare_old.md` and `compare_old.json`, one row per comparison in `outputs/runs/compare_history.csv`, and the log
in `outputs/runs/species_compare.log`. The old model's per-image results are cached in `outputs/runs/compare_cache/` and only recomputed
when the old weights, `infer.py` or the image list change. Settings: `compare` section of the config.

## 4. Outputs

- Logs: `outputs/runs/<task>.log`; task state: `outputs/runs/<task>.status` (pending / running / ok / failed).
- YOLO runs: `outputs/runs/<name>/weights/best.pt`, training plots, `test_metrics.json` (mAP50 and mAP50-95 on the test split).
- Classifier: `outputs/runs/swin_v2_t_species/best.pt` (weights, class names, mean/std, image size, temperature) and `metrics.json` (val macro-AP, test macro-AP plain and with 4-rotation TTA).
- Import only: `outputs/splits/split.json` and `split_stats.json`, written by `import_labelstudio.py`.

The script exits non-zero if any task failed; the summary lists `OK` / `FAILED` per task.

## Folder layout

```
run.sh            terminal UI: the single entry point
src/              Python code (data, training, comparison, Telegram monitor)
scripts/          shell: bootstrap.sh (machine setup), env.sh (venv paths), run_training.sh, run_sweep.sh
config/           config_training.yaml (all settings), requirements.txt, telegram.env (private, from telegram.env.example)
tests/            pytest unit tests
docs/             how_to_train.md (this file), training_guideline.md
weights/          pretrained YOLO weights (config full.weights / obb.weights)
input/            training data: input/datasets/<task>/{images,labels}/{train,val,test} + data.yaml (edit this)
outputs/          generated: runs/ (trained models, logs), splits/, datasets/
```

## Terminal UI

`./run.sh` (the only file at the top level) opens a menu for everything below: train detector / classifier (whole images or crops), settings (device, epochs, batch, name, data, imgsz), sweep, old-vs-new comparison, full-pipeline inference, tests, status. `./run.sh test` and `./run.sh status` run those directly.

## 5. Telegram monitoring

Bot: `@msis_mosquitto_training_bot`. The token lives in `config/telegram.env` (private, never commit it).

1. One-time setup: send **`/msis2026`** to the bot. That chat is saved and is the only one the bot answers afterwards.
2. Start training as usual. You get messages when training starts, and when each task starts, finishes or fails (a failure includes the last log line).
3. Send **`/status`** any time for each task's state, its latest log line and GPU usage, or **`/compare`** for the old-vs-new classifier scores (macro AP / F1 / exact match from `compare_old.json`).
4. Periodic progress messages arrive every `telegram.interval_min` minutes.

If Telegram is unreachable, training is not affected. To change the registered chat, delete the `TELEGRAM_CHAT_ID` line in `config/telegram.env` and send `/msis2026` again from the new chat.
Details: `src/telegram_monitor.py`.

## 6. Tests

```bash
./run.sh test
```

`test_real_data.py` reads the real dataset, so it needs the data export to be present.

## 7. Troubleshooting

- *"... has no images" / "class id N is not in data.yaml"*: run `./run.sh` -> Dataset, or `python src/dataset.py`, and fix what it names.
- *Out of GPU memory / GPU busy*: lower `batch`, or change `device` for that task.
- *No Telegram messages*: confirm you sent `/msis2026`, that `config/telegram.env` has both token and chat id, and check `outputs/runs/telegram.log`.
- Background reading: `training_guideline.md`.
