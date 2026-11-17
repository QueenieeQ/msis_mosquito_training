# Mosquito species classifier: training pipeline, weights and comparison with the old model

Knowledge base for building slides. Everything below was read from the project files and result files on 2026-11-12;
numbers are copied from `outputs/runs/*/metrics.json`, `outputs/runs/*/compare_old.json` and `outputs/runs/compare_history.csv`.
Project folder: `/projects/mosquitto_msis/code/mosquito_misslab_training`.

---

## 1. One-paragraph summary

We replaced the 4-class SwinV2-B classifier of the delivered mosquito app with a 14-species Swin Transformer V2 classifier
trained on our own labelled photos. Two ways of using it were tried. (a) On the **whole photo** it is far better than the
old model: macro AP 0.91 against 0.57 (old ONNX) and 0.77 (old best .pth). (b) Inside the **real detector pipeline**
(detector -> body detector -> crop -> classifier) the best new model, trained on per-mosquito crops, reaches macro AP 0.81
against 0.57 / 0.77 / 0.75: better than all three old models, but the gain is smaller and depends on the detector boxes.
Making the model bigger (Small, Base) or using stronger augmentation did not help. The detectors were **not** retrained:
the old detector weights are still used.

## 2. Headline numbers (for a results slide)

Test set: 142 held-out photos, scored on the old model's 4 classes (Anopheles / Culex tritaeniorhynchus / Non / Others), threshold 0.5.

| model | how it is used | macro AP | macro F1 | exact match | Anopheles AP | Culex trit. AP |
|---|---|---:|---:|---:|---:|---:|
| old `mosquito_cls.onnx` (delivered app) | detector pipeline | 0.573 | 0.269 | 0.204 | 0.565 | 0.267 |
| old `best_accuracy_top1_epoch_96.pth` | detector pipeline | 0.773 | 0.630 | 0.408 | 0.775 | 0.658 |
| old `epoch_100.pth` | detector pipeline | 0.748 | 0.623 | 0.423 | 0.777 | 0.578 |
| **new Swin V2-T, whole photo** (`swin_v2_t_focus_a`) | classifier on the full image | **0.914** | 0.845 | 0.732 | 0.905 | 0.891 |
| **new Swin V2-T, trained on crops** (`swin_v2_t_crops_a`) | detector pipeline | **0.808** | 0.758 | 0.507 | 0.783 | 0.801 |

Key messages:
- Whole-photo classifier: +0.34 macro AP over the delivered ONNX, +0.14 over the best old `.pth`.
- Detector-pipeline classifier: +0.24 over the delivered ONNX, +0.04 over the best old `.pth`; F1 0.76 against 0.27 / 0.63.
- Crop-level test (ground-truth boxes, 14 species): macro AP 0.944; Culex spp. 0.905, Anopheles spp. 0.926.
- A bigger model (Small 0.712, Base 0.803 in the pipeline) did not beat Tiny (0.808).

## 3. System overview

### 3.1 The inference pipeline (`mosquitto_msislab/infer.py`, the delivered app logic)

```
photo (9152 x 6944 px)
  -> mosquito detector   YOLOv8s, input 1520 px, conf 0.25, IoU 0.5      (models/mosquito_det.pt, 22.8 MB)
  -> merge overlapping boxes (threshold 0.4)
  -> body detector       YOLOv8n, input 512 px, inside each box          (models/mosquito_body_det.pt, 6.2 MB)
  -> merge bodies per box, then on the whole photo (threshold 0.6)
  -> Voronoi split when one box holds several bodies (other bodies painted grey)
  -> classifier          resize crop to a square, normalise, top-2 with a class threshold
```

Old classifier: SwinV2-Base, 384 px, 4 classes (`cls_labels.txt`: 얼룩날개모기 = Anopheles, 작은빨간집모기 = *Culex
tritaeniorhynchus*, 비모기 = Non-mosquito, 그 외 모기 = Other mosquitoes), classification threshold 0.1.
Note: 작은빨간집모기 is *Culex tritaeniorhynchus*; it is **not** *Culex pipiens* (빨간집모기).

### 3.2 What we changed

Only the **classifier** stage. We trained new classifiers (Swin V2 tiny / small / base from torchvision, ImageNet-1K
pre-trained) with 14 species and plugged them into the same pipeline through a new loader in
`mosquitto_msislab/cls_backend.py` (`load_species`). The two detectors are the original weights; the training code for
detectors exists (`src/train_yolo.py`, YOLOv8s HBB and YOLOv8s-OBB) but no detector was retrained in this work.

## 4. Data

### 4.1 Source
- Photos of sticky-trap strips taken with a microscope camera, 9152 x 6944 px, from field sites (Ansan reed wetland,
  Gaepo park, Poseung barn, Seosan Birdland, Jinju barn and others), labelled in Label Studio by 4 annotators.
- Each photo exists as an original (원본) and a labelled copy (라벨링); the two form one group and always share a split.
- Reference folders of single species (Ae. albopictus, Culex pipiens, Och. koreicus, Och. togoi) exist in
  `data/2nd_data` but are **not** used for training. There is therefore **no Culex pipiens class**; `Culex spp.` is the
  closest and is used as its proxy.

### 4.2 Training folder (`input/datasets/`, config `paths.input`)
```
input/datasets/
  full_body_detection/       whole-mosquito boxes, 14 species   -> full detector AND species classifier
    data.yaml (class names), images/{train,val,test}, labels/{train,val,test}  (YOLO txt: class cx cy w h)
  body_oriented_detection/   oriented body boxes, 1 class "body" -> OBB detector
```
Edit the files, run `./run.sh`, train. A checker (`python src/dataset.py`) validates labels and prints counts.

### 4.3 Size and split (photo-grouped 70 / 15 / 15, stratified by species and annotator, seed 0)

| set | train | val | test |
|---|---:|---:|---:|
| full_body_detection: images | 687 | 146 | 142 |
| full_body_detection: boxes (= crop samples) | 2815 | 535 | 561 |
| body_oriented_detection: images | 513 | 105 | 102 |
| body_oriented_detection: boxes | 1996 | 379 | 437 |

Boxes per species (train / val / test):

| species | train | val | test |
|---|---:|---:|---:|
| No Mosq. (non-mosquito) | 702 | 119 | 113 |
| Culex spp. | 575 | 123 | 125 |
| Ae. vexans nipponii | 244 | 35 | 42 |
| Anopheles spp. | 206 | 39 | 51 |
| Culex tritaeniorhynchus | 202 | 43 | 63 |
| Ae. albopictus | 175 | 25 | 40 |
| Och. togoi | 164 | 41 | 13 |
| Ar. subalbatus | 145 | 38 | 23 |
| Och. koreicus | 140 | 30 | 41 |
| Unknown | 126 | 12 | 23 |
| Mansonia uniformis | 71 | 18 | 22 |
| Culex orientalis | 30 | 8 | 1 |
| Mansonia ochracea | 27 | 2 | 3 |
| Ae. dorsalis | 8 | 2 | 1 |

The data are imbalanced (rare classes have under 30 training boxes); handled by class-weighted focal loss (4.5).

### 4.4 Two ways of building classifier samples
- **Whole image** (`species.data: images`): the full photo letterboxed to 640 px, multi-label (one output per species,
  1 if the species appears anywhere in the photo). Used by the first two runs.
- **Crops** (`species.data: crops`): one sample per ground-truth box: the box plus 5 % margin, neighbours' pixels
  replaced by grey with the same nearest-centre rule as the pipeline's Voronoi step, resized to a square 384 px exactly like
  `infer.py`. This matches what the classifier sees at deployment. 2815 / 535 / 561 crops. Crops are cached and rebuilt
  automatically when a file changes.

## 5. Training recipe (Swin V2 classifier, `src/train_classifier.py`)

| item | value |
|---|---|
| backbone | torchvision Swin V2, ImageNet-1K pre-trained; `species.arch`: `swin_v2_t` 28 M / `swin_v2_s` 50 M / `swin_v2_b` 88 M params |
| head | new linear layer, 14 outputs, sigmoid (multi-label) |
| loss | focal loss (gamma 2) with effective-number class weights (beta 0.95) |
| optimiser | AdamW, lr 3e-4 (head) and 3e-4 (backbone), weight decay 0.05 |
| schedule | 5 warm-up epochs, then cosine decay to 1e-6; up to 60 to 80 epochs |
| precision / hardware | bf16 autocast on an RTX 4090 24 GB (7 GPUs on the server) |
| batch / input | 16 at 640 px (whole image); 32 at 384 px (crops) |
| model selection | best epoch by the mean validation AP of the two focus classes (**Culex spp.** and **Anopheles spp.**); early stop after 15 epochs without gain |
| calibration | temperature scaling fitted on the validation set |
| test-time augmentation | optional average over 4 rotations (0 / 90 / 180 / 270 degrees) |
| checkpoint `best.pt` | weights, class names, mean/std, input size, temperature, architecture, input type (image or crop) |

Approximate speed: Tiny 12 to 20 s per epoch, Small about 30 s, Base about 70 s (crops, one 4090).

### 5.1 Augmentation
- **Original recipe:** random 90-degree rotation, flips, random scale 0.8 to 1.2 and translate 0.1, mild HSV jitter.
- **Onsite-studio recipe** (ported from `example_git/.../onsite_training_studio`, `src/augment.py`): every effect fires
  independently per sample with its own probability: brightness gain 0.75 to 1.25 and bias plus or minus 18 (p 0.55), Gaussian noise sigma 3 to
  18 (p 0.45), Gaussian or median blur kernel 3/5/7 (p 0.35), translate 10 % (p 0.55), scale 0.85 to 1.20 (p 0.55),
  rotate plus or minus 12 degrees (p 0.45), shear (p 0.30). All geometric effects are merged into one affine warp; order is
  geometry, brightness, noise, blur. All values are in `species.augment` of the config. About 10 ms per sample.

## 6. Experiment log

All runs: seed 0, same split. "Crop-test" = ground-truth crops of the test split (561 crops), 14 species.
"Pipeline" = the real detector pipeline on 142 test photos, 4 classes.

| # | run | data / input | arch | augmentation | epochs run | best epoch | val macro AP | crop-test macro AP (TTA) | Culex spp. AP | Anopheles spp. AP | pipeline macro AP |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `swin_v2_t_species` | whole image 640 | tiny | original | 62 (early stop) | 47 | 0.899 | 0.883 (image-level) | n/a | n/a | n/a (whole-photo 0.918) |
| 2 | `swin_v2_t_focus_a` | whole image 640, focus-class selection | tiny | original | 52 (early stop) | 37 | 0.879 | 0.819 (image-level) | 0.932 | 0.912 | whole-photo 0.914 |
| 3 | `swin_v2_t_crops_a` | crops 384 | tiny | original | 60 (all, no early stop) | not recorded | 0.924 | 0.944 | 0.905 | 0.926 | **0.808** |
| 4 | `swin_v2_t_crops_aug` | crops 384 | tiny | onsite-studio | 52 (early stop) | 37 | 0.918 | 0.939 | 0.885 | **0.977** | 0.760 |
| 5 | `swin_v2_s_crops` | crops 384 | small | original | 21 (early stop) | 6 | 0.852 | 0.810 | 0.848 | 0.922 | 0.712 |
| 6 | `swin_v2_b_crops` | crops 384 | base | original | 42 (early stop) | 27 | 0.908 | 0.894 | 0.897 | 0.919 | 0.803 |

(Runs 1 and 2 are image-level metrics on whole photos; their test AP is over all 14 species. Run 2's per-class numbers are
for the two focus classes. Run 3 ran all 60 epochs without an early stop; its log was overwritten, so its best epoch is not recorded.)

### 6.1 Learning-rate sweep (whole-image Tiny, 40 epochs, 6 combinations; `outputs/runs/sweep_species/summary.csv`)

| lr head | lr backbone | val macro AP | test macro AP (plain) |
|---|---|---:|---:|
| 3e-4 | 3e-4 | **0.898** | 0.878 |
| 1e-4 | 3e-4 | 0.894 | 0.840 |
| 1e-4 | 1e-4 | 0.890 | 0.876 |
| 3e-4 | 1e-4 | 0.886 | 0.876 |
| 3e-4 | 3e-5 | 0.860 | 0.866 |
| 1e-4 | 3e-5 | 0.857 | 0.857 |

The winner (3e-4 / 3e-4) is the config default.

### 6.2 Findings
1. **Training on crops matters for the pipeline.** The whole-image model, plugged into the pipeline on crops, was unreliable
   (it was never trained on single-mosquito images; it over-predicted Culex spp. and Mansonia ochracea). The crop-trained model
   assigned the right species in most test frames, including 3 Anopheles in the one frame that contained them.
2. **Bigger is not better here.** Small and Base did not beat Tiny; with 2815 training crops the limit is the data.
3. **Longer training will not help:** Tiny reached near-zero training loss (about 0.0005) while validation AP was flat; the
   cosine schedule just stretches. 200 to 300 epochs is not recommended.
4. **New augmentation: mixed.** Anopheles crop-test AP rose (0.926 to 0.977), Culex fell (0.905 to 0.885), and the pipeline
   macro AP dropped (0.808 to 0.760), mostly because predictions became less confident at the 0.5 cut-off. Single runs on a
   small test set: differences of a few points can be noise.
5. **Gap between crop-test (0.94) and pipeline (0.81):** the pipeline uses detector boxes and a fixed threshold, the crop
   test uses ground-truth boxes. Training on the detector's own boxes is the most promising next step.

## 7. Comparison with the old model in detail

### 7.1 Protocol
- 142 held-out test photos (never used for training or model selection).
- The new 14 species are collapsed onto the old 4 classes: Anopheles spp. -> Anopheles; Culex tritaeniorhynchus ->
  Culex tritaeniorhynchus; No Mosq. -> Non; everything else (including Culex spp., Unknown) -> Others.
- Per image, a class is present if any detected mosquito has it; its score is the maximum probability. Images with no
  detection count as Non. Metrics: average precision per class (AP, threshold free), precision / recall / F1 at 0.5,
  macro averages over the 4 classes, exact match (all 4 labels right for one photo).
- Image counts per class in the test set: Anopheles 30, Culex tritaeniorhynchus 26, Non 47, Others 115.
- Old models are scored with the complete old pipeline; new crop models with the same pipeline and the same detectors.

### 7.2 Full table (pipeline unless noted)

| model | macro AP | macro F1 | exact | AP Anoph. | AP Culex trit. | AP Non | AP Others | F1 Anoph. | F1 Culex trit. | F1 Non | F1 Others |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old ONNX | 0.573 | 0.269 | 0.204 | 0.565 | 0.267 | 0.543 | 0.918 | 0.062 | 0.393 | 0.080 | 0.541 |
| old `.pth` epoch 96 | 0.773 | 0.630 | 0.408 | 0.775 | 0.658 | 0.714 | 0.946 | 0.677 | 0.505 | 0.469 | 0.868 |
| old `.pth` epoch 100 | 0.748 | 0.623 | 0.423 | 0.777 | 0.578 | 0.679 | 0.959 | 0.691 | 0.500 | 0.419 | 0.883 |
| new Tiny whole photo (focus_a, plain) | 0.914 | 0.845 | 0.732 | 0.905 | 0.891 | 0.869 | 0.991 | 0.800 | 0.816 | 0.817 | 0.948 |
| new Tiny whole photo (focus_a, TTA) | 0.916 | 0.847 | 0.725 | 0.912 | 0.891 | 0.872 | 0.991 | 0.800 | 0.816 | 0.826 | 0.944 |
| new Tiny crops (`crops_a`) | 0.808 | 0.758 | 0.507 | 0.783 | 0.801 | 0.690 | 0.959 | 0.750 | 0.737 | 0.672 | 0.874 |
| new Tiny crops + new augmentation | 0.760 | 0.657 | 0.465 | 0.763 | 0.665 | 0.660 | 0.953 | 0.545 | 0.558 | 0.650 | 0.873 |
| new Small crops | 0.712 | 0.482 | 0.317 | 0.755 | 0.578 | 0.594 | 0.919 | 0.278 | 0.235 | 0.578 | 0.837 |
| new Base crops | 0.803 | 0.673 | 0.507 | 0.746 | 0.781 | 0.712 | 0.972 | 0.571 | 0.558 | 0.678 | 0.883 |

### 7.3 Caveats to state on slides
- The comparison favours the new models: they were trained on the same photos distribution as the test set, the old
  models were trained elsewhere (training data of the old models is not known to us).
- The old model only knows 4 classes; **Culex pipiens / Culex spp. cannot be compared** with it.
  The focus-class results for Culex spp. exist only for the new models.
- 142 test photos, one seed per run: small differences are within noise. Culex orientalis, Ae. dorsalis and Mansonia ochracea have
  1 to 3 test boxes and cannot be judged.
- Whole-photo results do not carry over to the detector pipeline (section 6.2).

## 8. Weights and files

### 8.1 New classifiers (`outputs/runs/<run>/best.pt`; git-ignored, binary)

| run | size | architecture | input | classes | note |
|---|---:|---|---|---:|---|
| `swin_v2_t_crops_a` | 111 MB | swin_v2_t | 384 px crop | 14 | **recommended for the pipeline** (pipeline macro AP 0.808) |
| `swin_v2_t_focus_a` | 111 MB | swin_v2_t | 640 px whole photo | 14 | best whole-photo model (macro AP 0.914) |
| `swin_v2_t_species` | 111 MB | swin_v2_t | 640 px whole photo | 14 | first baseline |
| `swin_v2_t_crops_aug` | 111 MB | swin_v2_t | 384 px crop | 14 | new augmentation, best Anopheles AP |
| `swin_v2_s_crops` | 197 MB | swin_v2_s | 384 px crop | 14 | not recommended |
| `swin_v2_b_crops` | 349 MB | swin_v2_b | 384 px crop | 14 | not better than tiny |

Each folder also holds `metrics.json` and, after comparison, `compare_old.json` / `compare_old.md`.
Run the pipeline with a new model: `infer.py --input <dir> --cls-model <run>/best.pt --device 3`.

### 8.2 Old / delivered weights (`mosquitto_msislab/models/`)
`mosquito_det.pt` (22.8 MB, YOLOv8s mosquito detector), `mosquito_body_det.pt` (6.2 MB, YOLOv8n body detector),
`mosquito_cls.onnx` (351 MB, delivered classifier), `cls_labels.txt`; in `new_models/Models_extracted/Models/`:
`best_accuracy_top1_epoch_96.pth` and `epoch_100.pth` (1.1 GB each, SwinV2-B mmpretrain checkpoints, same 4 classes),
`Mosquito_Body.pt`, `Mosquito_Classification.pt`, `Mosquito_total.pt` (YOLO detectors, byte-identical to the detectors
already in `models/`). Pre-trained starting weights for detector training: `weights/yolov8s.pt`, `weights/yolov8s-obb.pt`.

### 8.3 Other result files
`outputs/runs/compare_history.csv` (one row per comparison), `outputs/runs/sweep_species/summary.csv`,
`outputs/runs/<name>.log` (training curves per epoch).

## 9. Code and tooling

```
run.sh            terminal UI: train, settings, sweep, compare, pipeline inference, tests, dataset check, setup
src/              dataset.py, cls_data.py, crop_data.py, augment.py, train_classifier.py, train_yolo.py, yolo_dataset.py,
                  compare_models.py, compare_old.py, compare_classes.py, sweep_species.py, telegram_monitor.py,
                  import_labelstudio.py, make_splits.py, check_env.py, common.py, cls_metrics.py
scripts/          bootstrap.sh (one-click machine setup), env.sh, run_training.sh, run_sweep.sh
config/           config_training.yaml (every setting), requirements.txt, telegram.env.example
tests/            67 pytest tests (all pass)
docs/             how_to_train.md, training_guideline.md, this report
```
- One-click setup on a new machine: `./run.sh setup` (system tools, uv, Oh My Zsh, git, Python 3.12 venv, packages, tests).
- Training: `./run.sh` menu, or `scripts/run_training.sh species -- --data crops --imgsz 384 --epochs 60`.
- After each training the comparison with the old models runs and the result is sent to Telegram (`/compare` shows the latest).
- Environment: Python 3.12, torch 2.14 (CUDA 12.6 wheels), torchvision 0.29, ultralytics 8.4, mmpretrain 1.2, onnxruntime-gpu 1.23.

## 10. Limits and next steps

- Detectors are unchanged; any missed or merged mosquito in stage 1 or 2 limits the classifier.
- No Culex pipiens class: add labelled pipiens photos (the reference folder has 22) to get a real pipiens result.
- Train the classifier on the detector's own boxes (matched to ground truth by IoU) to close the crop-test / pipeline gap.
- More labelled Culex and Anopheles crops and rare species; average several seeds before drawing conclusions on small differences.
- Retrain and compare the two detectors (training code is ready, `./run.sh` -> Train).
- Possible tuning: weaker augmentation (halve the `p_*` values), `border: grey`, a decision threshold chosen on validation data.

## 11. Suggested slide storyline

1. Goal: better classifier for the delivered mosquito app (Anopheles and Culex are the priority species).
2. Pipeline diagram (section 3.1) and what was replaced.
3. Data: sources, 14 species, split, class imbalance (section 4).
4. Method: Swin V2, focal loss, crops as training input, focus-class selection (section 5).
5. Headline result: bar chart of macro AP, old ONNX / old pth / new whole-photo / new pipeline (section 2).
6. Per-class results: Anopheles and Culex tritaeniorhynchus AP and F1 (section 7.2).
7. What did not help: bigger models, stronger augmentation, longer training (section 6.2).
8. Caveats (section 7.3).
9. Tooling: run.sh, one-click setup, Telegram reports.
10. Next steps (section 10).

## 12. Glossary
- **AP / macro AP:** average precision of a class ranked by score; macro = mean over classes. 1.0 is perfect.
- **F1:** harmonic mean of precision and recall at a fixed threshold (0.5 here).
- **Exact match:** share of photos whose full set of present classes is predicted correctly.
- **TTA:** test-time augmentation, averaging the prediction over 4 rotations.
- **Focal loss / class weights:** training loss that focuses on hard examples and up-weights rare classes.
- **Temperature scaling:** a single factor that calibrates the output probabilities on the validation set.
- **Voronoi split:** when a detection holds several mosquitoes, each crop keeps only the pixels nearest its own body.
