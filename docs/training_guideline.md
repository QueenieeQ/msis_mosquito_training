Here is the updated, practical training guideline recalibrated specifically to your actual dataset statistics, exported data structures, and multi-task format.

---

### Updated Training & Model Configuration Guideline

#### 1. Data Partitioning & Dataset Characteristics

* **Real Dataset Scope:** 1,622 annotated tasks split across 32 projects.
* **`full_body_detection`:** 975 images (Horizontal Bounding Boxes in YOLO txt & Label Studio JSON).
* **`body_oriented_detection`:** 720 images (Rotated Bounding Boxes / Oriented BBoxes in YOLO-OBB format).
* **`species_classification`:** 975 images (Per-image species counts + `species_labels.csv`).


* **Train / Validation / Test Split:** Apply a **70% / 15% / 15%** split grouped by `task_id` / `image_id` across all task directories to avoid data leakage between tasks.
* **Stratification:** Stratify the splits based on species distributions extracted from `species_classification/species_labels.csv`.
* **Annotator Bias Mitigation:** Since 4 primary members (`dogyunkim`, `insungpark`, `jihoon.an`, `minhyuk.lee`) contributed roughly equal amounts (~390–412 tasks each), distribute each annotator’s tasks proportionally across train/val/test splits to minimize individual labeling bias.

---

#### 2. Pipeline-Specific Architectures & Framework Adapters

Given the dual format (`YOLO` horizontal boxes vs. `YOLO-OBB` rotated boxes), standard Faster R-CNN needs adaptation depending on the task target:

##### Task A: Standard Full Body Detection (`full_body_detection` — 975 images)

* **Framework:** Detectron2 / PyTorch or Ultralytics YOLOv8/v10.
* **Architecture Option 1 (Standard BBox):** Swin-T + Faster R-CNN (RPN + RoI Align).
* **Input Resolution:** 640 × 640 pixels with letterboxing.

##### Task B: Oriented Body Detection (`body_oriented_detection` — 720 images)

* **Framework:** MMRotate (PyTorch / Detectron2 extension) or YOLOv8-OBB.
* **Architecture Option 2 (Rotated BBox):** Oriented R-CNN with Swin-T backbone or YOLOv8n/s-OBB to handle arbitrary mosquito angles on sticky traps without bounding box overlap contamination.

##### Task C: Species Classification (`species_classification` — 975 images)

* **Architecture:** Swin Transformer (Swin-T) backbone initialized with ImageNet-1K weights, using a Multi-Label Classification head or Count Regression head based on `species_labels.csv`.

---

#### 3. Data Preprocessing & Augmentation Pipeline

* **Standard Resolution:** 640 × 640 pixels (Letterboxed, aspect-ratio preserved).
* **Multi-Modal / Spectral Note:** Since the current dataset consists of standard RGB images (no native UV channels in the current export), execute preprocessing strictly on 3-channel RGB:
* **Normalization:** ImageNet mean/std or dataset-specific z-score normalization computed strictly over the 70% training split.
* **Sharpening:** Optional mild Gaussian sharpening kernel to enhance wing venation.


* **Augmentation Strategy:**
* **Rigid Rotations:** 0°, 90°, 180°, and 270° rotations (ideal for orientation-agnostic sticky-trap images).
* **Geometric Transforms:** Random scale (0.8–1.2), translation (±10%), horizontal/vertical flips ($p = 0.5$).
* **Color & Intensity:** Mild HSV jitter (Hue $\pm 0.015$, Saturation $\pm 0.7$, Value $\pm 0.4$) to handle varying light conditions inside trap boxes.



---

#### 4. Optimization & Hyperparameter Settings

| Parameter | Value / Setting | Description |
| --- | --- | --- |
| **Optimizer** | AdamW | Weight decay regularized optimizer |
| **Base Learning Rate ($lr_0$)** | $1 \times 10^{-4}$ (Backbone: $1 \times 10^{-5}$) | Fine-tuning Swin-T backbone with differential learning rates |
| **Weight Decay** | 0.05 | Prevents overfitting on 1,622 samples |
| **Batch Size** | 16 (GPU VRAM $\ge$ 16GB) | Mini-batch sizing |
| **LR Schedule** | Cosine Annealing with Warmup | 5-epoch linear warmup, decaying to $1 \times 10^{-6}$ |
| **Training Epochs** | 80 Epochs (~8,000–10,000 iterations) | Early stopping based on Validation mAP@0.5:0.95 |
| **Validation Interval** | Every 200 iterations | Tracks validation loss and mAP |

---

#### 5. Class Imbalance & Loss Formulations

* **Effective-Number Class Weighting:** Scale loss weights based on species occurrences in `species_labels.csv`:

$$E_c = \frac{1 - \beta^{n_c}}{1 - \beta}, \quad w_c \propto \frac{1}{E_c} \quad (\beta = 0.95)$$


* **Loss Functions:**
* **Full Body BBox:** CIoU Loss + Cross-Entropy Classification Loss.
* **Oriented BBox (OBB):** Probabilistic IoU (PIoU) or Rotated IoU Loss.
* **Species Classification:** Focal Loss ($\gamma = 2.0$) to address species frequency disparities.



---

#### 6. Inference & Calibration

* **Test-Time Augmentation (TTA):** 4-rotation TTA (0°, 90°, 180°, 270°) during evaluation to stabilize detection counts.
* **Temperature Scaling:** Calibrate classification logits on the validation split ($T \approx 1.5 - 1.8$) to improve confidence score reliability.