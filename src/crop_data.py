"""Per-mosquito crop dataset for the species classifier, matching what infer.py feeds its classifier.

Every whole-mosquito box of the input folder's full_body_detection (YOLO txt) becomes one training sample: the box
plus a small margin, other mosquitoes' pixels replaced by grey with the pipeline's nearest-centre (Voronoi) rule, then
squashed to size x size exactly like infer.classify does. The split is the images/{train,val,test} folder. Crops are
cached under outputs/runs/crop_cache, keyed on the size, margin and every image / label file's size and mtime, so
editing the data rebuilds them automatically.
"""
from concurrent.futures import ThreadPoolExecutor

import hashlib

import numpy as np
import torch
from PIL import Image

import common
import dataset

Image.MAX_IMAGE_PIXELS = None
FILL = 114  # grey for masked-out neighbours (close to the train mean the pipeline fills with)
TASK = "full_body_detection"


def pixel_boxes(label_file, w, h):
    """[(class id, x1, y1, x2, y2)] in pixels from a YOLO label file."""
    return [(c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            for c, (cx, cy, bw, bh) in dataset.read_boxes(label_file)]


def crop_one(im, k, boxes, size, margin):
    """uint8 [3, size, size] crop of box k, neighbours whose centre lies inside the crop masked by nearest centre."""
    _, x1, y1, x2, y2 = boxes[k]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    X1, Y1, X2, Y2 = (max(int(round(v)), 0) for v in (x1 - mx, y1 - my, x2 + mx, y2 + my))
    X2, Y2 = min(X2, im.shape[1]), min(Y2, im.shape[0])
    sub = im[Y1:Y2, X1:X2].copy()
    cen = [((b[1] + b[3]) / 2, (b[2] + b[4]) / 2) for b in boxes]
    near = [j for j, (cx, cy) in enumerate(cen) if j == k or (X1 <= cx < X2 and Y1 <= cy < Y2)]
    if len(near) > 1:
        ys, xs = np.mgrid[Y1:Y2, X1:X2]
        d = np.stack([(xs - cen[j][0]) ** 2 + (ys - cen[j][1]) ** 2 for j in near], -1)
        sub[d.argmin(-1) != near.index(k)] = FILL
    sq = Image.fromarray(sub).resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(sq).copy()).permute(2, 0, 1)


def crops_of_image(img_file, label_file, size, margin):
    with Image.open(img_file) as im:
        w0, h0 = im.size
        im.draft("RGB", (w0 // 2, h0 // 2))  # JPEG DCT downscale: crops are still far larger than `size`
        arr = np.asarray(im.convert("RGB"))
    boxes = [(c, x1 * arr.shape[1] / w0, y1 * arr.shape[0] / h0, x2 * arr.shape[1] / w0, y2 * arr.shape[0] / h0)
             for c, x1, y1, x2, y2 in pixel_boxes(label_file, w0, h0)]
    return [(boxes[k][0], crop_one(arr, k, boxes, size, margin)) for k in range(len(boxes))]


def fingerprint(pairs, size, margin):
    """Hash of the settings and of every image / label file (size, mtime): changes whenever the data does."""
    h = hashlib.sha1(f"{size}:{margin}".encode())
    for img, lab in pairs:
        for f in (img, lab):
            st = f.stat() if f.exists() else None
            h.update(f"{f.name}:{st.st_size if st else 0}:{int(st.st_mtime) if st else 0}".encode())
    return h.hexdigest()[:12]


def build_split(pairs, size, margin):
    """(class ids, uint8 [N,3,S,S], stems) of every box in the (image, label) pairs."""
    with ThreadPoolExecutor(8) as ex:
        per = list(ex.map(lambda p: crops_of_image(p[0], p[1], size, margin), pairs))
    flat = [(c, x, p[0].stem) for p, items in zip(pairs, per) for c, x in items]
    return [c for c, _, _ in flat], torch.stack([x for _, x, _ in flat]), [s for _, _, s in flat]


def load_items(size, margin=0.05):
    """Same contract as cls_data.load_items: {split: (uint8 [N,3,S,S], one-hot [N,C], stems)}, class names."""
    task = "full_body_detection"
    names = dataset.class_names(task)
    dataset.check(task)
    out = {}
    for split in dataset.SPLITS:
        pairs = dataset.items(task, split)
        cache = common.RUNS / "crop_cache" / f"crops_{split}_{size}_{margin}_{fingerprint(pairs, size, margin)}.pt"
        if cache.exists():
            cls, img, stems = torch.load(cache, weights_only=True)
        else:
            cls, img, stems = build_split(pairs, size, margin)
            for old in cache.parent.glob(f"crops_{split}_{size}_{margin}_*.pt"):
                old.unlink()  # stale: the data changed
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save((cls, img, stems), cache)
        y = torch.zeros(len(cls), len(names))
        y[torch.arange(len(cls)), torch.tensor(cls, dtype=torch.long)] = 1
        out[split] = (img, y, stems)
    return out, names
