"""Training input folder: the one place the trainers read data from (default input/datasets, config paths.input).

    <input>/full_body_detection/     whole-mosquito boxes (YOLO txt: class cx cy w h) -> detector AND species classifier
        data.yaml                    names: {0: Culex spp., 1: ...}  (the class list; edit it to add / rename classes)
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/<same stem>.txt
    <input>/body_oriented_detection/ oriented body boxes (class x1 y1 x2 y2 x3 y3 x4 y4) -> OBB detector, same layout

Change the data by adding / removing / editing files there, then train; nothing else needs rebuilding.
A downloaded Label Studio export dropped into the folder is converted to this layout automatically (see ingest.py).
`python src/dataset.py [full|obb|species ...]` prints a summary and fails with a readable message on any problem.
"""
import sys
from collections import Counter

import yaml

import common
import ingest

SPLITS = ("train", "val", "test")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TASK_DIRS = {"full": "full_body_detection", "obb": "body_oriented_detection", "species": "full_body_detection"}
FIELDS = {"full_body_detection": 5, "body_oriented_detection": 9}  # numbers per label line


class DatasetError(SystemExit):
    """Raised with a message that tells the user what to fix; exits cleanly when uncaught."""


def task_root(task_dir):
    ingest.ensure_ready()  # converts a freshly dropped Label Studio export first (see ingest.py)
    return common.INPUT / task_dir


def class_names(task_dir):
    """Class list from data.yaml `names` (a {id: name} map or a list), ordered by id."""
    f = task_root(task_dir) / "data.yaml"
    if not f.exists():
        raise DatasetError(f"missing {f} (needs a `names:` list of class names)")
    names = yaml.safe_load(f.read_text()).get("names")
    if not names:
        raise DatasetError(f"{f}: no `names:`")
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def items(task_dir, split):
    """Sorted [(image file, label file)] of one split; the label file may not exist (= no objects)."""
    root = task_root(task_dir)
    imgs = sorted(p for p in (root / "images" / split).glob("*") if p.suffix.lower() in IMAGE_EXT)
    return [(p, root / "labels" / split / f"{p.stem}.txt") for p in imgs]


def read_boxes(label_file):
    """[(class id, [numbers...])] of a YOLO label file; [] if it does not exist."""
    if not label_file.exists():
        return []
    return [(int(p[0]), [float(v) for v in p[1:]]) for p in (ln.split() for ln in label_file.read_text().splitlines())
            if p]


def check(task_dir):
    """Validate one task folder; returns {split: {"images": n, "boxes": Counter of class names}}."""
    names, n_fields, out = class_names(task_dir), FIELDS[task_dir], {}
    for split in SPLITS:
        pairs = items(task_dir, split)
        if not pairs:
            raise DatasetError(f"{task_root(task_dir) / 'images' / split} has no images "
                               f"(each of train / val / test needs some; extensions {sorted(IMAGE_EXT)})")
        boxes = Counter()
        for img, lab in pairs:
            for ln in (lab.read_text().splitlines() if lab.exists() else []):
                p = ln.split()
                if not p:
                    continue
                if len(p) != n_fields:
                    raise DatasetError(f"{lab}: expected {n_fields} numbers per line, found {len(p)}")
                if not 0 <= int(p[0]) < len(names):
                    raise DatasetError(f"{lab}: class id {p[0]} is not in data.yaml names (0..{len(names) - 1})")
                boxes[names[int(p[0])]] += 1
        out[split] = {"images": len(pairs), "boxes": boxes}
    return out


def summary(task_dir):
    res, names = check(task_dir), class_names(task_dir)
    rows = [f"{task_dir}  ({task_root(task_dir)})"]
    rows.append("  " + "  ".join(f"{s}: {res[s]['images']} images / {sum(res[s]['boxes'].values())} boxes" for s in SPLITS))
    rows += [f"    {n:<28}" + " ".join(f"{res[s]['boxes'][n]:>6}" for s in SPLITS) for n in names]
    return "\n".join(rows)


def main(args):
    tasks = [TASK_DIRS[a] for a in args] if args else sorted(set(TASK_DIRS.values()))
    for t in dict.fromkeys(tasks):
        print(summary(t))


if __name__ == "__main__":
    main(sys.argv[1:])
