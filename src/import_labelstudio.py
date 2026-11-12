"""Build the training input folder (config paths.input) from a Label Studio export (config paths.labelstudio_export).

Only needed to (re)create the folder from the original export: it makes the photo-level 70/15/15 split
(make_splits.py), symlinks the images and writes the YOLO labels + data.yaml per task. The OBB task becomes the single
class `body` (the species text printed in the 라벨링 images cannot be used). After that, edit the input folder directly.

Usage: import_labelstudio.py [--out DIR] [--seed N] [--force]     (refuses to touch an existing task folder without --force)
"""
import argparse
import shutil
from pathlib import Path

import yaml

import common
import make_splits

TASKS = ("full_body_detection", "body_oriented_detection")
SINGLE_CLASS = {"body_oriented_detection": "body"}


def export_task(task_dir, split, out):
    src, root = common.LABELSTUDIO / task_dir, out / task_dir
    for kind in ("images", "labels"):
        shutil.rmtree(root / kind, ignore_errors=True)
    for stem, rec in common.load_records(task_dir).items():
        s = split[rec["image_path"]]
        img, lab = common.image_file(task_dir, stem), src / "annotations" / f"{stem}.txt"
        for kind in ("images", "labels"):
            (root / kind / s).mkdir(parents=True, exist_ok=True)
        (root / "images" / s / f"{stem}{img.suffix}").symlink_to(img.resolve())
        lines = [ln for ln in lab.read_text().splitlines() if ln.strip()]
        if task_dir in SINGLE_CLASS:
            lines = [" ".join(["0"] + ln.split()[1:]) for ln in lines]
        (root / "labels" / s / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    names = [SINGLE_CLASS[task_dir]] if task_dir in SINGLE_CLASS else \
        [n for n in (src / "classes.txt").read_text().split("\n") if n]
    (root / "data.yaml").write_text(yaml.safe_dump({"names": dict(enumerate(names))}, allow_unicode=True, sort_keys=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(common.INPUT), help="target folder (default: config paths.input)")
    ap.add_argument("--seed", type=int, help="override split.seed of the config")
    ap.add_argument("--force", action="store_true", help="replace the images/labels of existing task folders")
    a = ap.parse_args()
    out = Path(a.out)
    busy = [t for t in TASKS if (out / t / "images").exists()]
    if busy and not a.force:
        raise SystemExit(f"{out} already has {', '.join(busy)}; use --force to replace them (your edits there are lost)")
    split, report = make_splits.make(a.seed)
    for t in TASKS:
        export_task(t, split, out)
    print(report["per_split"], "->", out)


if __name__ == "__main__":
    main()
