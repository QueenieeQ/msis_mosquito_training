"""Drop a fresh Label Studio export into the training input folder and it is converted on first use.

Put the downloaded export (the `consolidated_dataset` folder, or the whole export folder that contains it) anywhere
in `input/datasets/` (config paths.input), or put its task folders (full_body_detection/, ...) straight in there.
The next training run (or `python src/dataset.py`) notices it, makes the photo-grouped 70/15/15 split, writes
images/{train,val,test} + labels/ + data.yaml per task, and trains on the result. The previous task folders are kept in
`input/datasets/_previous/<timestamp>/`. The same export is converted only once (marker `_imported.json`);
a different export triggers a new conversion. No export in the folder = the data there is used as it is.
"""
import fcntl
import hashlib
import json
import shutil
import time

import common
import import_labelstudio
import make_splits

MARKER = "_imported.json"
RAW_DIRS = ("full_body_detection", "body_oriented_detection", "species_classification")
_ready = set()


def _is_export(d):
    """A Label Studio export root: full_body_detection/{annotations/*.json, classes.txt}."""
    f = d / "full_body_detection"
    return (f / "classes.txt").is_file() and any((f / "annotations").glob("*.json"))


def find_export(root):
    """The export folder inside `root` (itself, consolidated_dataset/, or one folder deeper), else None."""
    for d in [root, root / "consolidated_dataset",
              *sorted(p for p in root.glob("*/") if p.is_dir() and not p.name.startswith("_")),
              *sorted(root.glob("*/consolidated_dataset"))]:
        if _is_export(d):
            return d
    return None


def fingerprint(export):
    """Stable across copies (names, sizes and class lists only, not timestamps)."""
    h = hashlib.sha1()
    for task in RAW_DIRS:
        for f in sorted((export / task / "annotations").glob("*.json")):
            h.update(f"{task}/{f.name}:{f.stat().st_size};".encode())
        if (export / task / "classes.txt").exists():
            h.update((export / task / "classes.txt").read_bytes())
    return h.hexdigest()


def _tuck_raw_folders(root):
    """Task folders holding raw `annotations/` directly in `root` move into root/consolidated_dataset/."""
    raw = [t for t in RAW_DIRS if (root / t / "annotations").is_dir() and not (root / t / "images" / "train").exists()]
    if raw:
        dest = root / "consolidated_dataset"
        dest.mkdir(exist_ok=True)
        for t in raw:
            shutil.move(str(root / t), str(dest / t))


def _convert(root, export, seed=None):
    prev = root / "_previous" / time.strftime("%Y%m%d_%H%M%S")
    for t in import_labelstudio.TASKS:
        if (root / t).exists():
            prev.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root / t), str(prev / t))
    old, common.LABELSTUDIO = common.LABELSTUDIO, export
    try:
        split, report = make_splits.make(seed)
        for t in import_labelstudio.TASKS:
            import_labelstudio.export_task(t, split, root)
    finally:
        common.LABELSTUDIO = old
    return report


def ensure_ready(root=None):
    """Convert a new export found in the input folder; no-op (and cheap) otherwise. Safe for parallel trainers."""
    root = root or common.INPUT
    if str(root) in _ready or not root.is_dir():
        return
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".ingest.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _tuck_raw_folders(root)
        export = find_export(root)
        if export is not None:
            marker = root / MARKER
            fp = fingerprint(export)
            done = json.loads(marker.read_text()).get("fingerprint") if marker.exists() else None
            if fp != done:
                print(f"[ingest] new Label Studio export found: {export}\n[ingest] converting to {root} ...", flush=True)
                report = _convert(root, export)
                marker.write_text(json.dumps({"fingerprint": fp, "export": str(export), "split": report["per_split"],
                                              "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
                print(f"[ingest] done, images per split: {report['per_split']}", flush=True)
    _ready.add(str(root))
