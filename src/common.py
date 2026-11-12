"""Shared paths/config, plus helpers for reading a Label Studio export (only used by import_labelstudio.py)."""
import json
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent  # project root; code lives in src/
CONFIG_FILE = ROOT / "config" / "config_training.yaml"


def load_config(path=None):
    return yaml.safe_load(Path(path or CONFIG_FILE).read_text())


def _path(p):
    return Path(p) if Path(p).is_absolute() else ROOT / p


CFG = load_config()
INPUT = _path(CFG["paths"]["input"])  # training data: <task>/{images,labels}/{train,val,test} + data.yaml
LABELSTUDIO = _path(CFG["paths"]["labelstudio_export"])
RUNS = _path(CFG["paths"]["runs"])
YOLO_DATASETS = _path(CFG["paths"]["yolo_datasets"])
SPLIT_FILE = _path(CFG["paths"]["splits"]) / "split.json"
TASK_DIRS = ("full_body_detection", "body_oriented_detection", "species_classification")
BODY_SUFFIX = "_body"


def group_key(image_path):
    """Photo identity shared by a 원본/라벨링 pair: folder below member/kind + name without `_(N)`.

    Same rule as scripts/repair_labelstudio_pairs.py; the pair is one photo, so it must share a split.
    """
    parts = image_path.split("/")
    return "/".join(parts[2:-1]) + "/" + re.sub(r"_\(\d+\)$", "", os.path.splitext(parts[-1])[0])


def primary_annotation(record):
    """Latest non-cancelled annotation (same rule the exporter used)."""
    done = [a for a in record["annotations"] if not a.get("was_cancelled")]
    return max(done, key=lambda a: a.get("updated_at") or "")


def _norm(label):
    return label.lower().removesuffix(BODY_SUFFIX).replace("_", " ").replace(".", "").strip()


def canonical_species(label, canon):
    """Map a raw/body label to its whole-mosquito species name (case/punctuation-insensitive)."""
    return canon.get(_norm(label), label.removesuffix(BODY_SUFFIX))


def load_canon():
    """{lowercase species: canonical species} from the full-body class list."""
    names = (LABELSTUDIO / "full_body_detection" / "classes.txt").read_text().split("\n")
    return {_norm(n): n for n in names if n}


def species_counts(record, canon):
    """Species label counts of an image, from whole-mosquito boxes, else body boxes."""
    ann = primary_annotation(record)
    full = [r["value"]["rectanglelabels"][0] for r in ann["result"] if r.get("from_name") == "label"]
    body = [r["value"]["rectanglelabels"][0] for r in ann["result"] if r.get("from_name") == "label_body"]
    counts = {}
    for lab in full or body:
        s = canonical_species(lab, canon)
        counts[s] = counts.get(s, 0) + 1
    return counts


def load_records(task_dir):
    """{stem: record} for every annotation JSON of a task directory."""
    out = {}
    for f in sorted((LABELSTUDIO / task_dir / "annotations").glob("*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


def image_file(task_dir, stem):
    hits = list((LABELSTUDIO / task_dir / "images").glob(f"{stem}.*"))
    return hits[0] if hits else None


def load_split():
    return json.loads(SPLIT_FILE.read_text())
