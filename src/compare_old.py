"""Run the old pipeline (mosquitto_msislab/infer.py: detectors -> body detector -> 4-class SwinV2-B) on a set of images.

Returns per-image scores [N, 4] (max probability per class over the image's detections; an image with no
detection is scored as Non). Results are cached under <runs>/compare_cache, keyed on the old weights and the image
list, so the old model only runs again when its weights or the test split change.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from compare_classes import COARSE


def _key(images, cls_path, infer_dir):
    h = hashlib.sha1()
    for p in (infer_dir / "infer.py", Path(cls_path)):
        st = Path(p).stat()
        h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
    h.update("|".join(str(p) for p in images).encode())
    return h.hexdigest()[:16]


def _config(infer_dir, tmp):
    """Copy of the old config.yaml without debug images / debug logging (the app config saves step images)."""
    cfg = yaml.safe_load((infer_dir / "config" / "config.yaml").read_text(encoding="utf-8"))
    for k in ("debug_step_images", "debug_enabled"):
        cfg[k]["Current"] = 0
    cfg["log_dir"]["Current"] = str(tmp / "log")
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def _scores(results, images, classes):
    """[N, len(classes)] max probability per class; an image without detection is scored as the "Non" / "No Mosq." class."""
    by_stem = {Path(r["file"]).stem: r["detections"] for r in results}
    out = torch.zeros(len(images), len(classes))
    for i, img in enumerate(images):
        dets = by_stem.get(Path(img).stem, [])
        if not dets:
            out[i, [j for j, c in enumerate(classes) if c in ("Non", "No Mosq.")][0]] = 1.0
        for d in dets:
            out[i] = torch.maximum(out[i], torch.tensor(d["probs"]))
    return out


def run_old(images, infer_dir, cls_model=None, device=None, cache_dir=None, classes=COARSE):
    infer_dir = Path(infer_dir).resolve()
    cls_path = (Path(cls_model).resolve() if Path(cls_model).exists() else Path(cls_model)) if cls_model else infer_dir / "models" / yaml.safe_load(
        (infer_dir / "config" / "config.yaml").read_text(encoding="utf-8"))["cls_model"]["Current"]
    cache = Path(cache_dir) / f"old_{_key(images, cls_path, infer_dir)}.json" if cache_dir else None
    if cache and cache.exists():
        return _scores(json.loads(cache.read_text()), images, classes), cls_path
    with tempfile.TemporaryDirectory(prefix="cmp_old_") as t:
        tmp = Path(t)
        (tmp / "in").mkdir()
        for img in images:  # infer.py takes a directory; link by name so results map back by stem
            (tmp / "in" / Path(img).name).symlink_to(Path(img).resolve())
        cmd = [sys.executable, str(infer_dir / "infer.py"), "--input", str(tmp / "in"), "--output", str(tmp / "out")]
        if cls_model:
            cmd += ["--cls-model", str(cls_path)]
        if device:
            cmd += ["--device", str(device).removeprefix("cuda:")]
        subprocess.run(cmd, check=True, cwd=infer_dir, env={**os.environ, "INFER_CONFIG": str(_config(infer_dir, tmp))},
                       stdout=subprocess.DEVNULL)
        results = json.loads((tmp / "out" / "results.json").read_text(encoding="utf-8"))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(results, ensure_ascii=False))
    return _scores(results, images, classes), cls_path
