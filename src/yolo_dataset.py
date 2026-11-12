"""Ultralytics data.yaml for a detector task, pointing straight at the files in the input folder (nothing is copied)."""
import yaml

import common
import dataset

TASKS = {"full": "full_body_detection", "obb": "body_oriented_detection"}


def class_names(task):
    return dataset.class_names(TASKS[task])


def build(task):
    """Validate the task's input folder and return the path of a generated data.yaml for model.train()."""
    task_dir = TASKS[task]
    dataset.check(task_dir)
    cfg = {"path": str(dataset.task_root(task_dir).resolve()), "train": "images/train", "val": "images/val",
           "test": "images/test", "names": dict(enumerate(class_names(task)))}
    yml = common.YOLO_DATASETS / f"{task_dir}.yaml"
    yml.parent.mkdir(parents=True, exist_ok=True)
    yml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return yml
