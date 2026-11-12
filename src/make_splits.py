"""70/15/15 split grouped by photo, stratified by species and balanced across annotators.

A 원본 image and its 라벨링 copy are the same photo, so they form one group (common.group_key) and
always land in the same split. split.json maps image_path -> train/val/test and is shared by all
three task types. Greedy proportional assignment per (species, annotator) stratum keeps every
stratum close to 70/15/15 even when it has few groups.
"""
import argparse
import json
import random
from collections import Counter, defaultdict

import common

RATIOS = common.CFG["split"]["ratios"]


def collect(canon):
    """group key -> {paths, species (most frequent), annotator} over all task directories."""
    groups, seen = {}, set()
    for task in common.TASK_DIRS:
        for rec in common.load_records(task).values():
            path = rec["image_path"]
            if path in seen:
                continue
            seen.add(path)
            g = groups.setdefault(common.group_key(path), {"paths": [], "counts": Counter(), "annotator": None})
            g["paths"].append(path)
            g["counts"].update(common.species_counts(rec, canon))
            g["annotator"] = g["annotator"] or common.primary_annotation(rec).get("completed_by_email") or "unknown"
    for g in groups.values():
        g["species"] = g["counts"].most_common(1)[0][0] if g["counts"] else "none"
    return groups


def assign(groups, seed=0):
    """{group key: split}; `groups` maps key -> {"species", "annotator"}."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for key, g in groups.items():
        strata[(g["species"], g["annotator"])].append(key)
    total, split = Counter(), {}
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        seen = Counter()
        for key in keys:
            n = sum(seen.values()) + 1
            # largest deficit vs. the target ratio inside the stratum; global deficit breaks ties
            pick = max(RATIOS, key=lambda s: (RATIOS[s] * n - seen[s],
                                              RATIOS[s] * (sum(total.values()) + 1) - total[s]))
            split[key] = pick
            seen[pick] += 1
            total[pick] += 1
    return split


def make(seed=None):
    """Write split.json + split_stats.json; return the image_path -> split mapping."""
    seed = common.CFG["split"]["seed"] if seed is None else seed
    groups = collect(common.load_canon())
    by_group = assign(groups, seed)
    split = {p: by_group[k] for k, g in groups.items() for p in g["paths"]}
    common.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    common.SPLIT_FILE.write_text(json.dumps(split, ensure_ascii=False, indent=1))
    species, annot = {s: Counter() for s in RATIOS}, {s: Counter() for s in RATIOS}
    for k, g in groups.items():
        species[by_group[k]][g["species"]] += len(g["paths"])
        annot[by_group[k]][g["annotator"]] += len(g["paths"])
    report = {"images": len(split), "groups": len(groups),
              "per_split": {s: sum(c.values()) for s, c in species.items()},
              "species": {s: dict(c) for s, c in species.items()},
              "annotator": {s: dict(c) for s, c in annot.items()}}
    (common.SPLIT_FILE.parent / "split_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return split, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, help="override split.seed of the config")
    _, report = make(ap.parse_args().seed)
    print(json.dumps(report["per_split"]), "->", common.SPLIT_FILE)


if __name__ == "__main__":
    main()
