"""Common label space and image-level metrics for comparing the new 14-species Swin V2-T with the old 4-class SwinV2-B.

The old classifier only knows Anopheles / Culex tritaeniorhynchus / Non / Others, so the new model's species are
collapsed onto those four classes. Both models are scored per image (multi-label): a class is present if any of its
mosquitoes is, and its score is the highest probability for that class.
"""
import torch

from cls_metrics import macro_ap

COARSE = ("Anopheles", "Culex tritaeniorhynchus", "Non", "Others")  # = class order of the old checkpoints / cls_labels.txt
_DIRECT = {"Anopheles spp.": 0, "Culex tritaeniorhynchus": 1, "No Mosq.": 2}


def coarse_index(species):
    """Index in COARSE of a new-model species name (everything not listed, incl. Unknown, is Others)."""
    return _DIRECT.get(species, 3)


def truth(species_names):
    """Multi-hot [4] over COARSE for the species present in one image."""
    y = torch.zeros(len(COARSE))
    for s in species_names:
        y[coarse_index(s)] = 1
    return y


def collapse(probs, classes):
    """[N, n_species] probabilities -> [N, 4], max over the species of each coarse class."""
    out = torch.zeros(probs.shape[0], len(COARSE))
    for j, s in enumerate(classes):
        k = coarse_index(s)
        out[:, k] = torch.maximum(out[:, k], probs[:, j])
    return out


def metrics(scores, y, thr=0.5):
    """Per-class AP / precision / recall / F1 at `thr`, plus macro AP, macro F1 and exact-match accuracy."""
    pred = scores >= thr
    res = {"classes": {}, "n_images": int(y.shape[0])}
    for k, name in enumerate(COARSE):
        tp = int((pred[:, k] & (y[:, k] == 1)).sum())
        fp = int((pred[:, k] & (y[:, k] == 0)).sum())
        fn = int((~pred[:, k] & (y[:, k] == 1)).sum())
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res["classes"][name] = {"support": int(y[:, k].sum()), "precision": p, "recall": r,
                                "f1": 2 * p * r / max(p + r, 1e-9),
                                "ap": macro_ap(scores[:, [k]], y[:, [k]])}
    c = res["classes"].values()
    res["macro_ap"] = sum(v["ap"] for v in c if v["support"]) / max(sum(1 for v in c if v["support"]), 1)
    res["macro_f1"] = sum(v["f1"] for v in c if v["support"]) / max(sum(1 for v in c if v["support"]), 1)
    res["exact_match"] = float((pred == (y == 1)).all(1).float().mean())
    return res


def table(rows, keys=("macro_ap", "macro_f1", "exact_match")):
    """Markdown table of {model name: metrics()} : one row per model, per-class AP / F1 appended."""
    head = list(keys) + [f"AP {c}" for c in COARSE] + [f"F1 {c}" for c in COARSE]
    lines = ["| model | " + " | ".join(head) + " |", "|---" * (len(head) + 1) + "|"]
    for name, m in rows.items():
        vals = [m[k] for k in keys] + [m["classes"][c]["ap"] for c in COARSE] + [m["classes"][c]["f1"] for c in COARSE]
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    return "\n".join(lines)
