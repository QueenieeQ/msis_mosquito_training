"""Loss, class weighting, LR schedule, metric and calibration helpers for the species classifier."""
import math

import torch
import torch.nn.functional as F


def class_weights(y, beta=0.95):
    """Effective-number weights w_c ~ (1 - beta) / (1 - beta^n_c), normalised to mean 1."""
    n = y.sum(0).clamp(min=1)
    w = (1 - beta) / (1 - beta ** n)
    return w / w.mean()


def focal(logits, y, w, gamma=2.0):
    """Multi-label (sigmoid) focal loss with per-class weights."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.exp(-bce)
    return ((1 - p) ** gamma * bce * w).mean()


def macro_ap(scores, y):
    """Mean average precision over the classes that have at least one positive."""
    aps = []
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0:
            continue
        hit = y[scores[:, c].argsort(descending=True), c]
        prec = hit.cumsum(0) / torch.arange(1, len(hit) + 1)
        aps.append(float((prec * hit).sum() / hit.sum()))
    return sum(aps) / max(len(aps), 1)


def focus_ap(scores, y, names, focus):
    """{class: AP} for the focus classes (the species that matter most); empty focus -> {}."""
    return {n: macro_ap(scores[:, [names.index(n)]], y[:, [names.index(n)]]) for n in focus}


def lr_at(ep, epochs, warmup, floor=0.01):
    """LR multiplier: linear warmup, then cosine from 1 down to `floor` (1e-6 / 1e-4)."""
    if ep < warmup:
        return (ep + 1) / warmup
    p = (ep - warmup) / max(1, epochs - warmup - 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def fit_temperature(logits, y):
    """Temperature T minimising validation BCE of sigmoid(logits / T)."""
    t = torch.nn.Parameter(torch.tensor(1.5))
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / t, y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.detach().clamp(0.5, 5))
