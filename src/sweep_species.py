"""Grid search over the species classifier hyper-parameters (config `sweep` section).

Each combination runs train_classifier.py with CLI overrides on one GPU from sweep.devices (one job per
GPU at a time). Runs land in <runs>/sweep_species/<tag>/. Ranking uses the VALIDATION macro-AP only;
the test score is printed for information and must not be used to pick the winner.
Telegram: progress and /status come from telegram_monitor.py (skip with --no-telegram).
Usage: sweep_species.py [--dry-run] [--epochs N] [--limit N] [--no-telegram]
"""
import argparse
import csv
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import common
import telegram_monitor

SWEEP = common.CFG["sweep"]
OUT = common.RUNS / "sweep_species"
SHORT = {"lr_head": "h", "lr_backbone": "b", "weight_decay": "wd"}


def combos(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def tag(params):
    return "_".join(f"{SHORT.get(k, k)}{v:g}" for k, v in params.items())


def command(params, device, epochs, limit):
    cmd = [sys.executable, "src/train_classifier.py", "--name", f"sweep_species/{tag(params)}",
           "--device", device, "--epochs", str(epochs)]
    if limit:
        cmd += ["--limit", str(limit)]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def set_status(t, value):
    (OUT / f"{t}.status").write_text(value + "\n")


def start_monitor(jobs):
    """Background telegram_monitor watching one pseudo-task per job (state/log files live in OUT)."""
    if not (common.ROOT / "config" / "telegram.env").exists():
        return None
    names = [f"sweep_species/{tag(j)}" for j in jobs]
    return subprocess.Popen([sys.executable, "src/telegram_monitor.py", "watch", "--title", "Species sweep", *names],
                            stdout=open(OUT / "telegram.log", "w"), stderr=subprocess.STDOUT)


def run_all(jobs, epochs, limit, notify=True):
    OUT.mkdir(parents=True, exist_ok=True)
    for j in jobs:
        set_status(tag(j), "ok" if (OUT / tag(j) / "metrics.json").exists() else "pending")
    mon = start_monitor(jobs) if notify else None
    free = Queue()
    for d in SWEEP["devices"]:
        free.put(d)

    def one(params):
        device = free.get()
        try:
            t = tag(params)
            if (OUT / t / "metrics.json").exists():
                return params
            print(f"start  {t} on {device}", flush=True)
            set_status(t, "running")
            with open(OUT / f"{t}.log", "w") as log:
                rc = subprocess.run(command(params, device, epochs, limit), stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"{'done  ' if rc == 0 else 'FAILED'} {t}", flush=True)
            set_status(t, "ok" if rc == 0 else "failed")
            return params
        finally:
            free.put(device)

    with ThreadPoolExecutor(len(SWEEP["devices"])) as ex:
        list(ex.map(one, jobs))
    if mon:
        mon.wait()


def summarize(jobs):
    rows = []
    for p in jobs:
        f = OUT / tag(p) / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            rows.append({**p, "val_macro_ap": m["val_macro_ap"], "test_plain": m.get("test_macro_ap_plain"),
                         "test_tta4": m.get("test_macro_ap_tta4")})
    rows.sort(key=lambda r: r["val_macro_ap"], reverse=True)
    if rows:
        with open(OUT / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    for r in rows:
        print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    if rows:
        best = {k: rows[0][k] for k in jobs[0]}
        print(f"\nbest by val_macro_ap: {best}\nCopy these values into the species section of config_training.yaml.")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print the planned runs and exit")
    ap.add_argument("--epochs", type=int, default=SWEEP["epochs"])
    ap.add_argument("--no-telegram", action="store_true", help="do not start the Telegram monitor")
    ap.add_argument("--limit", type=int, default=0, help="cap images per split (smoke test)")
    a = ap.parse_args()
    jobs = combos(SWEEP["grid"])
    if a.dry_run:
        for j in jobs:
            print(" ".join(command(j, SWEEP["devices"][0], a.epochs, a.limit)))
        return
    run_all(jobs, a.epochs, a.limit, not a.no_telegram)
    rows = summarize(jobs)
    if rows and not a.no_telegram:
        telegram_monitor.send("Sweep ranking (val macro-AP):\n" + "\n".join(
            f"{i + 1}. " + " ".join(f"{k}={v:.3g}" for k, v in r.items() if k in jobs[0]) + f"  val={r['val_macro_ap']:.4f}"
            for i, r in enumerate(rows[:5])))


if __name__ == "__main__":
    main()
