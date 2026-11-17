"""Telegram monitor for run_training.sh: pushes start/finish/fail messages and answers /status and /compare (old-vs-new classifier results).

Usage (run_training.sh does this for you):
    telegram_monitor.py watch [--title T] full obb species   # background: notify + answer commands until all tasks end
    telegram_monitor.py send "text"              # one-off message
One-time setup: open the bot in Telegram and send /msis2026; that chat is saved to config/telegram.env
and is the only one answered afterwards. Task state is read from <runs>/<task>.status written by the
shell script (pending | running | ok | failed) and progress from the tail of <runs>/<task>.log.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import common

ENV = common.ROOT / "config" / "telegram.env"
TCFG = common.CFG.get("telegram", {})
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def env():
    out = {k: v for k, v in (l.split("=", 1) for l in ENV.read_text().splitlines()
                              if "=" in l and not l.startswith("#"))} if ENV.exists() else {}
    for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        out[k] = os.environ.get(k, out.get(k, "")).strip()
    return out


def save_chat(chat_id):
    lines = [l for l in ENV.read_text().splitlines() if not l.startswith("TELEGRAM_CHAT_ID")] if ENV.exists() else []
    ENV.write_text("\n".join(lines + [f"TELEGRAM_CHAT_ID={chat_id}"]) + "\n")


def api(method, token, http_timeout=30, **params):
    url = f"https://api.telegram.org/bot{token}/{method}"
    req = urllib.request.Request(url, urllib.parse.urlencode(params).encode())
    with urllib.request.urlopen(req, timeout=http_timeout) as r:
        return json.load(r)["result"]


def send(text, chat_id=None):
    e = env()
    chat_id = chat_id or e["TELEGRAM_CHAT_ID"]
    if not (e["TELEGRAM_BOT_TOKEN"] and chat_id):
        return False
    try:
        api("sendMessage", e["TELEGRAM_BOT_TOKEN"], chat_id=chat_id, text=text[:4000])
        return True
    except Exception as ex:  # monitoring must never break training
        print(f"telegram send failed: {ex}", file=sys.stderr)
        return False


def state(task):
    f = common.RUNS / f"{task}.status"
    return f.read_text().strip() if f.exists() else "pending"


def progress(task):
    """Last meaningful log line (progress bars use \\r, so split on it)."""
    f = common.RUNS / f"{task}.log"
    if not f.exists():
        return ""
    tail = ANSI.sub("", f.read_bytes()[-4000:].decode(errors="replace"))
    lines = [l.strip() for l in re.split(r"[\r\n]", tail) if l.strip()]
    return lines[-1][:200] if lines else ""


def gpus():
    """One line per GPU: utilisation, memory used / total, temperature."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        rows = [[x.strip() for x in l.split(",")] for l in out.splitlines() if l.strip()]
        return "GPUs\n" + "\n".join(f"  {i}: {u}% | {int(m) / 1024:.1f}/{int(t) / 1024:.0f} GB | {c} C"
                                     for i, u, m, t, c in rows)
    except Exception:
        return ""


def elapsed(task):
    """Minutes since the status file last changed (= since the task entered its current state)."""
    f = common.RUNS / f"{task}.status"
    return int((time.time() - f.stat().st_mtime) / 60) if f.exists() else 0


def report(tasks, title="Training"):
    icon = {"pending": "⏳", "running": "🏃", "ok": "✅", "failed": "❌"}
    counts = {k: sum(state(t) == k for t in tasks) for k in icon}
    rows = [f"{title}: " + ", ".join(f"{n} {k}" for k, n in counts.items() if n)]
    for t in tasks:
        s = state(t)
        rows.append(f"{icon.get(s, '?')} {t}: {s}" + (f" ({elapsed(t)} min)" if s == "running" else "")
                    + (f"\n    {progress(t)}" if s in ("running", "failed") else ""))
    return "\n".join(rows + ["", gpus()]).strip()


def compare(run=None):
    """Old-vs-new classifier summary from <run>/compare_old.json (default: the most recent comparison)."""
    files = [common.RUNS / run / "compare_old.json"] if run else sorted(common.RUNS.glob("*/compare_old.json"),
                                                                       key=lambda p: p.stat().st_mtime)[-1:]
    if not files or not files[0].exists():
        return "No comparison yet: it is written by compare_models.py after the species task finishes."
    f = files[0]
    d = json.loads(f.read_text())
    rows = [f"Old vs new classifier: run {f.parent.name} ({d['split']} split, thr {d['threshold']}, "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(f.stat().st_mtime))})"]
    for name, m in d["models"].items():
        a = m["classes"].get("Anopheles", {})
        rows.append(f"{name}\n    macro AP {m['macro_ap']:.3f} | F1 {m['macro_f1']:.3f} | exact {m['exact_match']:.3f}"
                    f"\n    Anopheles AP {a.get('ap', 0):.3f} F1 {a.get('f1', 0):.3f}")
    mf, focus = f.parent / "metrics.json", d.get("focus_ap") or {}  # focus_ap: scored on the current test split
    if not focus and mf.exists():
        r = json.loads(mf.read_text())
        focus = {k: r.get("test_focus_ap_" + k) for k in ("plain", "tta4")}
    for k, v in focus.items():
        if v:
            rows.append(f"new 14-species test AP ({k}): " + ", ".join(f"{n} {x:.3f}" for n, x in v.items()))
    rows.append("(old model has no Culex spp. class, so only Anopheles is comparable)")
    return "\n".join(rows)


def handle_updates(token, tasks, offset, title):
    for u in api("getUpdates", token, http_timeout=40, offset=offset, timeout=20):
        offset = u["update_id"] + 1
        m = u.get("message") or {}
        chat, text = str((m.get("chat") or {}).get("id", "")), (m.get("text") or "").split("@")[0].strip()
        if not chat:
            continue
        saved = env()["TELEGRAM_CHAT_ID"]
        if not saved and text == "/msis2026":
            save_chat(chat)
            send("Registered. Commands: /status, /compare", chat)
        elif chat == saved and text == "/status":
            send(report(tasks, title), chat)
        elif chat == saved and text == "/compare":
            send(compare(), chat)
    return offset


def watch(tasks, title="Training"):
    token = env()["TELEGRAM_BOT_TOKEN"]
    if not token:
        raise SystemExit("telegram: no token (config/telegram.env)")
    interval = float(TCFG.get("interval_min", 30)) * 60
    send(f"{title} started: " + ", ".join(tasks))
    seen, offset, last_push = {}, None, time.time()
    while True:
        try:
            offset = handle_updates(token, tasks, offset, title)
        except Exception as ex:
            print(f"telegram poll failed: {ex}", file=sys.stderr)
            time.sleep(10)
        for t in tasks:
            s = state(t)
            if seen.get(t, "pending") != s and s in ("running", "ok", "failed"):
                send(f"{t}: {s}" + (f"\n{progress(t)}" if s != "running" else ""))
            seen[t] = s
        if all(state(t) in ("ok", "failed") for t in tasks):
            send(f"{title} finished.\n" + report(tasks, title))
            return
        if interval > 0 and time.time() - last_push >= interval:
            send("Progress update\n" + report(tasks, title))
            last_push = time.time()


if __name__ == "__main__":
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "watch" and args:
        title = "Training"
        if args[0] == "--title":
            title, args = args[1], args[2:]
        watch(args, title)
    elif cmd == "compare":
        sys.exit(0 if send(compare(*args[:1])) else 1)
    elif cmd == "send" and args:
        sys.exit(0 if send(" ".join(args)) else 1)
    else:
        raise SystemExit(__doc__)
