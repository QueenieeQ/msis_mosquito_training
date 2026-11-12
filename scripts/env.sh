# Sourced by run.sh and scripts/*.sh: sets ROOT (project folder), VENV and PY (the uv environment's python).
# VENV order: $VENV if set -> the shared /projects/mosquitto_msis/code/.venv if it exists -> <project>/.venv (made by setup).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -z "${VENV:-}" ]; then
  if [ -x /projects/mosquitto_msis/code/.venv/bin/python ]; then VENV=/projects/mosquitto_msis/code/.venv; else VENV="$ROOT/.venv"; fi
fi
PY="$VENV/bin/python"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
