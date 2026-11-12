#!/usr/bin/env bash
# One-click setup of a machine for this project: checks every requirement and installs what is missing.
#   ./run.sh setup                run everything below (asks for sudo / chsh passwords when needed)
#   ./run.sh setup --check        only report what is installed / missing, change nothing
# Options: -y|--yes (no questions)  --no-system (skip apt/dnf/brew)  --no-shell (skip zsh + Oh My Zsh)  -h
# Steps: system tools (git curl zsh gcc make unzip + OpenCV libs) -> uv -> Oh My Zsh (+ autosuggestions, syntax
# highlighting) -> git identity / ssh key -> GPU driver -> Python 3.12 + venv + config/requirements.txt (torch wheels
# matched to the driver) -> package check -> unit tests -> project data. Safe to re-run: finished steps are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."; . scripts/env.sh

CHECK=0 YES=0 SYSTEM=1 SHELLSETUP=1
for a in "$@"; do
  case $a in --check) CHECK=1 ;; -y|--yes) YES=1 ;; --no-system) SYSTEM=0 ;; --no-shell) SHELLSETUP=0 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; *) echo "unknown option $a (try -h)"; exit 1 ;; esac
done
G=$'\e[32m' RED=$'\e[31m' Y=$'\e[33m' B=$'\e[1m' R=$'\e[0m'
FAILED=() WARN=()
good() { printf "  ${G}✔${R} %s\n" "$1"; }
bad()  { printf "  ${RED}✘${R} %s\n" "$1"; FAILED+=("$1"); }
warn() { printf "  ${Y}!${R} %s\n" "$1"; WARN+=("$1"); }
head_() { printf "\n${B}%s${R}\n" "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
confirm() { [ $YES -eq 1 ] && return 0; [ -t 0 ] || return 1; local a; read -rp "  $1 [y/N] " a; [[ $a == [yY] ]]; }
[ "$(id -u)" -eq 0 ] && SUDO="" || SUDO="sudo"
export PATH="$HOME/.local/bin:$PATH"

step() {  # step "label" check_cmd install_fn: skip if the check passes, otherwise install (unless --check) and re-check
  local label=$1 chk=$2 inst=$3 fail=bad; [ "${4:-}" = soft ] && fail=warn   # soft: optional, a miss is only a warning
  if eval "$chk" >/dev/null 2>&1; then good "$label"; return 0; fi
  if [ $CHECK -eq 1 ]; then $fail "$label: missing"; return 1; fi
  printf "  … installing %s\n" "$label"
  "$inst" || true
  if eval "$chk" >/dev/null 2>&1; then good "$label (installed)"; else $fail "$label: install failed"; fi
}

# ---------------------------------------------------------------- system packages
pkg_manager() { for m in apt-get dnf brew; do have $m && { echo $m; return; }; done; }
libgl_ok() { [ "$(uname)" = Darwin ] || { ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null; } | grep -q "libGL.so.1"; }
install_system() {
  local pm; pm=$(pkg_manager)
  case $pm in
    apt-get) $SUDO apt-get update -y && $SUDO apt-get install -y git curl zsh build-essential unzip ca-certificates libgl1 libglib2.0-0 ;;
    dnf) $SUDO dnf install -y git curl zsh gcc gcc-c++ make unzip ca-certificates mesa-libGL glib2 ;;
    brew) brew install git curl zsh unzip ;;
    *) echo "  no supported package manager (apt-get, dnf, brew): install git curl zsh gcc make unzip by hand"; return 1 ;;
  esac
}
system_ok() { for c in git curl zsh gcc make unzip; do have $c || return 1; done; [ "$(uname)" = Darwin ] || libgl_ok; }

# ---------------------------------------------------------------- uv / shell / git
install_uv() { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
omz_ok() { [ -d "${ZSH:-$HOME/.oh-my-zsh}" ]; }
install_omz() { RUNZSH=no CHSH=no KEEP_ZSHRC=yes sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended; }
plugin_ok() { [ -d "${ZSH_CUSTOM:-${ZSH:-$HOME/.oh-my-zsh}/custom}/plugins/$1" ]; }
install_plugin() { git clone --depth 1 "https://github.com/zsh-users/$1" "${ZSH_CUSTOM:-${ZSH:-$HOME/.oh-my-zsh}/custom}/plugins/$1"; }
enable_plugins() {  # add the two plugins to the plugins=(...) line of ~/.zshrc (backup kept), only if absent
  local rc=$HOME/.zshrc; [ -f "$rc" ] || return 0
  for p in zsh-autosuggestions zsh-syntax-highlighting; do
    grep -Eq "^plugins=\(.*\b$p\b" "$rc" && continue
    [ -f "$rc.bak-setup" ] || cp "$rc" "$rc.bak-setup"
    sed -i -E "s/^plugins=\((.*)\)/plugins=(\1 $p)/" "$rc"
  done
}
git_identity() {
  local n e; n=$(git config --global user.name); e=$(git config --global user.email)
  if [ -n "$n" ] && [ -n "$e" ]; then good "git identity: $n <$e>"; return; fi
  if [ $CHECK -eq 0 ] && [ -t 0 ] && [ $YES -eq 0 ]; then
    read -rp "  git user.name: " n; read -rp "  git user.email: " e
    [ -n "$n" ] && [ -n "$e" ] && { git config --global user.name "$n"; git config --global user.email "$e"; good "git identity set"; return; }
  fi
  warn "git identity not set: git config --global user.name 'Your Name'; git config --global user.email you@example.com"
}

# ---------------------------------------------------------------- python environment
torch_backend() { if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then echo auto; else echo cpu; fi; }
install_python_env() {
  uv python install 3.12 || return 1
  [ -x "$PY" ] || uv venv --python 3.12 "$VENV" || return 1
  uv pip install --python "$PY" --torch-backend "$(torch_backend)" -r config/requirements.txt
}
python_ok() { [ -x "$PY" ] && "$PY" src/check_env.py >/dev/null 2>&1; }

printf "${B}Mosquito training: machine setup${R}  (project %s)\n" "$ROOT"
[ $CHECK -eq 1 ] && printf "check only, nothing will be changed\n"

head_ "1. System tools"
if [ $SYSTEM -eq 1 ]; then step "git curl zsh gcc make unzip + OpenCV libs" system_ok install_system
else warn "system packages skipped (--no-system)"; fi

head_ "2. uv (Python package manager)"
step "uv $(uv --version 2>/dev/null | cut -d' ' -f2)" "have uv" install_uv

head_ "3. Shell: zsh + Oh My Zsh"
if [ $SHELLSETUP -eq 1 ]; then
  step "Oh My Zsh" omz_ok install_omz
  for p in zsh-autosuggestions zsh-syntax-highlighting; do step "plugin $p" "plugin_ok $p" "install_plugin $p" soft; done
  [ $CHECK -eq 0 ] && omz_ok && enable_plugins
  if [ "$(basename "${SHELL:-}")" = zsh ]; then good "login shell is zsh"
  elif have zsh && [ $CHECK -eq 0 ] && confirm "Make zsh your login shell?"; then chsh -s "$(command -v zsh)" && good "login shell set to zsh" || warn "chsh failed"
  else warn "login shell is ${SHELL:-unknown}, not zsh (chsh -s \$(command -v zsh))"; fi
else warn "shell setup skipped (--no-shell)"; fi

head_ "4. Git"
git_identity
ls ~/.ssh/id_* >/dev/null 2>&1 && good "ssh key present" || warn "no ssh key (ssh-keygen -t ed25519 -C you@example.com), needed for git over ssh"

head_ "5. GPU"
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  good "NVIDIA driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1): $(nvidia-smi -L | wc -l) GPU(s)"
else warn "no NVIDIA GPU / driver found: CPU-only torch is installed, training will be very slow (drivers cannot be installed automatically)"; fi

head_ "6. Python 3.12, venv ($VENV), packages"
if have uv; then step "packages from config/requirements.txt" python_ok install_python_env; else bad "uv missing: packages not installed"; fi
[ -x "$PY" ] && { "$PY" src/check_env.py | tail -2 | sed 's/^/    /'; }

head_ "7. Project"
if [ -x "$PY" ] && python_ok; then
  "$PY" -m pytest tests -q >/tmp/bootstrap_pytest.log 2>&1 && good "unit tests pass ($(tail -1 /tmp/bootstrap_pytest.log))" || bad "unit tests fail (see /tmp/bootstrap_pytest.log)"
  if [ -f input/datasets/full_body_detection/data.yaml ]; then "$PY" src/dataset.py >/dev/null 2>&1 && good "training data in input/datasets is valid" || bad "input/datasets has errors: run ./run.sh -> Dataset"
  else warn "no training data in input/datasets yet (see docs/how_to_train.md, or ./run.sh -> Dataset -> import)"; fi
fi
[ -f config/telegram.env ] && good "Telegram configured" || warn "optional: cp config/telegram.env.example config/telegram.env (progress messages)"
[ -d "$(cd "$ROOT" && "$PY" -c 'import common;print(common._path(common.CFG["compare"]["old_infer_dir"]))' 2>/dev/null)" ] \
  && good "old pipeline found (for comparisons)" || warn "old pipeline not found: set compare.old_infer_dir in config/config_training.yaml to compare with it"

printf "\n${B}Summary${R}: ${G}%d warning(s)${R}, " ${#WARN[@]}
[ ${#FAILED[@]} -eq 0 ] && printf "${G}no failures. Ready: ./run.sh${R}\n" || { printf "${RED}%d failure(s):${R}\n" ${#FAILED[@]}; printf '  - %s\n' "${FAILED[@]}"; }
exit $((${#FAILED[@]} > 0))
