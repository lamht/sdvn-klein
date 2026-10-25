#!/usr/bin/env bash
set -Eeuo pipefail

# Native (non-Docker) installer for SDVN-training-colab-flux + AI Toolkit.
# Target: Ubuntu 22.04/24.04 with an NVIDIA driver compatible with CUDA 12.8.
# Usage:
#   chmod +x install_sdvn.sh
#   ./install_sdvn.sh
# Optional environment variables:
#   INSTALL_DIR="$HOME/SDVN-training-colab-flux"
#   VENV_DIR="$HOME/.venvs/sdvn-klein"
#   TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

INSTALL_DIR="${INSTALL_DIR:-$HOME/SDVN-training-colab-flux}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/sdvn-klein}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "This script supports Linux only."
command -v apt-get >/dev/null 2>&1 || die "apt-get not found; this installer expects Ubuntu/Debian."

log "Installing OS packages (sudo may ask for your password)"
sudo apt-get update
sudo apt-get install -y \
  ca-certificates curl git git-lfs wget aria2 build-essential pkg-config \
  ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
  libsndfile1 python3 python3-dev python3-venv python3-pip sqlite3

git lfs install || true

# Install Node.js 22 if it is not already installed.
if ! command -v node >/dev/null 2>&1 || [[ "$(node -p 'Number(process.versions.node.split(".")[0])')" -lt 22 ]]; then
  log "Installing Node.js 22"
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi

log "Checking host NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[WARNING] nvidia-smi not found. Install a suitable NVIDIA driver and reboot before GPU training."
fi

log "Creating Python virtual environment at: ${VENV_DIR}"
mkdir -p "$(dirname "$VENV_DIR")"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ -e "$INSTALL_DIR/.git" ]]; then
  log "Repository already exists; updating it"
  git -C "$INSTALL_DIR" pull --ff-only || echo "[WARNING] Git pull failed; using existing checkout."
else
  log "Cloning SDVN repository"
  mkdir -p "$(dirname "$INSTALL_DIR")"
  git clone --recurse-submodules https://github.com/StableDiffusionVN/SDVN-training-colab-flux.git "$INSTALL_DIR"
fi

log "Initializing submodules"
git -C "$INSTALL_DIR" submodule update --init --recursive
AITK_DIR="$INSTALL_DIR/ai-toolkit"
[[ -f "$INSTALL_DIR/requirements.txt" ]] || die "Missing $INSTALL_DIR/requirements.txt"
[[ -f "$AITK_DIR/requirements.txt" ]] || die "Missing $AITK_DIR/requirements.txt; check submodule checkout."

log "Installing SDVN and AI Toolkit Python requirements"
python -m pip install --no-cache-dir -r "$INSTALL_DIR/requirements.txt"
python -m pip install --no-cache-dir -r "$AITK_DIR/requirements.txt"

log "Installing pinned PyTorch CUDA 12.8 wheels (after project requirements to prevent replacement)"
python -m pip install --no-cache-dir --force-reinstall \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url "$TORCH_INDEX_URL"

log "Installing Hugging Face download support"
python -m pip install --no-cache-dir -U huggingface_hub hf_transfer

log "Installing/building AI Toolkit UI dependencies"
UI_DIR="$AITK_DIR/ui"
[[ -f "$UI_DIR/package.json" ]] || die "Missing UI package.json at $UI_DIR"
cd "$UI_DIR"
npm install
npx prisma generate
# The schema uses a relative SQLite path; this creates the expected file if absent.
touch "$AITK_DIR/aitk_db.db"
npx prisma db push
npm run build

log "Creating persistent data directories"
mkdir -p "$HOME/sdvn-data"/{models,datasets,outputs,logs,hf-cache}
cat <<EOF

============================================================
Installation finished.
============================================================
Python environment: ${VENV_DIR}
Repository:         ${INSTALL_DIR}
AI Toolkit UI:      ${UI_DIR}
Persistent data:    $HOME/sdvn-data/{models,datasets,outputs,logs,hf-cache}

Next:
  1. Copy start-sdvn.sh from this setup bundle to a convenient location,
     or run it from the extracted setup folder.
  2. Activate the environment with:
       source "${VENV_DIR}/bin/activate"
  3. Start the UI with:
       INSTALL_DIR="${INSTALL_DIR}" VENV_DIR="${VENV_DIR}" ./start-sdvn.sh

Note: GPU access is checked at runtime, not during installation.
EOF
