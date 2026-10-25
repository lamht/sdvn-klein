#!/usr/bin/env bash
set -Eeuo pipefail

INSTALL_DIR="${INSTALL_DIR:-$HOME/SDVN-training-colab-flux}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/sdvn-klein}"
DATA_ROOT="${DATA_ROOT:-$HOME/sdvn-data}"
MODEL_ROOT="${MODEL_ROOT:-$DATA_ROOT/models}"
DATASET_ROOT="${DATASET_ROOT:-$DATA_ROOT/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/outputs}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/logs}"
HF_HOME="${HF_HOME:-$DATA_ROOT/hf-cache}"
PORT="${PORT:-8675}"

AITK_DIR="$INSTALL_DIR/ai-toolkit"
UI_DIR="$AITK_DIR/ui"
mkdir -p "$MODEL_ROOT" "$DATASET_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT" "$HF_HOME"
exec > >(tee -a "$LOG_ROOT/startup.log") 2>&1

echo "=== SDVN / AI Toolkit native startup ==="
date
[[ -x "$VENV_DIR/bin/python" ]] || { echo "Missing venv: $VENV_DIR; run install_sdvn.sh first."; exit 1; }
[[ -d "$UI_DIR" ]] || { echo "Missing AI Toolkit UI: $UI_DIR"; exit 1; }
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "--- NVIDIA ---"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "WARNING: nvidia-smi not found"; fi
echo "--- PyTorch CUDA runtime ---"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    print("WARNING: PyTorch cannot see a CUDA GPU. Check NVIDIA driver and host setup.")
PY

cd "$UI_DIR"
echo "--- Prisma ---"
npx --yes prisma generate -y
touch "$AITK_DIR/aitk_db.db"
npx --yes prisma db push

export NODE_ENV=production
export PORT
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCHDYNAMO_VERBOSE="${TORCHDYNAMO_VERBOSE:-1}"
# Avoid enabling very verbose TORCH_LOGS by default; it can create huge logs.
if [[ -n "${AI_TOOLKIT_AUTH:-}" ]]; then
  echo "AI_TOOLKIT_AUTH is configured."
else
  echo "WARNING: AI_TOOLKIT_AUTH is not set. Do not expose this UI directly to the public internet."
fi

echo "UI should be available at http://127.0.0.1:${PORT}"
echo "For LAN access, ensure the app binds to 0.0.0.0 in the project's own start command."
exec npm run start -- --hostname 0.0.0.0 --port "$PORT"
