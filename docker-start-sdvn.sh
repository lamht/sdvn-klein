#!/usr/bin/env bash

set -Eeuo pipefail

APP_ROOT="/app"
SDVN_ROOT="/app/SDVN-training-colab-flux"
AITK_ROOT="/app/SDVN-training-colab-flux/ai-toolkit"

DATA_ROOT="/data"
MODEL_ROOT="/data/models"
DATASET_ROOT="/data/datasets"
OUTPUT_ROOT="/data/outputs"
LOG_ROOT="/data/logs"
HF_HOME="/data/hf-cache"

MODEL_REPO="black-forest-labs/FLUX.2-klein-base-9b"

STARTUP_LOG="${LOG_ROOT}/startup.log"
MODEL_LOG="${LOG_ROOT}/model-download.log"

mkdir -p \
    "${DATA_ROOT}" \
    "${MODEL_ROOT}" \
    "${DATASET_ROOT}" \
    "${OUTPUT_ROOT}" \
    "${LOG_ROOT}" \
    "${HF_HOME}"

exec > >(tee -a "${STARTUP_LOG}") 2>&1

echo
echo "============================================================"
echo " SDVN TRAINING - FLUX.2 KLEIN"
echo "============================================================"
echo

echo "[INFO] Date:"
date

echo
echo "[INFO] Python:"
python --version

echo
echo "[INFO] Python path:"
which python

echo
echo "[INFO] Node:"
node --version

echo
echo "[INFO] npm:"
npm --version

echo
echo "[INFO] NVIDIA:"
nvidia-smi || true

echo
echo "[INFO] PyTorch:"

python - <<'PY'
import torch

print("torch =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU =", torch.cuda.get_device_name(0))
    print(
        "VRAM =",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )
PY

# ==========================================================
# MODEL DOWNLOAD - BACKGROUND, AFTER THE GUI RESPONDS
# ==========================================================

download_model() {
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "[MODEL] HF_TOKEN is not set; skipping automatic download."
        return
    fi

    echo "[MODEL] Started:"
    date
    echo "[MODEL] Repository: ${MODEL_REPO}"
    echo "[MODEL] Hugging Face cache: ${HF_HOME}"

    python - <<'PY'
import os
from huggingface_hub import login, snapshot_download

token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN is empty")

login(token=token, add_to_git_credential=False)
print("[MODEL] Authenticated with Hugging Face.")
models = [
    (
        "Flux 2 Klein transformer",
        "black-forest-labs/FLUX.2-klein-base-9b",
        ["flux-2-klein-base-9b.safetensors"],
    ),
    ("Qwen3 text encoder", "Qwen/Qwen3-8B", None),
    ("Flux 2 Klein VAE", "ai-toolkit/flux2_vae", ["ae.safetensors"]),
]

for name, repo, allow_patterns in models:
    print(f"[MODEL] Prefetching {name}: {repo}", flush=True)
    snapshot_download(
        repo_id=repo,
        token=token,
        allow_patterns=allow_patterns,
    )
    print(f"[MODEL] Cached {name}.", flush=True)
PY

    echo
    echo "[MODEL] Finished:"
    date
}

start_model_download_after_gui() {
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "[MODEL] HF_TOKEN is not set; skipping automatic download."
        return
    fi

    (
        set -Eeuo pipefail
        echo "[MODEL] Waiting for the GUI to respond before downloading."
        until curl --fail --silent --show-error --max-time 2 \
            http://127.0.0.1:8675/ >/dev/null 2>&1
        do
            sleep 2
        done

        echo "[MODEL] GUI is responding; starting background prefetch."
        download_model
    ) >> "${MODEL_LOG}" 2>&1 &

    MODEL_PID=$!
    echo "[MODEL] GUI-ready download watcher PID: ${MODEL_PID}"
    echo "[MODEL] Log: ${MODEL_LOG}"
}

# ==========================================================
# DATABASE
# ==========================================================

echo
echo "[PRISMA] Initializing database..."

cd "${AITK_ROOT}/ui"

# Prisma database path from schema.prisma:
#
# file:../../aitk_db.db
#
# => /app/SDVN-training-colab-flux/aitk_db.db

DB="${AITK_ROOT}/aitk_db.db"

if [ -d "${DB}" ]; then
    echo "[PRISMA] WARNING: aitk_db.db is a directory."
    echo "[PRISMA] Removing invalid directory..."
    rm -rf "${DB}"
fi

if [ ! -f "${DB}" ]; then
    echo "[PRISMA] Creating SQLite database..."
    touch "${DB}"
fi

echo "[PRISMA] prisma generate..."
npx prisma generate

echo "[PRISMA] prisma db push..."
npx prisma db push

echo "[PRISMA] Database ready."

# ==========================================================
# GUI
# ==========================================================

echo
echo "============================================================"
echo " STARTING AI TOOLKIT GUI"
echo "============================================================"
echo
echo "URL:"
echo "http://0.0.0.0:8675"
echo

if [ -n "${AI_TOOLKIT_AUTH:-}" ]; then
    echo "[AUTH] AI_TOOLKIT_AUTH enabled."
else
    echo "[AUTH] WARNING: AI_TOOLKIT_AUTH is not set."
fi

/app/fix_torch_compile.sh || true

cd "${AITK_ROOT}/ui"
export TORCHDYNAMO_VERBOSE=1
export TORCH_LOGS="+dynamo"
# Do not use build_and_start here:
# dependencies/build were already completed in Docker image.
#
# Start worker + Next.js UI.
start_model_download_after_gui
exec npm run start
