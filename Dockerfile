FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

ENV APP_ROOT=/app
ENV SDVN_ROOT=/app/SDVN-training-colab-flux
ENV AITK_ROOT=/app/SDVN-training-colab-flux/ai-toolkit

ENV DATA_ROOT=/data
ENV MODEL_ROOT=/data/models
ENV DATASET_ROOT=/data/datasets
ENV OUTPUT_ROOT=/data/outputs
ENV LOG_ROOT=/data/logs
ENV HF_HOME=/data/hf-cache
ENV HUGGINGFACE_HUB_CACHE=/data/hf-cache/hub

WORKDIR ${APP_ROOT}

# =========================================================
# SYSTEM PACKAGES
# =========================================================

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    wget \
    ca-certificates \
    build-essential \
    pkg-config \
    software-properties-common \
    ffmpeg \
    aria2 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libsndfile1 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# =========================================================
# python -> python3
# =========================================================

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && python --version \
    && python3 --version

# =========================================================
# NODE.JS 22
# =========================================================

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install -y nodejs \
    && node --version \
    && npm --version \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# PYTHON VENV
# =========================================================

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# =========================================================
# PYTORCH CUDA 12.8
#
# RTX 5060 Ti / Blackwell
# =========================================================

RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# =========================================================
# CLONE SDVN REPOSITORY
# =========================================================

RUN git clone \
    --recurse-submodules \
    https://github.com/StableDiffusionVN/SDVN-training-colab-flux.git \
    ${SDVN_ROOT}

COPY fix_torch_compile.sh /app/fix_torch_compile.sh
RUN chmod +x /app/fix_torch_compile.sh

WORKDIR ${SDVN_ROOT}

RUN git submodule update --init --recursive

# Make sure the ai-toolkit submodule exists
RUN test -f ${AITK_ROOT}/requirements.txt

# =========================================================
# INSTALL SDVN REQUIREMENTS
#
# This is the REAL requirements.txt from SDVN.
# It does not contain torch, so our CUDA 12.8 torch remains.
# =========================================================

RUN pip install --no-cache-dir \
    -r ${SDVN_ROOT}/requirements.txt

# =========================================================
# AI TOOLKIT PYTHON REQUIREMENTS
# =========================================================

RUN pip install --no-cache-dir \
    -r ${AITK_ROOT}/requirements.txt

# =========================================================
# HUGGING FACE
# =========================================================

RUN pip install --no-cache-dir \
    -U \
    huggingface_hub==1.10.1 \
    hf_transfer

# =========================================================
# DATA DIRECTORIES
# =========================================================

RUN mkdir -p \
    ${DATA_ROOT} \
    ${MODEL_ROOT} \
    ${DATASET_ROOT} \
    ${OUTPUT_ROOT} \
    ${LOG_ROOT} \
    ${HF_HOME}

# =========================================================
# CHECK PYTORCH / CUDA DURING BUILD
# =========================================================

RUN python - <<'PY'
import torch

print("=" * 60)
print("PyTorch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )

print("=" * 60)
PY

# =========================================================
# NODE / PRISMA
# =========================================================

WORKDIR ${AITK_ROOT}/ui

# Install frontend dependencies
RUN npm install

# =========================================================
# PRISMA
#
# Same commands used by AI Toolkit:
# npm run update_db
# = prisma generate + prisma db push
# =========================================================

RUN npx prisma generate

# Create initial DB during image build.
# Runtime entrypoint will run db push again.
RUN touch ${AITK_ROOT}/aitk_db.db

RUN npx prisma db push

# =========================================================
# BUILD WEB UI
# =========================================================

RUN npm run build

# =========================================================
# ENTRYPOINT
# =========================================================

RUN cat > /usr/local/bin/start-sdvn.sh <<'EOF'
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

MODEL_REPO="black-forest-labs/FLUX.2-klein-base-9b-fp8"
MODEL_FILE="flux-2-klein-base-9b-fp8.safetensors"

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
# MODEL DOWNLOAD - BACKGROUND
# ==========================================================

download_model() {

    echo
    echo "============================================================"
    echo " BACKGROUND MODEL DOWNLOAD"
    echo "============================================================"

    if [ -z "${HF_TOKEN:-}" ]; then
        echo "[MODEL] HF_TOKEN is not set."
        echo "[MODEL] Skipping automatic download."
        echo "[MODEL] GUI will continue starting."
        return 0
    fi

    (
        set -Eeuo pipefail

        echo "[MODEL] Started:"
        date

        echo "[MODEL] Repository:"
        echo "${MODEL_REPO}"

        echo "[MODEL] Destination:"
        echo "${MODEL_ROOT}"

        python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo = "black-forest-labs/FLUX.2-klein-base-9b-fp8"

local_dir = Path("/data/models/flux-2-klein-base-9b-fp8")
local_dir.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN")

if not token:
    raise RuntimeError("HF_TOKEN is empty")

print("[MODEL] Starting Hugging Face download...")
print("[MODEL] repo =", repo)
print("[MODEL] local_dir =", local_dir)

snapshot_download(
    repo_id=repo,
    token=token,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("[MODEL] Download complete.")
PY

        echo
        echo "[MODEL] Files:"
        find "${MODEL_ROOT}/flux-2-klein-base-9b-fp8" \
            -maxdepth 2 \
            -type f \
            -printf "%p %s bytes\n" \
            2>/dev/null || true

        echo
        echo "[MODEL] Finished:"
        date

    ) >> "${MODEL_LOG}" 2>&1 &

    MODEL_PID=$!

    echo "[MODEL] Background PID: ${MODEL_PID}"
    echo "[MODEL] Log: ${MODEL_LOG}"
}

download_model

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
exec npm run start
EOF

RUN chmod +x /usr/local/bin/start-sdvn.sh

# =========================================================
# ENVIRONMENT
# =========================================================

ENV NODE_ENV=production
ENV PORT=8675

EXPOSE 8675

WORKDIR ${AITK_ROOT}/ui

ENTRYPOINT ["/usr/local/bin/start-sdvn.sh"]