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

COPY docker-start-sdvn.sh /usr/local/bin/start-sdvn.sh
RUN chmod +x /usr/local/bin/start-sdvn.sh

# =========================================================
# ENVIRONMENT
# =========================================================

ENV NODE_ENV=production
ENV PORT=8675

EXPOSE 8675

WORKDIR ${AITK_ROOT}/ui

ENTRYPOINT ["/usr/local/bin/start-sdvn.sh"]