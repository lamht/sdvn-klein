
#!/usr/bin/env bash
set -Eeuo pipefail

# =========================================================
# SDVN Training / AI Toolkit startup
# =========================================================

INSTALL_DIR="${INSTALL_DIR:-$HOME/SDVN-training-colab-flux}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/sdvn-klein}"
DATA_ROOT="${DATA_ROOT:-$HOME/sdvn-data}"
PORT="${PORT:-8675}"

# Automatically locate the project if the default is incorrect.
if [[ ! -f "$INSTALL_DIR/ai-toolkit/ui/package.json" ]]; then
  for candidate in \
    "$HOME/SDVN-training-colab-flux" \
    "/root/SDVN-training-colab-flux" \
    "/content/SDVN-training-colab-flux"
  do
    if [[ -f "$candidate/ai-toolkit/ui/package.json" ]]; then
      INSTALL_DIR="$candidate"
      break
    fi
  done
fi

AITK_DIR="$INSTALL_DIR/ai-toolkit"
UI_DIR="$AITK_DIR/ui"

MODEL_ROOT="$DATA_ROOT/models"
DATASET_ROOT="$DATA_ROOT/datasets"
OUTPUT_ROOT="$DATA_ROOT/outputs"
LOG_ROOT="$DATA_ROOT/logs"
HF_HOME="${HF_HOME:-$DATA_ROOT/hf-cache}"

mkdir -p \
  "$MODEL_ROOT" "$DATASET_ROOT" "$OUTPUT_ROOT" \
  "$LOG_ROOT" "$HF_HOME"

exec > >(tee -a "$LOG_ROOT/startup.log") 2>&1

echo "=== SDVN / AI Toolkit startup ==="
date
echo "INSTALL_DIR=$INSTALL_DIR"
echo "AITK_DIR=$AITK_DIR"
echo "UI_DIR=$UI_DIR"
echo "VENV_DIR=$VENV_DIR"
echo "PORT=$PORT"

# Validate paths before doing anything destructive.
if [[ ! -f "$UI_DIR/package.json" ]]; then
  echo "ERROR: UI package.json not found."
  echo "Set INSTALL_DIR to the directory containing ai-toolkit/ui."
  echo "Example: INSTALL_DIR=/root/SDVN-training-colab-flux"
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: Python virtual environment not found: $VENV_DIR"
  echo "Run your installation script first or set VENV_DIR correctly."
  exit 1
fi

# Activate Python environment.
source "$VENV_DIR/bin/activate"

echo "=== NVIDIA GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found"
fi

python - <<'PY'
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("PyTorch check failed:", repr(e))
PY

cd "$UI_DIR"

echo "=== Node.js / npm ==="
node --version
npm --version

echo "=== Package scripts ==="
node -e 'console.log(JSON.stringify(require("./package.json").scripts, null, 2))'

echo "=== Install dependencies ==="
npm install

# Prisma setup. Preserve the existing database.
cd "$AITK_DIR"
touch aitk_db.db
cd "$UI_DIR"

npm install --no-save --no-package-lock prisma@6.19.0

if [[ -f "$UI_DIR/prisma/schema.prisma" ]]; then
  npx --yes prisma db push --schema "$UI_DIR/prisma/schema.prisma"
elif [[ -f "$AITK_DIR/prisma/schema.prisma" ]]; then
  npx --yes prisma db push --schema "$AITK_DIR/prisma/schema.prisma"
else
  echo "WARNING: Prisma schema not found in expected locations."
  echo "Check the repository's database setup instructions."
fi

echo "=== Update database ==="
npm run update_db

echo "=== Build production UI ==="
npm run build

echo "=== Verify Next.js build ==="
if [[ ! -f "$UI_DIR/.next/BUILD_ID" ]]; then
  echo "ERROR: Next.js production build is missing."
  echo "Inspect the build errors above."
  exit 1
fi

echo "=== Verify Worker build ==="
if [[ ! -f "$UI_DIR/dist/cron/worker.js" ]]; then
  echo "ERROR: Worker not found at dist/cron/worker.js"
  echo "Searching for generated Worker files:"
  find "$UI_DIR" -path "$UI_DIR/node_modules" -prune -o \
    -type f -name 'worker.js' -print || true
  echo "Check package.json scripts and Worker TypeScript configuration."
  exit 1
fi

export NODE_ENV=production
export PORT
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TORCHDYNAMO_VERBOSE="${TORCHDYNAMO_VERBOSE:-1}"

if [[ -z "${AI_TOOLKIT_AUTH:-}" ]]; then
  echo "WARNING: AI_TOOLKIT_AUTH is unset."
  echo "Do not expose this UI directly to the public internet."
fi

# Prevent accidentally starting duplicate instances.
if command -v pgrep >/dev/null 2>&1 &&
   pgrep -af 'next start.*8675' | grep -v "$$" >/dev/null; then
  echo "WARNING: A process using next start on port 8675 may already exist."
  echo "Stop the previous instance before restarting."
  exit 1
fi

mkdir -p "$LOG_ROOT"

echo "=== Starting AI Toolkit UI ==="
nohup npm run start -- --hostname 0.0.0.0 --port "$PORT" \
  > "$LOG_ROOT/app.log" 2>&1 &

APP_PID=$!
echo "$APP_PID" > "$LOG_ROOT/app.pid"

sleep 3

if kill -0 "$APP_PID" 2>/dev/null; then
  echo "Startup process launched. PID=$APP_PID"
  echo "Local URL: http://127.0.0.1:$PORT"
  echo "App log: $LOG_ROOT/app.log"
else
  echo "ERROR: App process exited. Last log output:"
  tail -100 "$LOG_ROOT/app.log" || true
  exit 1
fi
