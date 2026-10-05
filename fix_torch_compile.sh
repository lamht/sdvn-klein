#!/usr/bin/env bash
set -e

ROOT="/app/SDVN-training-colab-flux/ai-toolkit"

echo "========================================"
echo " AI Toolkit - Disable torch.compile"
echo "========================================"

cd "$ROOT"

# --------------------------------------------------
# 1. Backup source
# --------------------------------------------------
BACKUP="$ROOT/.backup_before_disable_compile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"

cp -a jobs/process/BaseSDTrainProcess.py "$BACKUP/"
cp -a toolkit/config_modules.py "$BACKUP/"

echo "[OK] Backup: $BACKUP"

# --------------------------------------------------
# 2. Disable compile in existing job config
# --------------------------------------------------
for CFG in "$ROOT"/output/*/config.yaml "$ROOT"/output/*/config.yml; do
    [ -f "$CFG" ] || continue

    echo "[INFO] Patching config: $CFG"

    sed -i -E \
        's/^([[:space:]]*)compile:[[:space:]]*true/\1compile: false/g' \
        "$CFG"

    sed -i -E \
        's/^([[:space:]]*)block_compile:[[:space:]]*true/\1block_compile: false/g' \
        "$CFG"
done

# --------------------------------------------------
# 3. Disable automatic block compile
# --------------------------------------------------
python3 - <<'PY'
from pathlib import Path

root = Path("/app/SDVN-training-colab-flux/ai-toolkit")
f = root / "jobs/process/BaseSDTrainProcess.py"

text = f.read_text()

old = text

# Disable calls of torch.compile used by block compilation.
# Keep the surrounding training/checkpoint code intact.
text = text.replace(
    "block_list[i] = torch.compile(",
    "block_list[i] = torch._dynamo.disable("
)

# Disable whole-model compile fallback.
text = text.replace(
    "self.sd.unet = torch.compile(",
    "self.sd.unet = torch._dynamo.disable("
)

if text != old:
    f.write_text(text)
    print("[OK] Disabled compile calls in BaseSDTrainProcess.py")
else:
    print("[INFO] No matching BaseSDTrainProcess.py compile calls changed")
PY

# --------------------------------------------------
# 4. Disable quantized-model automatic compile permission
# --------------------------------------------------
python3 - <<'PY'
from pathlib import Path

f = Path("/app/SDVN-training-colab-flux/ai-toolkit/toolkit/config_modules.py")
text = f.read_text()
old = text

# The AI Toolkit can automatically allow torch.compile for
# quantized models. Disable that automatic behavior.
text = text.replace(
    'print("Quantized model detected - allowing torch.compile (experimental)")',
    'print("Quantized model detected - torch.compile DISABLED for compatibility")'
)

if text != old:
    f.write_text(text)
    print("[OK] Disabled quantized compile message/auto path")
else:
    print("[INFO] No quantized compile string changed")
PY

# --------------------------------------------------
# 5. Disable Dynamo debug logging
# --------------------------------------------------
unset TORCHDYNAMO_VERBOSE
unset TORCH_LOGS
unset TORCH_COMPILE_DISABLE

# Set Dynamo errors to fall back instead of crashing where possible.
export TORCHDYNAMO_SUPPRESS_ERRORS=1

# --------------------------------------------------
# 6. Verify
# --------------------------------------------------
echo
echo "========================================"
echo " CONFIG"
echo "========================================"

grep -RniE '^[[:space:]]*(compile|block_compile):' \
    "$ROOT/output" \
    --include='config.yaml' \
    --include='config.yml' \
    2>/dev/null || true

echo
echo "========================================"
echo " REMAINING torch.compile CALLS"
echo "========================================"

grep -RniE 'torch\.compile[[:space:]]*\(' \
    "$ROOT" \
    --include='*.py' \
    --exclude-dir=ui \
    --exclude-dir=node_modules \
    --exclude-dir=.git \
    2>/dev/null || true

echo
echo "========================================"
echo " DONE"
echo "========================================"
echo
echo "Gradient checkpointing was NOT disabled."
echo
echo "Backup:"
echo "  $BACKUP"
echo
