#!/usr/bin/env bash
# Download Meta MMS-LID and identify the language of 16 kHz mono WAV files.
#
# Usage:   scripts/run_mms_lid.sh <audio.wav> [audio.wav ...]
# Env var: MMS_MODEL_ID  -- model variant (default: facebook/mms-lid-256)
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${MMS_MODEL_ID:-facebook/mms-lid-256}"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <audio.wav> [audio.wav ...]" >&2
    exit 1
fi

echo ">> Installing dependencies (mms)..."
uv sync --extra mms

echo ">> Downloading model '$MODEL_ID'..."
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID')"

echo ">> Identifying language..."
uv run mms-lid --model "$MODEL_ID" "$@"
