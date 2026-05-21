#!/usr/bin/env bash
# Download FireRedLID and identify the language of 16 kHz mono WAV files.
#
# Usage:   scripts/run_firered_lid.sh <audio.wav> [audio.wav ...]
# Env var: FIRERED_MODEL_DIR  -- where weights are stored (default: models/FireRedLID)
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_REPO="FireRedTeam/FireRedLID"
MODEL_DIR="${FIRERED_MODEL_DIR:-models/FireRedLID}"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <audio.wav> [audio.wav ...]" >&2
    exit 1
fi

echo ">> Installing dependencies (firered)..."
uv sync --extra firered

if [ ! -f "$MODEL_DIR/model.pth.tar" ]; then
    echo ">> Downloading model '$MODEL_REPO' to '$MODEL_DIR'..."
    uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_REPO', local_dir='$MODEL_DIR')"
else
    echo ">> Model already present at '$MODEL_DIR'."
fi

echo ">> Identifying language..."
uv run firered-lid --model-dir "$MODEL_DIR" "$@"
