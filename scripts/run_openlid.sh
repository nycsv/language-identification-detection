#!/usr/bin/env bash
# Download OpenLID-v2 and identify the language of text.
#
# Usage: scripts/run_openlid.sh "text" ["text" ...]
#        echo "text" | scripts/run_openlid.sh
#        scripts/run_openlid.sh --file lines.txt
set -euo pipefail

cd "$(dirname "$0")/.."

echo ">> Installing dependencies (text)..." >&2
uv sync --extra text

echo ">> Downloading model 'laurievb/OpenLID-v2'..." >&2
uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download('laurievb/OpenLID-v2', 'model.bin')"

echo ">> Identifying language..." >&2
uv run openlid "$@"
