#!/usr/bin/env bash
# Download GlotLID and identify the language of text.
#
# Usage: scripts/run_glotlid.sh "text" ["text" ...]
#        echo "text" | scripts/run_glotlid.sh
#        scripts/run_glotlid.sh --file lines.txt
set -euo pipefail

cd "$(dirname "$0")/.."

echo ">> Installing dependencies (text)..." >&2
uv sync --extra text

echo ">> Downloading model 'cis-lmu/glotlid'..." >&2
uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download('cis-lmu/glotlid', 'model.bin')"

echo ">> Identifying language..." >&2
uv run glotlid "$@"
