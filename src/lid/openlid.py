"""Text language identification with OpenLID-v2.

OpenLID-v2 is a fastText classifier covering 201 languages, tuned for high
precision. See https://huggingface.co/laurievb/OpenLID-v2
"""

import argparse
import json

import fasttext
from huggingface_hub import hf_hub_download

from .text import gather_texts

REPO_ID = "laurievb/OpenLID-v2"
MODEL_FILE = "model.bin"


class OpenLid:
    """Wrapper around the OpenLID-v2 fastText model."""

    def __init__(self, repo_id: str = REPO_ID, model_file: str = MODEL_FILE):
        path = hf_hub_download(repo_id, model_file)
        self.model = fasttext.load_model(path)

    def identify(self, text: str) -> dict:
        """Return the predicted language for a single line of text."""
        labels, probs = self.model.predict(text)
        return {
            "text": text,
            "language": labels[0].removeprefix("__label__"),
            "confidence": round(float(probs[0]), 4),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="openlid",
        description="Identify the language of text with OpenLID-v2.",
    )
    parser.add_argument("text", nargs="*", help="text string(s) to identify")
    parser.add_argument("--file", help="read one text per line from this file")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args()

    texts = gather_texts(args.text, args.file)
    if not texts:
        parser.error("no input text provided")

    model = OpenLid()
    results = [model.identify(text) for text in texts]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(f"{r['text']}\t{r['language']}\t{r['confidence']}")


if __name__ == "__main__":
    main()
