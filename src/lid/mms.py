"""Spoken language identification with Meta's MMS-LID model.

MMS-LID is a Wav2Vec2 classifier from Meta's Massively Multilingual Speech
project. See https://huggingface.co/facebook/mms-lid-256
"""

import argparse
import json

import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from .audio import SAMPLE_RATE, load_wav

DEFAULT_MODEL = "facebook/mms-lid-256"


class MmsLid:
    """Wrapper around an MMS-LID checkpoint."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def identify(self, wav_path: str) -> dict:
        """Return the predicted language for a single WAV file."""
        audio = load_wav(wav_path)
        inputs = self.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).to(self.device)
        probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        index = int(probs.argmax())
        return {
            "wav": str(wav_path),
            "language": self.model.config.id2label[index],
            "confidence": round(float(probs[index]), 4),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mms-lid",
        description="Identify the spoken language of 16 kHz mono WAV files "
        "with Meta MMS-LID.",
    )
    parser.add_argument("wav", nargs="+", help="path(s) to 16 kHz mono WAV file(s)")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args()

    model = MmsLid(args.model)
    results = [model.identify(path) for path in args.wav]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(f"{r['wav']}\t{r['language']}\t{r['confidence']}")


if __name__ == "__main__":
    main()
