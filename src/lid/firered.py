"""Spoken language identification with FireRedLID.

FireRedLID is the LID module of FireRedASR2S. See
https://huggingface.co/FireRedTeam/FireRedLID
"""

import argparse
import json
from pathlib import Path

import torch
from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig

from .audio import validate_wav

DEFAULT_MODEL_DIR = "models/FireRedLID"


class FireRedLidModel:
    """Wrapper around a locally downloaded FireRedLID checkpoint."""

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        use_gpu: bool | None = None,
        use_half: bool = False,
    ):
        weights = Path(model_dir) / "model.pth.tar"
        if not weights.is_file():
            raise FileNotFoundError(
                f"FireRedLID weights not found at {weights}. "
                "Download them first (see scripts/run_firered_lid.sh)."
            )
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        config = FireRedLidConfig(use_gpu=use_gpu, use_half=use_half)
        self.model = FireRedLid.from_pretrained(str(model_dir), config)

    def identify(self, wav_paths: list[str]) -> list[dict]:
        """Return predicted languages for one or more WAV files."""
        paths = [str(validate_wav(p)) for p in wav_paths]
        uttids = [Path(p).stem for p in paths]
        results = self.model.process(uttids, paths)
        return [
            {
                "wav": r.get("wav", ""),
                "language": r.get("lang", ""),
                "confidence": r.get("confidence"),
            }
            for r in results
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="firered-lid",
        description="Identify the spoken language of 16 kHz mono WAV files "
        "with FireRedLID.",
    )
    parser.add_argument("wav", nargs="+", help="path(s) to 16 kHz mono WAV file(s)")
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"directory with FireRedLID weights (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args()

    model = FireRedLidModel(args.model_dir)
    results = model.identify(args.wav)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(f"{r['wav']}\t{r['language']}\t{r['confidence']}")


if __name__ == "__main__":
    main()
