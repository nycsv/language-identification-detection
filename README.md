# language-identification-detection

Spoken language identification (LID) with two state-of-the-art open models:

- **FireRedLID** — the LID module of FireRedASR2S. 100+ languages and 20+
  Chinese dialects, 97.18% accuracy on FLEURS, outperforming Whisper and
  SpeechBrain. <https://huggingface.co/FireRedTeam/FireRedLID>
- **Meta MMS-LID** — a Wav2Vec2 classifier from Meta's Massively Multilingual
  Speech project. Up to 4017 languages.
  <https://huggingface.co/facebook/mms-lid-256>

Audio input must be **16 kHz, 16-bit, mono PCM WAV**.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.11+

## Usage

Each script installs its dependencies, downloads the model on first run, and
runs inference:

```bash
# Meta MMS-LID
scripts/run_mms_lid.sh audio.wav [more.wav ...]

# FireRedLID
scripts/run_firered_lid.sh audio.wav [more.wav ...]
```

Output is one tab-separated line per file: `<wav>  <language>  <confidence>`.
Pass `--json` for JSON output.

### Options

- MMS model variant — `MMS_MODEL_ID=facebook/mms-lid-512 scripts/run_mms_lid.sh ...`
  (variants: `mms-lid-126`, `-256`, `-512`, `-1024`, `-2048`, `-4017`)
- FireRedLID weights location — `FIRERED_MODEL_DIR=/path scripts/run_firered_lid.sh ...`

Both models use the GPU automatically when CUDA is available, otherwise the CPU.

## Layout

```
src/lid/audio.py      shared 16 kHz mono WAV loading and validation
src/lid/mms.py        MMS-LID model wrapper + `mms-lid` CLI
src/lid/firered.py    FireRedLID model wrapper + `firered-lid` CLI
scripts/              one bash entry point per model
```
