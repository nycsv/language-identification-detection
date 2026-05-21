# language-identification-detection

Language identification (LID) for both **speech** and **text**, using
state-of-the-art open models.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.11+

Each script installs its dependencies, downloads the model on first run, and
runs inference. Both speech models use the GPU automatically when CUDA is
available, otherwise the CPU; the text models run on CPU.

## Spoken language identification

- **FireRedLID** — the LID module of FireRedASR2S. 100+ languages and 20+
  Chinese dialects, 97.18% accuracy on FLEURS, outperforming Whisper and
  SpeechBrain. <https://huggingface.co/FireRedTeam/FireRedLID>
- **Meta MMS-LID** — a Wav2Vec2 classifier from Meta's Massively Multilingual
  Speech project. Up to 4017 languages.
  <https://huggingface.co/facebook/mms-lid-256>

Audio input must be **16 kHz, 16-bit, mono PCM WAV**.

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

## Text language identification

- **GlotLID** — a fastText classifier covering 2,000+ language and script
  labels. Best recall and broadest coverage.
  <https://huggingface.co/cis-lmu/glotlid>
- **OpenLID-v2** — a fastText classifier covering 201 languages, tuned for high
  precision and a low false-positive rate.
  <https://huggingface.co/laurievb/OpenLID-v2>

Pass text as arguments, pipe it on stdin, or read one text per line from a file:

```bash
# GlotLID
scripts/run_glotlid.sh "Hello, world" "Bonjour le monde"
echo "Hola, ¿cómo estás?" | scripts/run_glotlid.sh
scripts/run_glotlid.sh --file lines.txt

# OpenLID-v2
scripts/run_openlid.sh "This is English" "Dies ist Deutsch"
```

Output is one tab-separated line per input: `<text>  <language>  <confidence>`,
where the language is an ISO code with script (e.g. `eng_Latn`). Pass `--json`
for JSON output.

## Layout

```
src/lid/audio.py      shared 16 kHz mono WAV loading and validation
src/lid/mms.py        MMS-LID model wrapper + `mms-lid` CLI
src/lid/firered.py    FireRedLID model wrapper + `firered-lid` CLI
src/lid/text.py       shared text input gathering and sanitizing
src/lid/glotlid.py    GlotLID model wrapper + `glotlid` CLI
src/lid/openlid.py    OpenLID-v2 model wrapper + `openlid` CLI
scripts/              one bash entry point per model
```
