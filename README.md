# Real-Time Dynamic Sign Language Translator

A dual-modal (vision + sequence) pipeline that watches a webcam feed,
extracts spatial hand/body landmarks frame-by-frame, models the
gesture sequence over time, and produces both live captions and
spoken audio in real time.

![demo](assets/demo.gif)
*(Record your own run and drop it in `assets/demo.gif` — see
[`assets/README.md`](assets/README.md) for a one-command `ffmpeg`
recipe.)*

```
webcam ──▶ MediaPipe landmarks ──▶ normalize + window ──▶ Transformer (CTC)
                                                              │
                                                              ▼
                                        gloss tokens ──▶ HF sentence refiner
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                       on-screen caption                text-to-speech
```

See [`docs/architecture.md`](docs/architecture.md) for the full data
flow and the reasoning behind each design choice.

## Features

- **Real-time webcam pipeline** — threaded capture keeps latency
  bounded even when inference can't keep up with camera FPS.
- **Spatial feature extraction** — MediaPipe Holistic pose + two-hand
  landmarks, normalized to be invariant to signer distance/position.
- **Sequence modeling** — a PyTorch Transformer encoder over a sliding
  window of frames, with a CTC head for frame-level gesture alignment
  on continuous (unsegmented) signing.
- **Gloss → fluent sentence** — a Hugging Face seq2seq model
  (`flan-t5-small` by default) turns raw gloss tokens into a
  grammatical sentence; falls back to templated joining if offline.
- **Text-to-speech** — offline `pyttsx3` by default, with an optional
  neural `SpeechT5` (Hugging Face) backend for higher audio quality.
- **Inference-time optimizations** — activity-gated inference (skip
  the model on idle frames), configurable inference stride, and a
  gloss-stability gate to avoid stuttering half-formed captions.
- **Modular, testable codebase** — clean `vision/` / `nlp/` split with
  a `pytest` suite covering the preprocessing math, CTC decoding, and
  model I/O contracts.

## Project structure

```
sign-language-translator/
├── main.py                      # real-time pipeline entry point
├── vision/                      # spatial feature extraction
│   ├── capture.py                # threaded webcam reader
│   ├── landmark_extractor.py     # MediaPipe Holistic wrapper
│   └── preprocessing.py          # normalization + sequence windowing
├── nlp/                          # sequence decoding
│   ├── sequence_model.py         # PyTorch Transformer (CTC head)
│   ├── decoder.py                # CTC greedy decode + HF sentence refiner
│   └── tts.py                    # pyttsx3 / SpeechT5 text-to-speech
├── tests/                        # pytest unit tests
├── configs/config.yaml           # all pipeline hyperparameters
├── models/                       # vocab.json + (optional) checkpoint.pt
├── docs/architecture.md          # data flow + design rationale
├── assets/                       # demo GIF lives here
└── .github/workflows/tests.yml   # CI: runs pytest on every push/PR
```

## Installation

Requires Python 3.9+.

> **Linux users:** the default `pyttsx3` TTS backend wraps `espeak`/
> `espeak-ng` on Linux, which is a system package, not a pip package.
> Install it first or `pyttsx3.init()` will raise a `RuntimeError`:
> ```bash
> sudo apt-get install espeak-ng   # Debian/Ubuntu
> ```
> macOS and Windows use their native OS speech engines instead and
> need no extra system package.

```bash
git clone https://github.com/<your-username>/sign-language-translator.git
cd sign-language-translator

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **MediaPipe version note:** `requirements.txt` pins
> `mediapipe==0.10.14`. `vision/landmark_extractor.py` uses the legacy
> `mediapipe.solutions.holistic` API, which Google removed from
> mediapipe releases after `0.10.14` in favor of the newer Tasks API.
> If you need a newer mediapipe, port `landmark_extractor.py` to
> `PoseLandmarker` + `HandLandmarker` from `mediapipe.tasks.python.vision`
> first (there's no direct combined-holistic equivalent in the Tasks API).

## Usage

Run the live pipeline (opens a webcam window with captions overlaid
and speaks recognized sentences aloud):

```bash
python main.py --config configs/config.yaml
```

Useful flags:

```bash
python main.py --camera 1                 # use a different webcam
python main.py --no-display                # headless (no cv2 window)
python main.py --no-display --max-frames 300   # short smoke test
```

> **Note:** No pretrained model weights are bundled (see
> [`models/README.md`](models/README.md)) — the pipeline runs
> end-to-end with randomly initialized weights out of the box so you
> can verify capture → landmarks → model → decoding → TTS wiring and
> latency before plugging in a trained checkpoint.

### Switching TTS backends

Edit `configs/config.yaml`:

```yaml
tts:
  backend: pyttsx3   # offline, low-latency (default)
  # backend: speecht5  # neural HF TTS — also `pip install -e .[speecht5]`
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite covers:

- `tests/test_preprocessing.py` — landmark normalization math (scale
  invariance, shoulder-centering, divide-by-zero guarding), feature
  flattening order, and sliding-window padding/truncation.
- `tests/test_landmark_extractor.py` — MediaPipe output handling with
  mocked results (missing-detection fallback to zero arrays).
- `tests/test_decoder.py` — CTC greedy decoding semantics (repeat
  collapsing, blank removal) and the offline gloss-joining fallback.
- `tests/test_sequence_model.py` — Transformer forward-pass shape
  contracts and valid log-probability output.

CI runs this suite automatically on every push/PR via
[`.github/workflows/tests.yml`](.github/workflows/tests.yml).

## Training your own model

This repo ships the full architecture and real-time inference
pipeline but not pretrained weights, since a production-quality
continuous sign language recognizer needs a licensed, large-scale
dataset (e.g. WLASL, How2Sign, PHOENIX-2014T). See
[`models/README.md`](models/README.md) for the training/checkpoint
contract `main.py` expects.

## Tech stack

| Layer | Tool |
|---|---|
| Frame capture / display | OpenCV |
| Landmark extraction | MediaPipe Holistic |
| Sequence modeling | PyTorch (custom Transformer + CTC) |
| Gloss → sentence, neural TTS | Hugging Face Transformers |
| Config | PyYAML |
| Testing / CI | pytest, GitHub Actions |

## Contributing

Issues and PRs welcome. Please run `pytest` before submitting a PR,
and keep new logic in `vision/` or `nlp/` covered by unit tests where
practical (see `docs/architecture.md` for where new pieces should
live).

## License

[MIT](LICENSE)
