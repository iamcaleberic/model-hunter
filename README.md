# model-hunter
Models evaluations

| Path | Model | What it does |
|------|-------|--------------|
| `src/meta/ijepa.py` | `facebook/ijepa_vith14_1k` | Embeds two COCO images and prints their cosine similarity |
| `src/meta/vjepa.py` | `facebook/vjepa2-vitl-fpc64-256` | Encodes 64 frames of a Kinetics clip with V-JEPA 2 |

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.12 (uv installs it from `.python-version` if missing)
- FFmpeg, for `torchcodec` in `vjepa.py`: `brew install ffmpeg`
- Network access: images, video and model weights are downloaded at run time. Weights are cached under `~/.cache/huggingface`.

Models load with `device_map="auto"`, so they use a GPU or Apple MPS when one is available and the CPU otherwise.

## Setup

```sh
uv sync
```

## Run

```sh
uv run python src/meta/ijepa.py
uv run python src/meta/vjepa.py
```
