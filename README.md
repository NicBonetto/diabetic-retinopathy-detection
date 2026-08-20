# diabetic-retinopathy-detection

A deep learning model that detects and classifies diabetic retinopathy.

## Description

An end-to-end deep learning system for automated detection and grading of diabetic
retinopathy from retinal images. This project implements transfer learning with CNN
architectures (ResNet, EfficientNet), handles class imbalance, and provides
interpretability through Grad-CAM visualizations.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Python 3.11+ (uv will fetch it if you don't have it)
- GPU recommended but not required
- ~10GB free for the dataset

## Setup

```bash
uv sync
```

That creates `.venv/`, installs the project in editable mode, and writes `uv.lock`.
Add `--group dev` for the test and lint tooling:

```bash
uv sync --group dev
```

torch is resolved per-platform: macOS and Windows take the default PyPI wheel (on
Apple Silicon that's the MPS-capable build), Linux takes the CUDA 12.4 index. To
target a different CUDA version, edit the `pytorch-cu124` index in `pyproject.toml`.

## Usage

Download the APTOS 2019 dataset (requires a Kaggle API token in `~/.kaggle/kaggle.json`):

```bash
./scripts/download_data.sh
```

Build the train/val split:

```bash
uv run dr-preprocess \
  --data-dir data/raw/train_images \
  --labels data/raw/train.csv \
  --output-dir data/processed
```

Train:

```bash
cp config_example.yaml config.yaml   # config.yaml is gitignored
uv run dr-train --config config.yaml
```

Evaluate a checkpoint and emit metrics, ROC curves, confusion matrices, and Grad-CAM samples:

```bash
uv run dr-evaluate \
  --checkpoint checkpoints/best_model.pth \
  --data-dir data/processed/val \
  --labels data/processed/val_labels.csv \
  --output-dir results/evaluation
```

## Development

```bash
uv run pytest
uv run ruff check .
```

## Configuration

`config_example.yaml` documents every key. Two notes:

- `use_class_weights` reweights the loss; `use_weighted_sampler` oversamples rare
  classes. They address the same problem, so enabling both double-corrects it and
  training will refuse to start.
- `image_size` must suit the backbone. `vit_b_16` is fixed at 224x224.

## License

MIT. See [LICENSE](LICENSE).
