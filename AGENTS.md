# Repository Guidelines

## Project Structure & Module Organization

This repository trains and deploys the SPAN-F x2 super-resolution model. Core entry points live at the repository root: `train.py`, `test.py`, `inference.py`, `export_onnx.py`, and `dataset.py`. Model definitions and the model factory are in `archs/`; shared image, checkpoint, metric, and device helpers belong in `util.py`. JSON defaults live in `config/`. ONNX calibration and post-training quantization tools are isolated under `quantization/`. Unit tests mirror these areas in `tests/test_*.py`. Treat `checkpoints/`, `results/`, datasets, and local configuration files as generated or machine-specific artifacts.

## Build, Test, and Development Commands

Create the documented Python 3.10 environment and install dependencies:

```bash
uv venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
```

Run the complete test suite with `python -m unittest discover -s tests -v`. Start training with `.venv/bin/python train.py --config config/train.local.json`; copy `config/train.json` first and adjust local dataset paths. Evaluate a checkpoint with `.venv/bin/python test.py --weights checkpoints/<run>/best_model.pth`. Use `inference.py` for image or directory upscaling and `export_onnx.py` for a re-parameterized deployment graph. See `README.md` for tiling, dataset preview, and PTQ examples.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, PEP 8 spacing, snake_case functions and variables, PascalCase classes, and UPPER_CASE module constants. Prefer `pathlib.Path`, explicit validation with useful error messages, and small reusable helpers. Keep imports grouped as standard library, third-party, then local modules. Add concise docstrings where behavior or tensor layout is not obvious. No formatter or linter is configured, so keep changes consistent with neighboring code.

## Testing Guidelines

Tests use `unittest`, temporary directories, and small deterministic tensors or images. Name files `tests/test_<module>.py`, test classes `<Feature>Test`, and methods `test_<behavior>`. Add regression coverage for shape contracts, checkpoint compatibility, deterministic validation behavior, and deploy/re-parameterization equivalence. Run the full suite before submitting; tests should not require a GPU or external datasets.

## Commit & Pull Request Guidelines

Recent history uses short Conventional Commit-style subjects, primarily `feat: <imperative summary>` (for example, `feat: add high-order degradation pipeline`). Use an appropriate prefix such as `feat:`, `fix:`, `test:`, or `docs:` and keep each commit focused. Pull requests should explain the motivation and implementation, list validation commands, link relevant issues, and call out configuration or checkpoint compatibility changes. Include representative metrics or output images when model quality or visual behavior changes.
