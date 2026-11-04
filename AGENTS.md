# Repository Guidelines

## Project Structure & Module Organization

This repository implements a five-class flower image classification workflow in Python and PyTorch.

- `src/eda.py`: dataset inspection, duplicate analysis, and exploratory figures.
- `src/split_data.py`: duplicate-aware, stratified train/validation/test splitting.
- `src/dataset.py`: dataset loading, augmentation, normalization, and data loaders.
- `src/preprocess.py`: preprocessing validation and visualization.
- `src/model.py` and `src/train.py`: SimpleCNN architecture, training, evaluation, and report export.
- `splits/`: tracked CSV manifests and split summaries.
- `reports/`: tracked Markdown reports, figures, and metric tables.
- `flower_photos/`: local dataset, organized by class; ignored by Git. Checkpoints and training logs belong in ignored `runs/`.

## Build, Test, and Development Commands

Run commands from the repository root in an activated Python environment:

```powershell
python -m pip install -r requirements.txt
python src/eda.py
python src/split_data.py --seed 42
python src/preprocess.py --full-check
python src/model.py
python src/train.py --epochs 30 --batch-size 32
```

These commands install dependencies, generate EDA outputs, regenerate splits, validate preprocessing, check model output shape, and train the baseline, respectively. Place the dataset in `flower_photos/` before running data-dependent commands. Use `--help` to inspect script options. For experiments, select separate `--out-dir` and `--reports-dir` paths in the training command to preserve baseline artifacts.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Use type hints, concise docstrings, and `pathlib.Path` for filesystem operations. Keep reusable transforms in `dataset.py` and architecture changes in `model.py`. No formatter or linter is currently configured.

## Testing Guidelines

Validation currently uses executable scripts; no dedicated test suite or coverage threshold is configured. Run `python src/model.py` for a forward-pass smoke check and `python src/preprocess.py --full-check` for data pipeline changes. Verify finite float32 image tensors, label counts, deterministic validation transforms, and zero duplicate clusters crossing splits. For training changes, run a short experiment in separate output directories and inspect metrics and generated figures. Keep augmentation confined to training data.

## Commit & Pull Request Guidelines

Existing commits use short imperative subjects, such as `Add SimpleCNN baseline and training pipeline`. Follow that pattern and keep each commit focused. PR descriptions should explain the change, list validation commands and results, and link relevant issues. For model experiments, record the seed, split configuration, hyperparameters, and metrics; include updated plots when visual outputs change. Commit relevant reports alongside code, and keep datasets and checkpoints out of Git.
