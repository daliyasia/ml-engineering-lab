# Repository Guidelines

## Project Structure & Module Organization

`src/train.py` and `src/eval.py` are Hydra entry points. Place dataset loaders in `src/data/`, Lightning modules in `src/models/`, network architectures in `src/models/components/`, and shared helpers in `src/utils/`. Keep callbacks in `src/callbacks/`.

`configs/` contains YAML groups for data, models, trainers, callbacks, loggers, and experiments. Tests live in `tests/`, with shared fixtures in `tests/conftest.py`. Use `notebooks/` for exploration and `inference/` for inference work. Downloaded data and generated outputs belong in ignored `data/` and `logs/` directories.

## Build, Test, and Development Commands

Run commands from the repository root:

- `uv sync --locked`: install runtime and default development dependencies locally.
- `docker compose build tests`: build the CUDA image with pinned uv and locked dependencies.
- `docker compose run --build --rm tests uv run pytest tests/test_configs.py`: verify configuration instantiation without training.
- `uv run pytest --collect-only`: check test discovery and imports.
- `uv run pytest`: run the full suite, including bounded training and sweep tests.
- `uv run python src/train.py trainer=cpu trainer.precision=32-true`: launch CPU training only when intended.

Compose requests NVIDIA GPUs; the host needs NVIDIA Container Toolkit. Preserve Python 3.12, PyTorch 2.6.0, torchvision 0.21.0, and CUDA 12.6 compatibility. Update `uv.lock` with uv whenever dependencies change.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Follow existing type annotations and Hydra `_target_` configuration patterns. Match configuration names to modules, such as `data/mnist.yaml`. No formatter or linter is currently configured; avoid unrelated formatting changes.

## Testing Guidelines

Use pytest files named `test_*.py` and functions named `test_*`. Reuse fixtures and mark expensive integration checks with `@pytest.mark.slow`. No coverage threshold is configured. Configuration tests are verified; the full suite is not established as passing. Report exact commands, failures, and skips, and preserve meaningful assertions.

## Commit & Pull Request Guidelines

History uses concise imperative subjects, such as “Restore Hydra training configuration.” Keep commits focused. PR descriptions should explain behavior changes, affected configurations, validation results, and relevant issue links. Distinguish configuration verification from executed training.

## Security & Configuration

Keep credentials in ignored `.env` files and machine overrides in `configs/local/`. Never commit secrets, datasets, checkpoints, or experiment outputs. Preserve unrelated working-tree changes.
