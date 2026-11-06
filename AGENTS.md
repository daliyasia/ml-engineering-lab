# Repository Guidelines

## Project Structure & Module Organization

`src/train.py` and `src/eval.py` are Hydra entry points. Place dataset loaders in `src/data/`, Lightning modules in `src/models/`, network architectures in `src/models/components/`, and shared helpers in `src/utils/`. Keep callbacks in `src/callbacks/`.

`configs/` contains YAML groups for data, models, trainers, callbacks, loggers, and experiments. Tests live in `tests/`, with shared fixtures in `tests/conftest.py`. Use `notebooks/` for exploration and `inference/` for inference work. Downloaded data and generated outputs belong in ignored `data/` and `logs/` directories.

## Build, Test, and Development Commands

When asked to run project code, always use Docker Compose, including training, evaluation, tests, and diagnostic scripts. Do not run project Python or uv commands directly on the host unless explicitly requested.

Run commands from the repository root:

- `docker compose build tests`: build the CUDA image with pinned uv and locked dependencies.
- `docker compose run --build --rm tests uv run pytest tests/test_configs.py`: verify configuration instantiation without training.
- `docker compose run --build --rm tests uv run pytest tests/test_train.py`: run bounded CPU/GPU training, distributed training, and checkpoint-resume tests.
- `docker compose run --build --rm tests uv run pytest --collect-only`: check test discovery and imports.
- `docker compose run --build --rm tests uv run pytest`: run the full suite, including bounded training and sweep tests.
- `docker compose run --build --rm tests uv run python src/train.py`: launch training with the default configuration.
- `docker compose run --build --rm tests uv run python src/train.py trainer=cpu trainer.precision=32-true`: launch CPU training only when intended.

Compose requests NVIDIA GPUs; the host needs NVIDIA Container Toolkit. Preserve Python 3.12, PyTorch 2.6.0, torchvision 0.21.0, and CUDA 12.6 compatibility. Update `uv.lock` with uv whenever dependencies change.

## Continuous Integration

`.github/workflows/ci.yml` runs Python 3.12 CPU checks on PRs to `main`, pushes to `main`, and manual dispatches. Direct uv commands in that workflow run on GitHub Actions runners; local reproduction must use Docker Compose. Preserve locked dependencies and the sweep exclusion; GPU tests skip on hosted CPU runners. Reproduce the CPU test command locally with `docker compose run --build --rm -e CUDA_VISIBLE_DEVICES="" tests uv run --locked pytest --ignore=tests/test_sweeps.py -ra`. Inspect the seven-day `pytest-cpu` JUnit artifact for failures. Local Compose still validates GPU behavior when CUDA devices are visible.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Follow existing type annotations and Hydra `_target_` configuration patterns. Match configuration names to modules, such as `data/mnist.yaml`. Ruff enforces lint, import ordering, and formatting with a 99-character line target. MyPy checks the core training path configured in `pyproject.toml`; avoid broad typing suppressions. Run quality tools through Docker Compose, for example `docker compose run --build --rm tests uv run --locked ruff check .`, `docker compose run --build --rm tests uv run --locked ruff format --check .`, and `docker compose run --build --rm tests uv run --locked mypy`.

## Testing Guidelines

Use pytest files named `test_*.py` and functions named `test_*`. Reuse fixtures and mark expensive integration checks with `@pytest.mark.slow`. No coverage threshold is configured. The GPU Compose baseline is 15 passed and one optional W&B test skipped. Report exact commands, failures, and skips, and preserve meaningful assertions.

## Commit & Pull Request Guidelines

History uses concise imperative subjects, such as “Restore Hydra training configuration.” Keep commits focused. PR descriptions should explain behavior changes, affected configurations, validation results, and relevant issue links. Distinguish configuration verification from executed training.

For every “commit & push” request, run `docker compose run --build --rm tests uv run pytest --ignore=tests/test_sweeps.py` first. Exclude sweep tests from this workflow unless explicitly requested. Fix failures without weakening assertions, then commit and push the verified changes. Report results and the sweep exclusion.

## Security & Configuration

Keep credentials in ignored `.env` files and machine overrides in `configs/local/`. Never commit secrets, datasets, checkpoints, or experiment outputs. Preserve unrelated working-tree changes.
