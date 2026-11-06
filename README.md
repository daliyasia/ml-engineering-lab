# ML Engineering Lab

A Python scaffold for image-classification experiments with PyTorch, Lightning, and Hydra. The current implementation trains a custom ResNet on MNIST and provides reusable training, checkpoint evaluation, metrics, configuration, and experiment-logging code.

The repository is being simplified around classification. Earlier detection and license-plate models have been removed; some supporting utilities and configuration examples still need cleanup.

## Current implementation

- **Dataset:** MNIST grayscale digit images, normalized with mean `0.1307` and standard deviation `0.3081`.
- **Splits:** the original training and test datasets are combined, then split into 55,000 training, 5,000 validation, and 10,000 test samples with a fixed split seed of 42. This is a custom split, rather than the official MNIST test split.
- **Network:** a configurable ResNet with one input channel and ten output classes.
- **Training:** cross-entropy loss, Adam, and a ReduceLROnPlateau scheduler monitoring `val/loss` in the MNIST model config.
- **Metrics:** loss, accuracy, precision, recall, and F1 for training, validation, and testing. The Lightning module also maintains confusion-matrix state.
- **Configuration:** Hydra instantiates datasets, models, optimizers, trainers, callbacks, and loggers from YAML.

## Repository structure

```text
.
├── configs/
│   ├── callbacks/           # Checkpoints, early stopping, logging hooks
│   ├── cluster/             # Custom cluster environment configuration
│   ├── data/mnist.yaml      # Dataset paths, splits, batching, workers
│   ├── debug/               # Debugging presets; some need migration
│   ├── experiment/          # Legacy experiment example
│   ├── extras/              # Config printing and tag handling
│   ├── hparams_search/      # Optuna example requiring updates
│   ├── hydra/               # Logging and output-directory settings
│   ├── local/               # Ignored machine-specific overrides
│   ├── logger/              # Experiment tracker configurations
│   ├── model/mnist.yaml     # ResNet, optimizer, and scheduler settings
│   ├── paths/               # Project, dataset, and output paths
│   ├── trainer/             # CPU, GPU, DDP, and MPS presets
│   └── eval.yaml            # Checkpoint evaluation configuration
├── src/
│   ├── data/                # MNIST LightningDataModule
│   ├── models/
│   │   ├── components/      # ResNet architecture
│   │   ├── utils/           # Retained metrics and detection utilities
│   │   └── mnist_module.py  # Classification loss, steps, metrics, optimizer
│   ├── callbacks/           # MLflow and W&B callbacks
│   ├── cluster/             # Custom Lightning cluster environment
│   ├── utils/               # Instantiation, logging, and run helpers
│   ├── train.py             # Training and optional best-checkpoint testing
│   └── eval.py              # Evaluation of an existing checkpoint
├── tests/                   # Dataset, config, training, evaluation, sweep tests
├── scripts/schedule.sh      # Example sequential training runs
├── Dockerfile/              # CUDA image with pinned uv
├── pyproject.toml           # Runtime and default development dependencies
├── uv.lock                  # Reproducible dependency resolutions
├── .dvc/                    # DVC configuration and local-state ignore rules
├── .gitignore               # Caches, datasets, outputs, and private settings
└── .project-root            # Root marker used by pyrootutils
```

## How the code fits together

`src/train.py` composes a Hydra config and instantiates the data module, Lightning model, callbacks, loggers, and Trainer. It calls `trainer.fit()` and optionally tests the best saved checkpoint. It also supports seeding, checkpoint resumption, and optional `torch.compile`.

`src/data/mnist_datamodule.py` downloads MNIST, normalizes samples, creates the splits, and supplies data loaders. `src/models/components/mnist.py` defines the PyTorch network. `src/models/mnist_module.py` wraps it with Lightning training, validation, testing, and optimizer hooks.

`src/eval.py` instantiates the configured model and data module, then calls `trainer.test()` with a required checkpoint path. Evaluation must use the same architecture as the checkpoint.

## Environment setup

Clone the repository and enter its root:

```bash
git clone https://github.com/ohadjerci/ml-engineering-lab.git
cd ml-engineering-lab
```

The image installs uv 0.13.7 and uses Python 3.12, CUDA 12.6.3, PyTorch 2.6.0, and torchvision 0.21.0. The lockfile targets Linux x86-64 CUDA deployment. Linux torch wheels come from the explicit CUDA 12.6 index; other dependencies come from PyPI. NumPy remains pinned to 1.26.4 for legacy utilities.

Run the GPU-enabled test service from the repository root (requires Docker Compose 2.30+):

```bash
docker compose up --build --abort-on-container-exit --exit-code-from tests
```

The service builds `mlcycle:uv`, exposes all NVIDIA GPUs, and runs the image's default `uv run pytest`. For the current configuration-only verification scope, build and run:

```bash
docker compose run --build --rm tests uv run pytest tests/test_configs.py
```

Build from the repository root and run the tests (the image defaults to `uv run pytest`):

```bash
docker build -f Dockerfile/Dockerfile -t mlcycle:uv .
docker run --rm --gpus all mlcycle:uv
```

GPU containers require an NVIDIA driver and NVIDIA Container Toolkit on the host. The image includes the CUDA toolkit, cuDNN, source, and tests. To test your working tree without rebuilding source changes:

```bash
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace mlcycle:uv uv run pytest
```

The non-root user owns `/opt/venv`, outside the repository mount. The development group includes pytest, shell command support, the Hydra Optuna plugin, and setuptools for existing `pkg_resources` imports. No host uv installation or environment activation is needed for Docker.

If `uv.lock` is missing, generate it using the Dockerfile's base stage, then build the final image:

```bash
docker compose build lock
LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)" docker compose run --rm lock uv lock
```

The `lock` service is an opt-in maintenance service under the `tools` profile. Normal `docker compose up` starts only `tests`.

For local development with uv installed:

```bash
uv sync --locked
uv run pytest tests/test_configs.py
uv run pytest
```

### Open a container for debugging

From the repository root, start an interactive Bash shell with GPU access and your working tree mounted:

```bash
docker compose run --build --rm -v "$PWD:/workspace" -w /workspace tests bash
```

Edits made on the host are immediately available in `/workspace`. Inside the container, run configuration checks or enter Python's debugger on a test failure:

```bash
uv run pytest tests/test_configs.py
uv run pytest tests/test_configs.py --pdb
```

Type `exit` to leave the shell and remove the temporary container. Changes to mounted project files remain on the host.

## Training configuration

The default configuration selects MNIST, the classification ResNet, default callbacks monitoring `val/loss`, MLflow, and a GPU trainer. CPU use requires full precision:

```bash
uv run python src/train.py trainer=cpu trainer.precision=32-true
uv run python src/train.py trainer=gpu
uv run python src/train.py trainer.max_epochs=20 model.optimizer.lr=0.0001 data.batch_size=64
uv run python src/train.py ckpt_path=/absolute/path/to/last.ckpt
```

To train on a GPU inside Docker:

```bash
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace \
  mlcycle:uv uv run python src/train.py
```

## Checkpoint evaluation

Evaluate a matching MNIST checkpoint with the existing evaluation config:

```bash
uv run python src/eval.py \
  ckpt_path=/absolute/path/to/model.ckpt \
  trainer=cpu trainer.precision=32-true
```

The evaluation default selects a GPU trainer; the command explicitly chooses CPU and full precision. Data is downloaded if needed.

## Data, outputs, and logging

`pyrootutils` locates `.project-root`, adds the repository to the Python path, and loads local environment variables. Default paths resolve to `data/` and `logs/` under the repository root; MNIST uses `data/MNIST/`.

Hydra stores single runs under `logs/<task_name>/runs/<timestamp>/` and multiruns under `logs/<task_name>/multiruns/<timestamp>/<job_number>/`. Checkpoint locations are configured through callbacks. The default training config writes them into each run's `checkpoints/` directory.

Logger configs are included for CSV, TensorBoard, MLflow, W&B, Comet, Neptune, and Aim. External backends require their own dependencies and settings. Use `logger=csv` for local CSV logging.

Datasets, checkpoints, run outputs, Python caches, virtual environments, and `.env` files are ignored by Git. DVC metadata and example environment files remain eligible for version control.

## Continuous integration

GitHub Actions runs **Python tests (CPU)** on GitHub-hosted Ubuntu 24.04 for pull requests targeting `main`, pushes to `main`, and manual runs from the Actions tab. It uses Python 3.12 and uv 0.13.7, checks `uv.lock`, installs with `uv sync --locked`, and checks dependency consistency before pytest.

CI excludes sweeps and runs the bounded CPU training, evaluation, distributed training, checkpoint-resume, data, and configuration tests. The two GPU-only tests skip using their existing decorators. To reproduce the test command locally after `uv sync --locked`:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --locked pytest --ignore=tests/test_sweeps.py -ra --junitxml=/tmp/pytest-cpu.xml
```

To check the CPU test selection in the local Compose environment:

```bash
docker compose run --build --rm -e CUDA_VISIBLE_DEVICES= tests \
  uv run --locked pytest --ignore=tests/test_sweeps.py -ra
```

The hosted job runs uv directly and needs no self-hosted runner or repository secrets. It preserves the locked CUDA dependencies; it does not build the Docker image or validate GPU execution. uv's cache is keyed by `uv.lock` and stored alongside `.venv` for space-efficient installation. Cold runs download several GB of dependencies plus MNIST; inspect dependency-step logs and available runner disk space if installation fails. The job times out after 30 minutes and cancels superseded runs.

The `pytest-cpu` JUnit artifact is uploaded even after pytest fails and retained for seven days. Installation failures before pytest produce no test report; check the failed step's logs instead. Download the artifact from the workflow run to inspect failures and skips.

After the first successful hosted run, repository administrators can require **Python tests (CPU)** under the branch rules for `main`. Branch protection is not changed automatically.

## Code quality

CI runs Ruff linting, Ruff formatting verification, and MyPy in the existing hosted CPU
job. Each check can fail the job independently, and they still run if pytest fails.
Coverage reporting, JUnit uploads, workflow triggers, and the local GPU-enabled Compose
service are preserved. This workflow has no self-hosted GPU job or coverage threshold.

Ruff checks Python files throughout the repository for common errors, unused imports
and variables, and import ordering (`E4`, `E7`, `E9`, `F`, `I`). Its formatter uses four
spaces, double quotes, and a 99-character line target. Datasets, generated files,
experiment outputs, build artifacts, caches, and notebooks are outside these checks.
Black is not required.

Run checks locally through Compose:

```bash
docker compose run --build --rm tests uv run --locked ruff check .
docker compose run --build --rm tests uv run --locked ruff format --check .
docker compose run --build --rm tests uv run --locked mypy
```

To apply reviewed safe lint fixes and formatting to the checkout, use the existing
Compose tools service, which mounts the checkout with the configured local user:

```bash
docker compose run --rm lock uv tool run --from ruff==0.17.4 ruff check . --fix
docker compose run --rm lock uv tool run --from ruff==0.17.4 ruff format .
```

MyPy targets Python 3.12 and checks annotated code plus untyped function bodies, without
global strict mode. Its initial enforced scope covers training, evaluation, the MNIST
DataModule and Lightning module, network components, and shared utilities. Running
`mypy` without paths uses the scope in `pyproject.toml`; CI passes the same paths
explicitly. Only the untyped `torchvision` and `pyrootutils` imports and optional `wandb`
imports have missing-import overrides.

The adoption baseline was 41 Ruff diagnostics, 17 files requiring formatting, and 90
MyPy errors across 12 source files. Type enforcement will expand first to callbacks,
then the legacy detection helpers under `src/models/utils/`, then the unfinished
cluster plugin. The latter two need tensor-state annotations, invalid tensor-call
repairs, and cluster interface implementation. They are not covered by the initial
MyPy command and have no project-wide error suppression. Use `mypy src` for a full
backlog audit; Ruff continues to check all these modules.

After the core repairs, the full-source audit reports 47 errors in four deferred
files: `box_utils.py` and `loss_utils.py` have two invalid tensor-call overloads,
`custom_metric.py` has 38 tensor-state operator errors, and `cluster_module.py`
has seven missing-return, signature, and method-definition errors. The enforced
core scope passes with no errors across 14 source files.

## Tests and remaining migration work

The tests cover MNIST batches, Hydra object instantiation, short training runs, checkpoint resumption, evaluation, and sweeps. Running the full suite downloads MNIST and executes these bounded training tests; it does not start the default 500-epoch training job. For configuration checks without training, use:

```bash
docker compose run --build --rm tests uv run pytest tests/test_configs.py
docker compose run --rm tests uv run pytest --collect-only
```

Verified in the GPU-enabled Compose image: `tests/test_configs.py` — **2 passed, 0 skipped**, with 16 dependency/framework warnings (9.29 seconds). CUDA tensor execution was also verified on an NVIDIA GeForce RTX 2060.

Full-suite verification in the GPU-enabled Compose image: **15 passed, 1 skipped**, with 46 warnings (205.86 seconds), using `docker compose run --build --rm tests uv run pytest`. The optional W&B sweep test skipped because W&B is not installed. Future commit-and-push checks exclude `tests/test_sweeps.py` unless explicitly requested. Configuration tests instantiate the data module, model, and trainer without training or downloading MNIST. Some legacy debug presets and retained detection utilities still need cleanup.

To add another classification dataset, implement a data module under `src/data/` and a matching data config. Update the network input channels, output classes, and Lightning metrics together: the current network expects grayscale images, and the module's metrics are fixed to ten classes.
