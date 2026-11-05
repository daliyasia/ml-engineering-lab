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

## Tests and remaining migration work

The tests cover MNIST batches, Hydra object instantiation, short training runs, checkpoint resumption, evaluation, and sweeps. Running the full suite downloads MNIST and executes these bounded training tests; it does not start the default 500-epoch training job. For configuration checks without training, use:

```bash
docker compose run --build --rm tests uv run pytest tests/test_configs.py
docker compose run --rm tests uv run pytest --collect-only
```

Verified in the GPU-enabled Compose image: `tests/test_configs.py` — **2 passed, 0 skipped**, with 16 dependency/framework warnings (9.29 seconds). CUDA tensor execution was also verified on an NVIDIA GeForce RTX 2060.

The full integration suite is not verified as passing. Configuration tests instantiate the data module, model, and trainer without training or downloading MNIST. Some legacy debug presets and retained detection utilities still need cleanup.

To add another classification dataset, implement a data module under `src/data/` and a matching data config. Update the network input channels, output classes, and Lightning metrics together: the current network expects grayscale images, and the module's metrics are fixed to ten classes.
