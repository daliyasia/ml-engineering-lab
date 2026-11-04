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
├── Dockerfile/              # Current CUDA image and Python requirements
├── Dockerfile_ubuntu19/      # Legacy container setup
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

The current Docker image uses Ubuntu 24.04, CUDA 12.6.3, PyTorch 2.6.0, and torchvision 0.21.0. Build with `Dockerfile/` as the context because the Dockerfile copies `requirements.txt` from that directory:

```bash
docker build -t ml-engineering-lab:local Dockerfile/
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  ml-engineering-lab:local
```

Omit `--gpus all` for CPU-only use. The container starts in `/workspace`, where the project is mounted.

**The checked-in requirements are not yet a complete runtime environment.** Source imports also require packages absent from `Dockerfile/requirements.txt`, including `lightning`, `torchmetrics`, `pyrootutils`, `rich`, `mlflow`, and `hydra-colorlog`. Tests require `pytest`; Optuna sweeps require the Hydra Optuna plugin. Install and validate compatible versions before running. The code currently mixes imports from `lightning` and `pytorch_lightning`.

## Training configuration

**`configs/train.yaml` is currently missing**, although `src/train.py`, the training fixtures, and `scripts/schedule.sh` expect it. Training is not ready to launch from a fresh checkout until that config is restored or created.

For a minimal classification configuration, create `configs/train.yaml` with:

```yaml
# @package _global_
defaults:
  - _self_
  - data: mnist.yaml
  - model: mnist.yaml
  - callbacks: model_checkpoint.yaml
  - logger: csv.yaml
  - trainer: cpu.yaml
  - paths: default.yaml
  - extras: default.yaml
  - hydra: default.yaml

task_name: train
tags: [mnist]
train: true
test: true
compile: false
ckpt_path: null
seed: 42

trainer:
  max_epochs: 10
  precision: 32

callbacks:
  model_checkpoint:
    dirpath: ${paths.output_dir}/checkpoints
    monitor: val/loss
    mode: min
```

After completing environment setup and adding that configuration, run from the repository root:

```bash
# Train on CPU
python src/train.py

# Override hyperparameters
python src/train.py trainer.max_epochs=20 model.optimizer.lr=0.0001 data.batch_size=64

# Train on one GPU
python src/train.py trainer=gpu

# Exercise one batch per loop
python src/train.py +trainer.fast_dev_run=true

# Resume an existing run
python src/train.py ckpt_path=/absolute/path/to/last.ckpt
```

These commands use the sample training configuration above. Additional presets need review before use; they are not all aligned with classification.

## Checkpoint evaluation

After installing the runtime dependencies, evaluate a matching MNIST checkpoint with the existing evaluation config:

```bash
python src/eval.py \
  ckpt_path=/absolute/path/to/model.ckpt \
  trainer=cpu trainer.precision=32
```

The evaluation default selects a GPU trainer; the command explicitly chooses CPU and full precision. Data is downloaded if needed.

## Data, outputs, and logging

`pyrootutils` locates `.project-root`, adds the repository to the Python path, and loads local environment variables. Default paths resolve to `data/` and `logs/` under the repository root; MNIST uses `data/MNIST/`.

Hydra stores single runs under `logs/<task_name>/runs/<timestamp>/` and multiruns under `logs/<task_name>/multiruns/<timestamp>/<job_number>/`. Checkpoint locations are configured through callbacks. The sample training config writes them into each run's `checkpoints/` directory.

Logger configs are included for CSV, TensorBoard, MLflow, W&B, Comet, Neptune, and Aim. External backends require their own dependencies and settings. The sample above selects CSV to avoid account setup.

Datasets, checkpoints, run outputs, Python caches, virtual environments, and `.env` files are ignored by Git. DVC metadata and example environment files remain eligible for version control.

## Tests and remaining migration work

The tests cover MNIST batches, Hydra object instantiation, short training runs, checkpoint resumption, evaluation, and sweeps. Once the environment and training config are ready, run:

```bash
python -m pytest tests/test_datamodules.py tests/test_configs.py
python -m pytest tests/
```

The full suite is not currently a passing baseline. Known inconsistencies include:

- `configs/callbacks/default.yaml` monitors `val/loss_final`, while classification logs `val/loss`. The sample training config uses the standalone checkpoint callback instead.
- `configs/experiment/example.yaml` and parts of `configs/debug/default.yaml` still refer to removed detection models and datasets.
- The Optuna example searches parameters absent from the current ResNet and requests an outdated metric name. Classification logs `val_best/accuracy`.
- Some training tests assert older metric names such as `train/acc` and `val/acc`; the module logs `train/accuracy` and `val/accuracy`.
- `MNISTLitModule.on_validation_epoch_end()` currently logs the best F1 value under `val_best/loss`; use `val/loss` for loss-based monitoring until this is corrected.
- Retained detection utilities and W&B media callbacks are legacy code, rather than a supported detection pipeline.

To add another classification dataset, implement a data module under `src/data/` and a matching data config. Update the network input channels, output classes, and Lightning metrics together: the current network expects grayscale images, and the module's metrics are fixed to ten classes.
