"""Selection must never replace the alias with a worse or unmeasured model."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from mlflow.exceptions import MlflowException
from omegaconf import OmegaConf

from src.utils.model_registry import extract_model_metadata, promote_if_better


def test_extract_metadata_prefers_runtime_values():
    class Normalize:
        mean = (0.25,)
        std = (0.75,)

    datamodule = SimpleNamespace(
        name="RuntimeDataset",
        transforms_test=SimpleNamespace(transforms=[Normalize()]),
        data_test=[(torch.zeros(1, 12, 16), 0)],
        classes=["zero", "one"],
        num_classes=2,
    )
    model = SimpleNamespace(net=torch.nn.Linear(1, 1))
    data_cfg = OmegaConf.create(
        {"name": "ConfiguredDataset", "mean": [0.0], "input_shape": [3, 28, 28]}
    )
    model_cfg = OmegaConf.create({"_target_": "configured.Model", "net": {"width": 8}})

    metadata = extract_model_metadata(datamodule, model, data_cfg, model_cfg)

    assert metadata["dataset"] == "RuntimeDataset"
    assert metadata["preprocessing"] == ["Normalize"]
    assert metadata["mean"] == [0.25]
    assert metadata["std"] == [0.75]
    assert metadata["input_shape"] == [1, 12, 16]
    assert metadata["image_size"] == [12, 16]
    assert metadata["labels"] == ["zero", "one"]
    assert metadata["num_classes"] == 2
    assert metadata["model_architecture"] == "Linear"
    assert metadata["model_config"] == {"width": 8}


def test_extract_metadata_uses_config_fallback_and_omits_unknown_values():
    metadata = extract_model_metadata(
        SimpleNamespace(),
        SimpleNamespace(),
        OmegaConf.create(
            {
                "name": "ConfiguredDataset",
                "input_shape": [3, 20, 30],
                "image_size": [20, 30],
                "labels": ["cat", "dog"],
                "num_classes": 2,
                "preprocessing": ["Resize", "ToTensor"],
                "mean": [0.5],
                "std": [0.25],
            }
        ),
        OmegaConf.create({"_target_": "models.Example", "net": {"width": 16}}),
    )

    assert metadata["dataset"] == "ConfiguredDataset"
    assert metadata["input_shape"] == [3, 20, 30]
    assert metadata["image_size"] == [20, 30]
    assert metadata["labels"] == ["cat", "dog"]
    assert metadata["num_classes"] == 2
    assert metadata["preprocessing"] == ["Resize", "ToTensor"]
    assert metadata["mean"] == [0.5]
    assert metadata["std"] == [0.25]
    assert metadata["model_architecture"] == "models.Example"
    assert "mnist" not in str(metadata).lower()
    assert "image_size" not in extract_model_metadata(
        SimpleNamespace(),
        SimpleNamespace(),
        OmegaConf.create({}),
        OmegaConf.create({}),
    )


@pytest.mark.parametrize("candidate,expected", [(0.9, True), (1.0, False), (1.1, False)])
def test_promote_only_strict_improvement(candidate, expected):
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(aliases={"best_model": "1"})
    client.get_model_version.return_value = SimpleNamespace(
        tags={"selection_metric": "test/loss", "selection_score": "1.0"}
    )
    assert (
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", candidate) is expected
    )
    assert client.set_registered_model_alias.called is expected


def test_first_version_is_promoted():
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(aliases={})
    assert promote_if_better(client, "MNIST", "best_model", "1", "test/loss", 1.0)
    client.set_registered_model_alias.assert_called_once_with("MNIST", "best_model", "1")


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_nonfinite_score_is_rejected(score):
    client = Mock()
    with pytest.raises(ValueError, match="Non-finite"):
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", score)
    client.set_registered_model_alias.assert_not_called()


def test_incompatible_selection_metadata_is_rejected():
    client = Mock()
    client.get_registered_model.return_value = SimpleNamespace(aliases={"best_model": "1"})
    client.get_model_version.return_value = SimpleNamespace(tags={})
    with pytest.raises(ValueError, match="incompatible"):
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", 1.0)
    client.set_registered_model_alias.assert_not_called()


def test_connection_errors_do_not_look_like_missing_aliases():
    client = Mock()
    client.get_registered_model.side_effect = MlflowException("unavailable")
    with pytest.raises(MlflowException, match="unavailable"):
        promote_if_better(client, "MNIST", "best_model", "2", "test/loss", 1.0)
    client.set_registered_model_alias.assert_not_called()
