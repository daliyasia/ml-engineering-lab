from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.utils.rich_utils import enforce_tags, print_config_tree
from src.utils.utils import get_metric_value


def test_metric_value_optional_and_missing():
    assert get_metric_value({}, None) is None
    assert get_metric_value({"score": torch.tensor(0.75)}, "score") == 0.75
    with pytest.raises(Exception, match="Metric value not found"):
        get_metric_value({}, "score")


def test_config_utilities_save_output(cfg_train, monkeypatch):
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.tags = []
    monkeypatch.setattr("src.utils.rich_utils.Prompt.ask", lambda *args, **kwargs: "dev, test")

    enforce_tags(cfg_train, save_to_file=True)
    print_config_tree(cfg_train, save_to_file=True)

    assert list(cfg_train.tags) == ["dev", "test"]
    output_dir = Path(cfg_train.paths.output_dir)
    assert "dev" in (output_dir / "tags.log").read_text()
    assert "CONFIG" in (output_dir / "config_tree.log").read_text()
