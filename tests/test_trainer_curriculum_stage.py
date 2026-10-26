from pathlib import Path

import pytest
import torch
import yaml

from training.mappo_trainer import MAPPOTrainer

ROOT = Path(__file__).resolve().parents[1]


class DummyEnv:
    def __init__(self):
        self.curriculum_stage = None

    def set_curriculum_stage(self, stage):
        self.curriculum_stage = stage


def build_trainer(config_name, tmp_path, monkeypatch):
    config_path = ROOT / "configs" / config_name
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["logging"]["checkpoint_dir"] = str(tmp_path / "checkpoints")

    # Prevent tests from opening a WandB logging session.
    monkeypatch.setattr(MAPPOTrainer, "_init_wandb", lambda self: False)

    env = DummyEnv()
    actor = torch.nn.Linear(1, 1)
    critic = torch.nn.Linear(1, 1)

    trainer = MAPPOTrainer(
        config, actor, critic, env=env, device=torch.device("cpu")
    )
    return config, trainer, env


def test_default_config_starts_trainer_and_environment_at_stage1(
    tmp_path, monkeypatch
):
    config, trainer, env = build_trainer(
        "default_config.yaml", tmp_path, monkeypatch
    )

    assert "initial_curriculum_stage" not in config["training"]
    assert trainer.curriculum_stage == 1
    assert env.curriculum_stage == 1


def test_osm_stage4_config_starts_trainer_and_environment_at_stage4(
    tmp_path, monkeypatch
):
    config, trainer, env = build_trainer(
        "config_stage3_osm_ecity.yaml", tmp_path, monkeypatch
    )

    assert config["training"]["initial_curriculum_stage"] == 4
    assert trainer.curriculum_stage == 4
    assert env.curriculum_stage == 4


def test_invalid_initial_curriculum_stage_is_rejected(tmp_path, monkeypatch):
    config = yaml.safe_load(
        (ROOT / "configs" / "default_config.yaml").read_text(encoding="utf-8")
    )
    config["training"]["initial_curriculum_stage"] = 5
    config["logging"]["checkpoint_dir"] = str(tmp_path / "checkpoints")

    monkeypatch.setattr(MAPPOTrainer, "_init_wandb", lambda self: False)

    with pytest.raises(ValueError, match="initial_curriculum_stage"):
        MAPPOTrainer(
            config,
            torch.nn.Linear(1, 1),
            torch.nn.Linear(1, 1),
            env=DummyEnv(),
            device=torch.device("cpu"),
        )
