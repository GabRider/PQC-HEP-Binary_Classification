"""Run configuration loaded from TOML. Every key is optional; see example.toml."""

import tomllib

from pydantic import Field

from pqc_hep.config.base import Section
from pqc_hep.config.data import DataConfig
from pqc_hep.config.model import ModelConfig
from pqc_hep.config.seeds import SeedsConfig


class Config(Section):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    seeds: SeedsConfig = Field(default_factory=SeedsConfig)  # fresh seeds per Config


def load(path: str | None = None) -> Config:
    if path is None:
        return Config()
    with open(path, "rb") as f:
        return Config.model_validate(tomllib.load(f))
