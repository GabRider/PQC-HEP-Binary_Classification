"""[seeds] table: every random seed of the run."""

import secrets

from pydantic import Field

from pqc_hep.config.base import Section


def _random_seed() -> int:
    return secrets.randbits(32)


class SeedsConfig(Section):
    # Missing seeds are drawn at random and saved in the TOML copy of the run folder.
    data_shuffle: int = Field(default_factory=_random_seed)
    theta_init: int = Field(default_factory=_random_seed)
    pass_reshuffle: int = Field(default_factory=_random_seed)
    simulator: int = Field(default_factory=_random_seed, ge=0)
    validation: int = Field(default_factory=_random_seed)
