"""[data] table: input files, event selection and ordering, output tag."""

from pathlib import Path
from typing import Literal

from pydantic import Field, FilePath, PositiveInt, model_validator

from pqc_hep.config.base import Section


class DataConfig(Section):
    # strict=False lets TOML strings become paths; the files must exist.
    sig_path: FilePath = Field(Path("Data/train_sig.csv"), strict=False)
    bkg_path: FilePath = Field(Path("Data/train_bkg.csv"), strict=False)
    max_events: PositiveInt | Literal["all"] = 1000
    balance_classes: bool = False  # equal S and B counts; needs max_events
    event_order: Literal["random", "signal_first", "background_first"] = "random"
    reshuffle_between_passes: bool = False
    run_tag: str = ""  # output folder prefix

    @property
    def event_limit(self) -> int | None:
        """max_events as a number of events; None = "all"."""
        return None if self.max_events == "all" else self.max_events

    @model_validator(mode="after")
    def _reshuffle_needs_random_order(self):
        if self.reshuffle_between_passes and self.event_order != "random":
            raise ValueError(
                "reshuffle_between_passes=true requires event_order='random'."
            )
        return self

    @model_validator(mode="after")
    def _balance_needs_max_events(self):
        if self.balance_classes and self.max_events == "all":
            raise ValueError("balance_classes=true requires max_events to be a number.")
        return self
