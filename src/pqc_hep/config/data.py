"""[data] table: input files, event selection and ordering, output folder."""

from pathlib import Path
from typing import Literal

from pydantic import Field, FilePath, PositiveInt, model_validator

from pqc_hep.config.base import Section


class DataConfig(Section):
    # signal variable declaration
    sig_path: FilePath = Field(Path("Data/train_sig.csv"), strict=False)
    bkg_path: FilePath = Field(Path("Data/train_bkg.csv"), strict=False)
    max_events: PositiveInt | Literal["all"] = 1000
    # validation set declaration
    val_sig_path: FilePath | None = Field(None, strict=False)
    val_bkg_path: FilePath | None = Field(None, strict=False)
    n_val: PositiveInt = 2000
    # data manipulation options
    feature_scaling: Literal["zscore", "none"] = "none"
    balance_classes: bool = False
    event_order: Literal["random", "signal_first", "background_first"] = "random"
    reshuffle_between_passes: bool = False
    # output information
    results_dir: Path = Field(Path("result"), strict=False)

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

    @model_validator(mode="after")
    def _val_needs_both_files(self):
        if (self.val_sig_path is None) != (self.val_bkg_path is None):
            raise ValueError("val_sig_path and val_bkg_path must be given together.")
        return self
