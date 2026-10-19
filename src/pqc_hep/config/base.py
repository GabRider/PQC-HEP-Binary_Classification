"""Base class shared by every configuration section."""

from pydantic import BaseModel, ConfigDict


class Section(BaseModel):
    # frozen: read-only; extra="forbid": typos are errors; strict: no "10" -> 10.
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
