"""Base class shared by every configuration section."""

from pydantic import BaseModel, ConfigDict


class Section(BaseModel):
    # frozen: read-only; extra="forbid": typos are errors; strict: no "10" -> 10.
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    def lines(self, table=""):
        """Printed configuration: a [table] header, then one "key : value" per field."""
        fields = [(k, v) for k, v in self if not isinstance(v, Section)]
        if fields:
            yield f"[{table}]"
            yield from (f"  {key:<26}: {value}" for key, value in fields)
        for key, value in self:
            if isinstance(value, Section):
                yield from value.lines(f"{table}.{key}" if table else key)

    def __str__(self):
        return "\n".join(self.lines())
