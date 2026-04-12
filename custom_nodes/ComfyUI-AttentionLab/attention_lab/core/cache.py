from __future__ import annotations


class MutableCache:
    """Shared mutable container returned empty by recorder nodes and filled by
    forward hooks during sampling. See spec §0.4 for the barrier pattern."""

    def __init__(self, model_key: str = "", meta: dict | None = None):
        self.data: dict = {}  # {BlockId: {t_bin: tensor}}
        self.model_key = model_key
        self.meta = meta or {}

    def is_filled(self) -> bool:
        return bool(self.data)

    def clear(self) -> None:
        self.data.clear()

    def __repr__(self) -> str:
        n = sum(len(v) for v in self.data.values())
        return f"<MutableCache blocks={len(self.data)} entries={n} key={self.model_key!r}>"
