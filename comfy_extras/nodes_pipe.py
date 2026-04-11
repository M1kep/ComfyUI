"""Pipe nodes: a single wire that carries a bag of named, typed values.

Inspired by OpenTelemetry baggage - values propagate along the wire, intermediate
nodes derive new pipes by adding/removing entries, and terminal nodes unpack
entries back into typed outputs.

Wire type is ``PIPE``. Structural typing (per-key types) lives in a manifest
attached to the runtime value and mirrored in editor-time state managed by the
companion JS extension in ``pipe_nodes_web/pipe_nodes.js``.
"""
from __future__ import annotations

from typing import Any

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


WEB_DIRECTORY = "./pipe_nodes_web"


# Maximum fanout for `Pipe Out`. Outputs are wildcard-typed on the backend;
# the companion JS extension trims and retypes slots per the upstream manifest.
MAX_PIPE_OUT_SLOTS = 32

# Single wire type. Structural typing travels in the manifest, not the type
# string, so subgraph boundaries stay clean.
PipeType = io.Custom("PIPE")


class Pipe:
    """Runtime pipe value: ordered entries + parallel type manifest.

    Values are held by reference; functional ops (``set``/``remove``) return
    new ``Pipe`` instances without cloning the underlying values.
    """

    __slots__ = ("entries", "manifest")

    def __init__(self, entries: dict[str, Any] | None = None, manifest: dict[str, str] | None = None):
        self.entries = entries or {}
        self.manifest = manifest or {}

    def set(self, key: str, value: Any, type_str: str) -> "Pipe":
        return Pipe({**self.entries, key: value}, {**self.manifest, key: type_str})

    def remove(self, key: str) -> "Pipe":
        return Pipe(
            {k: v for k, v in self.entries.items() if k != key},
            {k: t for k, t in self.manifest.items() if k != key},
        )

    def merge(self, other: "Pipe", right_wins: bool) -> "Pipe":
        if right_wins:
            return Pipe({**self.entries, **other.entries}, {**self.manifest, **other.manifest})
        return Pipe({**other.entries, **self.entries}, {**other.manifest, **self.manifest})

    def get(self, key: str, default: Any = None) -> Any:
        return self.entries.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.entries

    def type_of(self, key: str) -> str:
        return self.manifest.get(key, "*")

    def __repr__(self) -> str:
        return f"Pipe({list(self.manifest.items())})"


def _workflow_node(extra_pnginfo: dict | None, unique_id: str | int | None) -> dict | None:
    if not extra_pnginfo or unique_id is None:
        return None
    workflow = extra_pnginfo.get("workflow") if isinstance(extra_pnginfo, dict) else None
    if not workflow:
        return None
    target = str(unique_id)
    for node in workflow.get("nodes", []):
        if str(node.get("id")) == target:
            return node
    return None


def _connected_input_types(node_entry: dict | None) -> dict[str, str]:
    if not node_entry:
        return {}
    return {
        inp["name"]: inp.get("type") or "*"
        for inp in (node_entry.get("inputs") or [])
        if inp.get("name") and inp.get("link") is not None
    }


def _check_type(expected: str, actual: str, key: str) -> str | None:
    if expected != "*" and actual != "*" and expected != actual:
        return f"key {key!r} expected type {expected!r}, got {actual!r}"
    return None


class PipeSource(io.ComfyNode):
    """Source node: dynamic named+typed inputs -> single PIPE output."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Pipe",
            display_name="Pipe",
            category="pipe",
            inputs=[],
            outputs=[PipeType.Output(display_name="pipe")],
            accept_all_inputs=True,
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        # Empty key names can slip in if the user clears a slot label in the
        # editor; the prompt dict already dedups by construction.
        for key in kwargs:
            if not key:
                return "Pipe key name cannot be empty"
        return True

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        node_entry = _workflow_node(cls.hidden.extra_pnginfo, cls.hidden.unique_id)
        input_types = _connected_input_types(node_entry)
        manifest = {key: input_types.get(key, "*") for key in kwargs}
        return io.NodeOutput(Pipe(dict(kwargs), manifest))


class PipeOut(io.ComfyNode):
    """Unpack a PIPE into typed outputs, one per manifest key.

    Backend declares ``MAX_PIPE_OUT_SLOTS`` wildcard outputs; the JS extension
    trims and retypes slots to match the upstream manifest.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PipeOut",
            display_name="Pipe Out",
            category="pipe",
            inputs=[PipeType.Input("pipe")],
            outputs=[
                io.AnyType.Output(id=f"out_{i}", display_name=f"out_{i}")
                for i in range(MAX_PIPE_OUT_SLOTS)
            ],
        )

    @classmethod
    def execute(cls, pipe: Pipe) -> io.NodeOutput:
        values: list[Any] = [None] * MAX_PIPE_OUT_SLOTS
        for idx, key in enumerate(list(pipe.manifest)[:MAX_PIPE_OUT_SLOTS]):
            values[idx] = pipe.get(key)
        return io.NodeOutput(*values)


class PipeSet(io.ComfyNode):
    """``pipe + key + value -> new pipe`` with the key added or replaced."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PipeSet",
            display_name="Pipe Set",
            category="pipe",
            inputs=[
                PipeType.Input("pipe"),
                io.String.Input("key", default=""),
                io.AnyType.Input("value"),
            ],
            outputs=[PipeType.Output(display_name="pipe")],
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def validate_inputs(cls, pipe=None, key: str = "", value=None) -> bool | str:
        if not key:
            return "Pipe Set: key must not be empty"
        return True

    @classmethod
    def execute(cls, pipe: Pipe, key: str, value: Any) -> io.NodeOutput:
        node_entry = _workflow_node(cls.hidden.extra_pnginfo, cls.hidden.unique_id)
        value_type = _connected_input_types(node_entry).get("value", "*")
        return io.NodeOutput(pipe.set(key, value, value_type))


class PipeRemove(io.ComfyNode):
    """``pipe + key -> new pipe`` with the key dropped."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PipeRemove",
            display_name="Pipe Remove",
            category="pipe",
            inputs=[
                PipeType.Input("pipe"),
                io.String.Input("key", default=""),
            ],
            outputs=[PipeType.Output(display_name="pipe")],
        )

    @classmethod
    def validate_inputs(cls, pipe=None, key: str = "") -> bool | str:
        if not key:
            return "Pipe Remove: key must not be empty"
        return True

    @classmethod
    def execute(cls, pipe: Pipe, key: str) -> io.NodeOutput:
        if not pipe.has(key):
            raise ValueError(f"Pipe Remove: key {key!r} is not present in the upstream pipe")
        return io.NodeOutput(pipe.remove(key))


class PipeGet(io.ComfyNode):
    """``pipe + key -> single typed output``."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PipeGet",
            display_name="Pipe Get",
            category="pipe",
            inputs=[
                PipeType.Input("pipe"),
                io.String.Input("key", default=""),
                io.String.Input("expected_type", default="*", optional=True),
            ],
            outputs=[io.AnyType.Output(display_name="value")],
        )

    @classmethod
    def validate_inputs(cls, pipe=None, key: str = "", expected_type: str = "*") -> bool | str:
        if not key:
            return "Pipe Get: key must not be empty"
        return True

    @classmethod
    def execute(cls, pipe: Pipe, key: str, expected_type: str = "*") -> io.NodeOutput:
        if not pipe.has(key):
            raise ValueError(f"Pipe Get: key {key!r} not present in pipe")
        mismatch = _check_type(expected_type, pipe.type_of(key), key)
        if mismatch:
            raise ValueError(f"Pipe Get: {mismatch}")
        return io.NodeOutput(pipe.get(key))


class PipeMerge(io.ComfyNode):
    """Union of two pipes with a collision policy."""

    COLLISION_OPTIONS = ["left_wins", "right_wins", "error"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PipeMerge",
            display_name="Pipe Merge",
            category="pipe",
            inputs=[
                PipeType.Input("a"),
                PipeType.Input("b"),
                io.Combo.Input("collision", options=cls.COLLISION_OPTIONS),
            ],
            outputs=[PipeType.Output(display_name="pipe")],
        )

    @classmethod
    def execute(cls, a: Pipe, b: Pipe, collision: str) -> io.NodeOutput:
        if collision == "error":
            overlap = set(a.entries).intersection(b.entries)
            if overlap:
                keys = ", ".join(sorted(overlap))
                raise ValueError(f"Pipe Merge: colliding keys with collision=error: {keys}")
        return io.NodeOutput(a.merge(b, right_wins=(collision == "right_wins")))


class PipeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PipeSource, PipeOut, PipeSet, PipeRemove, PipeGet, PipeMerge]


async def comfy_entrypoint() -> PipeExtension:
    return PipeExtension()
