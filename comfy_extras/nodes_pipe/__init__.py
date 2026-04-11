"""Pipe nodes: a single wire that carries a bag of named, typed values.

Inspired by OpenTelemetry baggage - values propagate along the wire, intermediate
nodes derive new pipes by adding/removing entries, and terminal nodes unpack
entries back into typed outputs.

Wire type is ``PIPE``. Structural typing (per-key types) lives in a manifest
attached to the runtime value and mirrored in editor-time state managed by the
companion JS extension in ``web/pipe_nodes.js``.
"""
from __future__ import annotations

import json
from typing import Any

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


WEB_DIRECTORY = "./web"


# Maximum fanout for `Pipe Out`. Outputs are wildcard-typed on the backend;
# the companion JS extension trims and retypes slots per the upstream manifest.
MAX_PIPE_OUT_SLOTS = 32

# Single wire type. Structural typing travels in the manifest, not the type
# string, so subgraph boundaries stay clean.
PipeType = io.Custom("PIPE")

WILDCARD = "*"


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
        return self.manifest.get(key, WILDCARD)

    def __repr__(self) -> str:
        return f"Pipe({list(self.manifest.items())})"


def _resolve_input_types_from_prompt(prompt: dict | None, unique_id: str | int | None) -> dict[str, str]:
    """Return ``{input_name: received_type}`` by walking upstream ``RETURN_TYPES``.

    The execution prompt dict is the canonical source: each connected input is
    ``[upstream_id, slot_idx]`` and the upstream class's ``RETURN_TYPES`` gives
    us the authoritative type string. Works for every execution mode,
    including API-driven runs where ``extra_pnginfo`` is absent.
    """
    if not prompt or unique_id is None:
        return {}
    node_info = prompt.get(str(unique_id)) or prompt.get(unique_id)
    if not isinstance(node_info, dict):
        return {}

    node_mappings = _node_class_mappings()
    if node_mappings is None:
        return {}

    resolved: dict[str, str] = {}
    for input_name, value in (node_info.get("inputs") or {}).items():
        if not (isinstance(value, list) and len(value) == 2):
            continue
        upstream_id, slot_idx = value
        upstream_info = prompt.get(str(upstream_id)) or prompt.get(upstream_id)
        if not isinstance(upstream_info, dict):
            continue
        cls = node_mappings.get(upstream_info.get("class_type"))
        if cls is None:
            continue
        try:
            return_types = cls.RETURN_TYPES
        except AttributeError:
            continue
        if 0 <= slot_idx < len(return_types):
            resolved[input_name] = str(return_types[slot_idx])
    return resolved


def _node_class_mappings() -> dict | None:
    """Fetch ``nodes.NODE_CLASS_MAPPINGS`` without crashing if unavailable.

    Importing ComfyUI's ``nodes`` module from tests (or from an environment
    without torch) raises; swallow that and return ``None`` so callers can
    degrade gracefully to wildcard types.
    """
    try:
        import nodes as _nodes
    except Exception:
        return None
    return getattr(_nodes, "NODE_CLASS_MAPPINGS", None)


def _check_type(expected: str, actual: str, key: str) -> str | None:
    """Return a mismatch message or None when the pair is compatible."""
    if expected != WILDCARD and actual != WILDCARD and expected != actual:
        return f"key {key!r} expected type {expected!r}, got {actual!r}"
    return None


def _parse_manifest_json(raw: str) -> list[tuple[str, str]]:
    """Parse a JSON-encoded ``[[key, type], ...]`` manifest, dropping garbage."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[tuple[str, str]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
    return out


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
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        # Empty key names can slip in if a user clears a slot label in the
        # editor. The prompt dict itself already collapses exact duplicates.
        for key in kwargs:
            if not key:
                return "Pipe key name cannot be empty"
        return True

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        types = _resolve_input_types_from_prompt(cls.hidden.prompt, cls.hidden.unique_id)
        manifest = {key: types.get(key, WILDCARD) for key in kwargs}
        return io.NodeOutput(Pipe(dict(kwargs), manifest))


class PipeOut(io.ComfyNode):
    """Unpack a PIPE into typed outputs, one per manifest key.

    The ``expected_manifest`` widget is populated by the JS extension with the
    last-reshaped manifest. At execute time we:

    1. Validate the incoming pipe contains every expected key with a
       compatible type. Drift (upstream edited after wiring) fails pre-exec
       rather than silently emitting stale values.
    2. Emit values in the expected order so downstream slot indices map
       stably even if the upstream manifest's insertion order has changed.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PipeOut",
            display_name="Pipe Out",
            category="pipe",
            inputs=[
                PipeType.Input("pipe"),
                io.String.Input(
                    "expected_manifest",
                    default="[]",
                    optional=True,
                    tooltip="JSON [[key, type], ...] snapshot; managed by the editor - do not edit.",
                ),
            ],
            outputs=[
                io.AnyType.Output(id=f"out_{i}", display_name=f"out_{i}")
                for i in range(MAX_PIPE_OUT_SLOTS)
            ],
        )

    @classmethod
    def validate_inputs(cls, pipe=None, expected_manifest: str = "[]") -> bool | str:
        # Manifest is a JS-managed mirror; tolerate malformed values here and
        # defer strict checking to execute() where the real pipe is visible.
        return True

    @classmethod
    def execute(cls, pipe: Pipe, expected_manifest: str = "[]") -> io.NodeOutput:
        expected = _parse_manifest_json(expected_manifest)
        # Fall back to the incoming pipe's own manifest when no snapshot has
        # been persisted yet (freshly-authored node before the first reshape).
        if not expected:
            expected = list(pipe.manifest.items())

        if len(expected) > MAX_PIPE_OUT_SLOTS:
            raise ValueError(
                f"Pipe Out: manifest has {len(expected)} keys but the node supports "
                f"at most {MAX_PIPE_OUT_SLOTS}. Insert a Pipe Remove to trim."
            )

        errors: list[str] = []
        values: list[Any] = [None] * MAX_PIPE_OUT_SLOTS
        for idx, (key, expected_type) in enumerate(expected):
            if not pipe.has(key):
                errors.append(f"missing key {key!r}")
                continue
            mismatch = _check_type(expected_type, pipe.type_of(key), key)
            if mismatch:
                errors.append(mismatch)
            values[idx] = pipe.get(key)

        if errors:
            raise ValueError("Pipe Out manifest mismatch: " + "; ".join(errors))
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
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
        )

    @classmethod
    def validate_inputs(cls, pipe=None, key: str = "", value=None) -> bool | str:
        if not key:
            return "Pipe Set: key must not be empty"
        return True

    @classmethod
    def execute(cls, pipe: Pipe, key: str, value: Any) -> io.NodeOutput:
        types = _resolve_input_types_from_prompt(cls.hidden.prompt, cls.hidden.unique_id)
        return io.NodeOutput(pipe.set(key, value, types.get("value", WILDCARD)))


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
                io.String.Input("expected_type", default=WILDCARD, optional=True),
            ],
            outputs=[io.AnyType.Output(display_name="value")],
        )

    @classmethod
    def validate_inputs(cls, pipe=None, key: str = "", expected_type: str = WILDCARD) -> bool | str:
        if not key:
            return "Pipe Get: key must not be empty"
        return True

    @classmethod
    def execute(cls, pipe: Pipe, key: str, expected_type: str = WILDCARD) -> io.NodeOutput:
        if not pipe.has(key):
            raise ValueError(f"Pipe Get: key {key!r} not present in pipe")
        mismatch = _check_type(expected_type, pipe.type_of(key), key)
        if mismatch:
            raise ValueError(f"Pipe Get: {mismatch}")
        return io.NodeOutput(pipe.get(key))


class PipeMerge(io.ComfyNode):
    """Union of two pipes with a collision policy."""

    COLLISION_LEFT = "left_wins"
    COLLISION_RIGHT = "right_wins"
    COLLISION_ERROR = "error"
    COLLISION_OPTIONS = [COLLISION_LEFT, COLLISION_RIGHT, COLLISION_ERROR]

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
        if collision == cls.COLLISION_ERROR:
            overlap = set(a.entries).intersection(b.entries)
            if overlap:
                keys = ", ".join(sorted(overlap))
                raise ValueError(f"Pipe Merge: colliding keys with collision=error: {keys}")
        return io.NodeOutput(a.merge(b, right_wins=collision == cls.COLLISION_RIGHT))


class PipeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PipeSource, PipeOut, PipeSet, PipeRemove, PipeGet, PipeMerge]


async def comfy_entrypoint() -> PipeExtension:
    return PipeExtension()
