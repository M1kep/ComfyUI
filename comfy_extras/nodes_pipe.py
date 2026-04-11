import json


MAX_PIPE_KEYS = 20
ANY = "*"


class PipeValue:
    """Runtime value carried on a PIPE wire: a bag of named, typed object refs."""

    def __init__(self, values: dict | None = None, manifest: dict | None = None):
        self.values = dict(values) if values else {}
        self.manifest = dict(manifest) if manifest else {}

    def __repr__(self):
        return f"PipeValue({self.manifest})"


def _parse_manifest(raw):
    """Parse a manifest widget value into an ordered list of {name, type} entries."""
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(entries, list):
        return []
    out = []
    for e in entries:
        if isinstance(e, dict) and "name" in e:
            out.append({"name": str(e["name"]), "type": str(e.get("type", "*"))})
    return out


def _check_duplicates(entries, label):
    seen = set()
    for e in entries:
        if e["name"] in seen:
            return f"{label}: duplicate key '{e['name']}'"
        seen.add(e["name"])
    return None


def _require_pipe(pipe, label):
    if not isinstance(pipe, PipeValue):
        raise ValueError(f"{label}: input is not a PIPE (got {type(pipe).__name__})")


def _check_pipe_contents(pipe, entries, label):
    """Validate at execute time that the runtime pipe satisfies the expected manifest."""
    _require_pipe(pipe, label)
    for e in entries:
        name = e["name"]
        if name not in pipe.values:
            raise ValueError(
                f"{label}: pipe is missing key '{name}' "
                f"(pipe contains: {sorted(pipe.manifest.keys())})"
            )
        expected = e["type"]
        actual = pipe.manifest.get(name, "*")
        if expected != "*" and actual != "*" and expected != actual:
            raise ValueError(
                f"{label}: key '{name}' has type '{actual}', expected '{expected}'"
            )


class Pipe:
    """Source: bundle named, typed inputs into a single PIPE wire.

    Dynamic input slots are managed by the frontend extension; each connected
    slot's name is the pipe key. The ``_manifest`` widget (JSON list of
    ``{name, type}``) is kept in sync by the frontend and tells the backend
    which kwargs to collect and what type each carries.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "_manifest": ("STRING", {"default": "[]", "multiline": False}),
            },
        }

    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "make_pipe"
    CATEGORY = "utils/pipe"
    DESCRIPTION = "Bundle multiple named values into a single PIPE wire."

    @classmethod
    def VALIDATE_INPUTS(cls, input_types, _manifest="[]"):
        entries = _parse_manifest(_manifest)
        err = _check_duplicates(entries, "Pipe")
        if err:
            return err
        for e in entries:
            received = input_types.get(e["name"])
            if received is not None and e["type"] not in ("*", received):
                return f"Pipe: key '{e['name']}' declared as '{e['type']}' but connected to '{received}'"
        return True

    def make_pipe(self, _manifest="[]", **kwargs):
        entries = _parse_manifest(_manifest)
        values = {}
        manifest = {}
        for e in entries:
            name = e["name"]
            if name in kwargs:
                values[name] = kwargs[name]
                manifest[name] = e["type"]
        return (PipeValue(values, manifest),)


class PipeOut:
    """Sink: unpack a PIPE into one typed output per manifest key.

    Output slots are reshaped by the frontend extension based on the upstream
    manifest, which is mirrored into the ``_manifest`` widget so the backend
    knows the key→slot ordering.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE",),
            },
            "optional": {
                "_manifest": ("STRING", {"default": "[]", "multiline": False}),
            },
        }

    RETURN_TYPES = (ANY,) * MAX_PIPE_KEYS
    RETURN_NAMES = tuple(f"value{i}" for i in range(MAX_PIPE_KEYS))
    FUNCTION = "unpack"
    CATEGORY = "utils/pipe"
    DESCRIPTION = "Unpack a PIPE into individual typed outputs."

    @classmethod
    def VALIDATE_INPUTS(cls, _manifest="[]"):
        entries = _parse_manifest(_manifest)
        if len(entries) > MAX_PIPE_KEYS:
            return f"PipeOut: manifest has {len(entries)} keys, max is {MAX_PIPE_KEYS}"
        return _check_duplicates(entries, "PipeOut") or True

    def unpack(self, pipe, _manifest="[]"):
        entries = _parse_manifest(_manifest)
        _check_pipe_contents(pipe, entries, "PipeOut")
        out = [pipe.values[e["name"]] for e in entries]
        out += [None] * (MAX_PIPE_KEYS - len(out))
        return tuple(out)


class PipeSet:
    """Derive a new PIPE with one key added or replaced."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE",),
                "key": ("STRING", {"default": ""}),
            },
            "optional": {
                "value": (ANY,),
                "_value_type": ("STRING", {"default": "*"}),
            },
        }

    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "set_key"
    CATEGORY = "utils/pipe"
    DESCRIPTION = "Add or replace a single key on a PIPE."

    @classmethod
    def VALIDATE_INPUTS(cls, input_types, key=""):
        if not key:
            return "PipeSet: key must not be empty"
        if "value" not in input_types:
            return "PipeSet: 'value' input must be connected"
        return True

    def set_key(self, pipe, key, value=None, _value_type="*"):
        _require_pipe(pipe, "PipeSet")
        new_values = dict(pipe.values)
        new_values[key] = value
        new_manifest = dict(pipe.manifest)
        new_manifest[key] = _value_type
        return (PipeValue(new_values, new_manifest),)


class PipeRemove:
    """Derive a new PIPE with one key dropped."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE",),
                "key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "remove_key"
    CATEGORY = "utils/pipe"
    DESCRIPTION = "Remove a single key from a PIPE."

    @classmethod
    def VALIDATE_INPUTS(cls, key=""):
        if not key:
            return "PipeRemove: key must not be empty"
        return True

    def remove_key(self, pipe, key):
        _require_pipe(pipe, "PipeRemove")
        if key not in pipe.values:
            raise ValueError(
                f"PipeRemove: key '{key}' not present in pipe "
                f"(pipe contains: {sorted(pipe.manifest.keys())})"
            )
        new_values = dict(pipe.values)
        new_manifest = dict(pipe.manifest)
        del new_values[key]
        new_manifest.pop(key, None)
        return (PipeValue(new_values, new_manifest),)


class PipeGet:
    """Extract a single key from a PIPE as a typed output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE",),
                "key": ("STRING", {"default": ""}),
            },
            "optional": {
                "_value_type": ("STRING", {"default": "*"}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_key"
    CATEGORY = "utils/pipe"
    DESCRIPTION = "Extract one key from a PIPE."

    @classmethod
    def VALIDATE_INPUTS(cls, key=""):
        if not key:
            return "PipeGet: key must not be empty"
        return True

    def get_key(self, pipe, key, _value_type="*"):
        _check_pipe_contents(pipe, [{"name": key, "type": _value_type}], "PipeGet")
        return (pipe.values[key],)


class PipeMerge:
    """Union two PIPEs into one, with a configurable collision policy."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("PIPE",),
                "b": ("PIPE",),
                "collision": (["left_wins", "right_wins", "error"], {"default": "left_wins"}),
            },
        }

    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "merge"
    CATEGORY = "utils/pipe"
    DESCRIPTION = "Merge two PIPEs into one."

    def merge(self, a, b, collision):
        _require_pipe(a, "PipeMerge")
        _require_pipe(b, "PipeMerge")
        overlap = set(a.values) & set(b.values)
        if overlap and collision == "error":
            raise ValueError(f"PipeMerge: key collision on {sorted(overlap)}")
        if collision == "right_wins":
            values = {**a.values, **b.values}
            manifest = {**a.manifest, **b.manifest}
        else:
            values = {**b.values, **a.values}
            manifest = {**b.manifest, **a.manifest}
        return (PipeValue(values, manifest),)


WEB_DIRECTORY = "./nodes_pipe_web"

NODE_CLASS_MAPPINGS = {
    "Pipe": Pipe,
    "PipeOut": PipeOut,
    "PipeSet": PipeSet,
    "PipeRemove": PipeRemove,
    "PipeGet": PipeGet,
    "PipeMerge": PipeMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pipe": "Pipe",
    "PipeOut": "Pipe Out",
    "PipeSet": "Pipe Set",
    "PipeRemove": "Pipe Remove",
    "PipeGet": "Pipe Get",
    "PipeMerge": "Pipe Merge",
}
