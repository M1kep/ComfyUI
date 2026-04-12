import json


WEB_DIRECTORY = "./js"

ANY = "*"


class _UnboundedTypeList(list):
    """RETURN_TYPES helper that yields ``*`` for any out-of-range index so
    downstream type validation accepts dynamically-added output slots."""

    def __getitem__(self, i):
        if isinstance(i, slice):
            return list.__getitem__(self, i)
        if isinstance(i, int) and (i >= len(self) or i < -len(self)):
            return ANY
        return list.__getitem__(self, i)


def _parse_manifest(raw):
    """Parse a manifest string (JSON list of ``[key, type, ...]``) into an
    ordered list of (key, type) pairs. Tolerates legacy dict form and ignores
    any extra elements (e.g. nested manifests carried for the frontend)."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if isinstance(data, dict):
        return [(str(k), str(v)) for k, v in data.items()]
    if isinstance(data, list):
        out = []
        for entry in data:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                out.append((str(entry[0]), str(entry[1])))
        return out
    return []


class PipeValue:
    """Runtime pipe: a bag of named values plus a parallel type manifest.

    Values are stored by reference; deriving a new pipe (set / remove / merge)
    shallow-copies the dicts but never the underlying objects."""

    __slots__ = ("values", "types")

    def __init__(self, values=None, types=None):
        self.values = dict(values) if values else {}
        self.types = dict(types) if types else {}

    def keys(self):
        return self.values.keys()

    def manifest(self):
        return [(k, self.types.get(k, ANY)) for k in self.values]

    def with_set(self, key, value, type_string):
        out = PipeValue(self.values, self.types)
        out.values[key] = value
        out.types[key] = type_string or ANY
        return out

    def without(self, key):
        out = PipeValue(self.values, self.types)
        out.values.pop(key, None)
        out.types.pop(key, None)
        return out

    def merged(self, other, collision):
        out = PipeValue(self.values, self.types)
        for k in other.values:
            if k in out.values:
                if collision == "left_wins":
                    continue
                if collision == "error":
                    raise PipeError(
                        f"Pipe Merge: key '{k}' present in both inputs (collision=error)"
                    )
            out.values[k] = other.values[k]
            out.types[k] = other.types.get(k, ANY)
        return out

    def __repr__(self):
        return f"PipeValue({self.manifest()})"


class PipeError(ValueError):
    pass


def _require_pipe(pipe, node_name):
    if not isinstance(pipe, PipeValue):
        raise PipeError(f"{node_name}: input is not a PIPE (got {type(pipe).__name__})")
    return pipe


def _check_type(node_name, key, expected, actual):
    if expected in (None, "", ANY) or actual in (None, "", ANY):
        return
    if expected != actual:
        raise PipeError(
            f"{node_name}: key '{key}' expected type {expected} but pipe carries {actual}"
        )


class PipeCreate:
    """Source node. Dynamic named/typed inputs (managed by the JS extension)
    are bundled into a single ``PIPE`` output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "_manifest": ("STRING", {"default": "[]", "multiline": False}),
            },
        }

    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "execute"
    CATEGORY = "utils/pipe"

    @classmethod
    def VALIDATE_INPUTS(cls, _manifest="[]", **_):
        seen = set()
        for key, _type in _parse_manifest(_manifest):
            if not key:
                return "Pipe: empty key name"
            if key in seen:
                return f"Pipe: duplicate key '{key}'"
            seen.add(key)
        return True

    def execute(self, _manifest="[]", **kwargs):
        manifest = _parse_manifest(_manifest)
        types = {k: t for k, t in manifest}
        # kwargs are the dynamically-added input slots, named after the keys.
        values = {k: kwargs[k] for k, _ in manifest if k in kwargs}
        return (PipeValue(values, types),)


class PipeOut:
    """Unpacks a ``PIPE`` into one typed output per manifest key, plus a
    passthrough ``pipe`` output at slot 0 so the wire can keep flowing.
    Output slots are added by the JS extension; the persisted ``_manifest``
    widget records their order so the returned tuple lines up."""

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

    RETURN_TYPES = _UnboundedTypeList(["PIPE", ANY])
    RETURN_NAMES = ("pipe", "*")
    FUNCTION = "execute"
    CATEGORY = "utils/pipe"

    def execute(self, pipe, _manifest="[]"):
        pipe = _require_pipe(pipe, "Pipe Out")
        manifest = _parse_manifest(_manifest)
        if not manifest:
            manifest = pipe.manifest()
        out = [pipe]
        for key, expected in manifest:
            if key not in pipe.values:
                raise PipeError(
                    f"Pipe Out: key '{key}' not present in pipe "
                    f"(have: {sorted(pipe.keys())})"
                )
            _check_type("Pipe Out", key, expected, pipe.types.get(key))
            out.append(pipe.values[key])
        return tuple(out)


class PipeSet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE",),
                "key": ("STRING", {"default": ""}),
                "value": (ANY,),
            },
            "optional": {
                "_value_type": ("STRING", {"default": ANY, "multiline": False}),
            },
        }

    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "execute"
    CATEGORY = "utils/pipe"

    @classmethod
    def VALIDATE_INPUTS(cls, key="", **_):
        if not key:
            return "Pipe Set: key must not be empty"
        return True

    def execute(self, pipe, key, value, _value_type: str = ANY):
        pipe = _require_pipe(pipe, "Pipe Set")
        return (pipe.with_set(key, value, _value_type),)


class PipeRemove:
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
    FUNCTION = "execute"
    CATEGORY = "utils/pipe"

    def execute(self, pipe, key):
        pipe = _require_pipe(pipe, "Pipe Remove")
        if key not in pipe.values:
            raise PipeError(
                f"Pipe Remove: key '{key}' not present in pipe "
                f"(have: {sorted(pipe.keys())})"
            )
        return (pipe.without(key),)


class PipeGet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE",),
                "key": ("STRING", {"default": ""}),
            },
            "optional": {
                "_value_type": ("STRING", {"default": ANY, "multiline": False}),
            },
        }

    RETURN_TYPES = _UnboundedTypeList([ANY])
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "utils/pipe"

    def execute(self, pipe, key, _value_type: str = ANY):
        pipe = _require_pipe(pipe, "Pipe Get")
        if key not in pipe.values:
            raise PipeError(
                f"Pipe Get: key '{key}' not present in pipe "
                f"(have: {sorted(pipe.keys())})"
            )
        _check_type("Pipe Get", key, _value_type, pipe.types.get(key))
        return (pipe.values[key],)


class PipeMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("PIPE",),
                "b": ("PIPE",),
                "collision": (["right_wins", "left_wins", "error"],),
            },
        }

    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "execute"
    CATEGORY = "utils/pipe"

    def execute(self, a, b, collision):
        a = _require_pipe(a, "Pipe Merge")
        b = _require_pipe(b, "Pipe Merge")
        return (a.merged(b, collision),)


class PipePick:
    """Selectively unpack a subset of pipe keys. The JS extension adds one
    growing key dropdown per output; the chosen keys are persisted in the
    ``_manifest`` widget so the returned tuple lines up with the slots."""

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

    RETURN_TYPES = _UnboundedTypeList(["PIPE", ANY])
    RETURN_NAMES = ("pipe", "*")
    FUNCTION = "execute"
    CATEGORY = "utils/pipe"

    def execute(self, pipe, _manifest="[]"):
        pipe = _require_pipe(pipe, "Pipe Pick")
        out = [pipe]
        for key, expected in _parse_manifest(_manifest):
            if key not in pipe.values:
                raise PipeError(
                    f"Pipe Pick: key '{key}' not present in pipe "
                    f"(have: {sorted(pipe.keys())})"
                )
            _check_type("Pipe Pick", key, expected, pipe.types.get(key))
            out.append(pipe.values[key])
        return tuple(out)


class PipePreview:
    """Render a pipe's manifest and a short repr of each value as text in the
    UI — useful for debugging mid-chain. Passes the pipe through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("PIPE",)}}

    RETURN_TYPES = ("PIPE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "utils/pipe"

    @staticmethod
    def _format(pipe):
        if not pipe.values:
            return "(empty pipe)"
        rows = []
        for k, t in pipe.manifest():
            v = pipe.values.get(k)
            try:
                rep = repr(v)
            except Exception:
                rep = f"<{type(v).__name__}>"
            if len(rep) > 60:
                rep = rep[:57] + "..."
            rows.append(f"{k}: {t} = {rep}")
        return "\n".join(rows)

    def execute(self, pipe):
        pipe = _require_pipe(pipe, "Pipe Preview")
        return {"ui": {"text": (self._format(pipe),)}, "result": (pipe,)}


NODE_CLASS_MAPPINGS = {
    "PipeCreate": PipeCreate,
    "PipeOut": PipeOut,
    "PipePick": PipePick,
    "PipePreview": PipePreview,
    "PipeSet": PipeSet,
    "PipeRemove": PipeRemove,
    "PipeGet": PipeGet,
    "PipeMerge": PipeMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PipeCreate": "Pipe",
    "PipeOut": "Pipe Out",
    "PipePick": "Pipe Pick",
    "PipePreview": "Pipe Preview",
    "PipeSet": "Pipe Set",
    "PipeRemove": "Pipe Remove",
    "PipeGet": "Pipe Get",
    "PipeMerge": "Pipe Merge",
}
