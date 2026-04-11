import json
import pytest

from comfy_extras.nodes_pipe import (
    MAX_PIPE_KEYS,
    PipeValue,
    Pipe,
    PipeOut,
    PipeSet,
    PipeRemove,
    PipeGet,
    PipeMerge,
    _parse_manifest,
)


def manifest(*pairs):
    return json.dumps([{"name": n, "type": t} for n, t in pairs])


# --- _parse_manifest -------------------------------------------------------

class TestParseManifest:
    def test_empty(self):
        assert _parse_manifest("") == []
        assert _parse_manifest("[]") == []
        assert _parse_manifest(None) == []

    def test_basic(self):
        assert _parse_manifest(manifest(("a", "MODEL"))) == [{"name": "a", "type": "MODEL"}]

    def test_garbage(self):
        assert _parse_manifest("not json") == []
        assert _parse_manifest('{"a": 1}') == []
        assert _parse_manifest("[1, 2]") == []

    def test_missing_type_defaults_any(self):
        assert _parse_manifest('[{"name": "x"}]') == [{"name": "x", "type": "*"}]


# --- Pipe (source) ---------------------------------------------------------

class TestPipe:
    def test_make_pipe(self):
        m = manifest(("model", "MODEL"), ("clip", "CLIP"))
        (pv,) = Pipe().make_pipe(_manifest=m, model="M", clip="C")
        assert isinstance(pv, PipeValue)
        assert pv.values == {"model": "M", "clip": "C"}
        assert pv.manifest == {"model": "MODEL", "clip": "CLIP"}

    def test_make_pipe_ignores_undeclared_kwargs(self):
        m = manifest(("model", "MODEL"))
        (pv,) = Pipe().make_pipe(_manifest=m, model="M", extra="E")
        assert "extra" not in pv.values

    def test_make_pipe_missing_kwarg(self):
        m = manifest(("model", "MODEL"), ("clip", "CLIP"))
        (pv,) = Pipe().make_pipe(_manifest=m, model="M")
        assert pv.values == {"model": "M"}

    def test_validate_duplicate_keys(self):
        m = manifest(("a", "INT"), ("a", "INT"))
        assert "duplicate key 'a'" in Pipe.VALIDATE_INPUTS({}, m)

    def test_validate_type_mismatch(self):
        m = manifest(("a", "INT"))
        assert "declared as 'INT'" in Pipe.VALIDATE_INPUTS({"a": "FLOAT"}, m)

    def test_validate_ok(self):
        m = manifest(("a", "INT"))
        assert Pipe.VALIDATE_INPUTS({"a": "INT"}, m) is True

    def test_validate_any_accepts(self):
        m = manifest(("a", "*"))
        assert Pipe.VALIDATE_INPUTS({"a": "MODEL"}, m) is True


# --- PipeOut ---------------------------------------------------------------

class TestPipeOut:
    def test_unpack(self):
        pv = PipeValue({"a": 1, "b": 2}, {"a": "INT", "b": "INT"})
        out = PipeOut().unpack(pv, manifest(("a", "INT"), ("b", "INT")))
        assert out[0] == 1
        assert out[1] == 2
        assert len(out) == MAX_PIPE_KEYS
        assert all(v is None for v in out[2:])

    def test_unpack_missing_key(self):
        pv = PipeValue({"a": 1}, {"a": "INT"})
        with pytest.raises(ValueError, match="missing key 'b'"):
            PipeOut().unpack(pv, manifest(("a", "INT"), ("b", "INT")))

    def test_unpack_type_mismatch(self):
        pv = PipeValue({"a": 1}, {"a": "FLOAT"})
        with pytest.raises(ValueError, match="has type 'FLOAT', expected 'INT'"):
            PipeOut().unpack(pv, manifest(("a", "INT")))

    def test_unpack_any_skips_type_check(self):
        pv = PipeValue({"a": 1}, {"a": "FLOAT"})
        out = PipeOut().unpack(pv, manifest(("a", "*")))
        assert out[0] == 1

    def test_unpack_not_pipe(self):
        with pytest.raises(ValueError, match="not a PIPE"):
            PipeOut().unpack({"a": 1}, manifest(("a", "INT")))

    def test_validate_too_many_keys(self):
        m = json.dumps([{"name": f"k{i}", "type": "*"} for i in range(MAX_PIPE_KEYS + 1)])
        assert "max is" in PipeOut.VALIDATE_INPUTS(m)

    def test_validate_duplicate(self):
        m = manifest(("a", "INT"), ("a", "INT"))
        assert "duplicate" in PipeOut.VALIDATE_INPUTS(m)

    def test_return_shape(self):
        assert len(PipeOut.RETURN_TYPES) == MAX_PIPE_KEYS
        assert len(PipeOut.RETURN_NAMES) == MAX_PIPE_KEYS


# --- PipeSet ---------------------------------------------------------------

class TestPipeSet:
    def test_add(self):
        pv = PipeValue({"a": 1}, {"a": "INT"})
        (out,) = PipeSet().set_key(pv, "b", value=2, _value_type="INT")
        assert out.values == {"a": 1, "b": 2}
        assert out.manifest == {"a": "INT", "b": "INT"}
        assert pv.values == {"a": 1}

    def test_replace(self):
        pv = PipeValue({"a": 1}, {"a": "INT"})
        (out,) = PipeSet().set_key(pv, "a", value="x", _value_type="STRING")
        assert out.values == {"a": "x"}
        assert out.manifest == {"a": "STRING"}

    def test_validate_empty_key(self):
        assert "key must not be empty" in PipeSet.VALIDATE_INPUTS({"value": "INT"}, "")

    def test_validate_value_unconnected(self):
        assert "must be connected" in PipeSet.VALIDATE_INPUTS({}, "k")

    def test_validate_ok(self):
        assert PipeSet.VALIDATE_INPUTS({"value": "INT"}, "k") is True

    def test_not_pipe(self):
        with pytest.raises(ValueError, match="not a PIPE"):
            PipeSet().set_key("oops", "k", value=1)


# --- PipeRemove ------------------------------------------------------------

class TestPipeRemove:
    def test_remove(self):
        pv = PipeValue({"a": 1, "b": 2}, {"a": "INT", "b": "INT"})
        (out,) = PipeRemove().remove_key(pv, "a")
        assert out.values == {"b": 2}
        assert out.manifest == {"b": "INT"}
        assert pv.values == {"a": 1, "b": 2}

    def test_remove_missing(self):
        pv = PipeValue({"a": 1}, {"a": "INT"})
        with pytest.raises(ValueError, match="key 'b' not present"):
            PipeRemove().remove_key(pv, "b")

    def test_validate_empty_key(self):
        assert "key must not be empty" in PipeRemove.VALIDATE_INPUTS("")


# --- PipeGet ---------------------------------------------------------------

class TestPipeGet:
    def test_get(self):
        pv = PipeValue({"a": 1}, {"a": "INT"})
        assert PipeGet().get_key(pv, "a") == (1,)

    def test_get_missing(self):
        pv = PipeValue({"a": 1}, {"a": "INT"})
        with pytest.raises(ValueError, match="missing key 'b'"):
            PipeGet().get_key(pv, "b")

    def test_get_type_mismatch(self):
        pv = PipeValue({"a": 1}, {"a": "FLOAT"})
        with pytest.raises(ValueError, match="expected 'INT'"):
            PipeGet().get_key(pv, "a", _value_type="INT")

    def test_validate_empty_key(self):
        assert "key must not be empty" in PipeGet.VALIDATE_INPUTS("")


# --- PipeMerge -------------------------------------------------------------

class TestPipeMerge:
    def test_left_wins(self):
        a = PipeValue({"x": 1, "y": 2}, {"x": "INT", "y": "INT"})
        b = PipeValue({"y": 99, "z": 3}, {"y": "FLOAT", "z": "INT"})
        (out,) = PipeMerge().merge(a, b, "left_wins")
        assert out.values == {"x": 1, "y": 2, "z": 3}
        assert out.manifest["y"] == "INT"

    def test_right_wins(self):
        a = PipeValue({"x": 1, "y": 2}, {"x": "INT", "y": "INT"})
        b = PipeValue({"y": 99, "z": 3}, {"y": "FLOAT", "z": "INT"})
        (out,) = PipeMerge().merge(a, b, "right_wins")
        assert out.values == {"x": 1, "y": 99, "z": 3}
        assert out.manifest["y"] == "FLOAT"

    def test_error(self):
        a = PipeValue({"x": 1}, {"x": "INT"})
        b = PipeValue({"x": 2}, {"x": "INT"})
        with pytest.raises(ValueError, match="collision"):
            PipeMerge().merge(a, b, "error")

    def test_no_collision(self):
        a = PipeValue({"x": 1}, {"x": "INT"})
        b = PipeValue({"y": 2}, {"y": "INT"})
        (out,) = PipeMerge().merge(a, b, "error")
        assert out.values == {"x": 1, "y": 2}

    def test_not_pipe(self):
        with pytest.raises(ValueError, match="not a PIPE"):
            PipeMerge().merge("a", PipeValue(), "left_wins")


# --- Functional immutability ----------------------------------------------

class TestImmutability:
    def test_values_are_referenced_not_copied(self):
        big = object()
        pv = PipeValue({"m": big}, {"m": "MODEL"})
        (out,) = PipeSet().set_key(pv, "x", value=1, _value_type="INT")
        assert out.values["m"] is big
