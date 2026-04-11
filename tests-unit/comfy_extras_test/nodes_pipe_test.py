import json

import pytest

from comfy_extras.nodes_pipe import (
    PipeCreate,
    PipeError,
    PipeGet,
    PipeMerge,
    PipeOut,
    PipeRemove,
    PipeSet,
    PipeValue,
    _UnboundedTypeList,
    _parse_manifest,
)


# ---------------------------------------------------------------------------
# helpers / value type
# ---------------------------------------------------------------------------

def test_unbounded_type_list_indexes_past_end():
    rt = _UnboundedTypeList(["*"])
    assert rt[0] == "*"
    assert rt[5] == "*"
    assert rt[-100] == "*"
    assert len(rt) == 1
    assert json.dumps(rt) == '["*"]'


def test_parse_manifest_forms():
    assert _parse_manifest("") == []
    assert _parse_manifest(None) == []
    assert _parse_manifest("not json") == []
    assert _parse_manifest('[["a","MODEL"],["b","CLIP"]]') == [
        ("a", "MODEL"), ("b", "CLIP"),
    ]
    assert _parse_manifest('{"a":"MODEL"}') == [("a", "MODEL")]


def test_pipe_value_is_reference_preserving():
    big = object()
    p = PipeValue({"m": big}, {"m": "MODEL"})
    p2 = p.with_set("c", "x", "CLIP")
    assert p2.values["m"] is big
    assert "c" not in p.values  # original unchanged
    p3 = p2.without("m")
    assert "m" not in p3.values and "c" in p3.values


# ---------------------------------------------------------------------------
# PipeCreate
# ---------------------------------------------------------------------------

def test_pipe_create_builds_value_from_manifest_and_kwargs():
    manifest = json.dumps([["model", "MODEL"], ["clip", "CLIP"]])
    (pipe,) = PipeCreate().execute(_manifest=manifest, model="M", clip="C")
    assert pipe.values == {"model": "M", "clip": "C"}
    assert pipe.types == {"model": "MODEL", "clip": "CLIP"}


def test_pipe_create_ignores_unmanifested_kwargs():
    manifest = json.dumps([["model", "MODEL"]])
    (pipe,) = PipeCreate().execute(_manifest=manifest, model="M", stray="?")
    assert list(pipe.keys()) == ["model"]


def test_pipe_create_validate_rejects_duplicate_keys():
    bad = json.dumps([["a", "MODEL"], ["a", "CLIP"]])
    assert PipeCreate.VALIDATE_INPUTS(_manifest=bad) != True  # noqa: E712
    good = json.dumps([["a", "MODEL"], ["b", "CLIP"]])
    assert PipeCreate.VALIDATE_INPUTS(_manifest=good) is True


# ---------------------------------------------------------------------------
# PipeOut
# ---------------------------------------------------------------------------

def test_pipe_out_unpacks_in_manifest_order():
    p = PipeValue({"a": 1, "b": 2}, {"a": "INT", "b": "INT"})
    out = PipeOut().execute(p, _manifest=json.dumps([["b", "INT"], ["a", "INT"]]))
    assert out == (2, 1)


def test_pipe_out_missing_key_raises():
    p = PipeValue({"a": 1}, {"a": "INT"})
    with pytest.raises(PipeError, match="key 'b' not present"):
        PipeOut().execute(p, _manifest=json.dumps([["b", "INT"]]))


def test_pipe_out_type_mismatch_raises():
    p = PipeValue({"a": 1}, {"a": "INT"})
    with pytest.raises(PipeError, match="expected type MODEL"):
        PipeOut().execute(p, _manifest=json.dumps([["a", "MODEL"]]))


def test_pipe_out_empty_manifest_falls_back_to_pipe_order():
    p = PipeValue({"a": 1, "b": 2}, {})
    out = PipeOut().execute(p, _manifest="[]")
    assert out == (1, 2)


# ---------------------------------------------------------------------------
# PipeSet / PipeRemove / PipeGet
# ---------------------------------------------------------------------------

def test_pipe_set_adds_and_replaces():
    p = PipeValue({"a": 1}, {"a": "INT"})
    (p2,) = PipeSet().execute(p, "b", "x", _value_type="STRING")
    assert p2.values == {"a": 1, "b": "x"}
    assert p2.types["b"] == "STRING"
    (p3,) = PipeSet().execute(p2, "a", 99, _value_type="FLOAT")
    assert p3.values["a"] == 99 and p3.types["a"] == "FLOAT"
    assert p.values == {"a": 1}  # original untouched


def test_pipe_set_validate_rejects_empty_key():
    assert PipeSet.VALIDATE_INPUTS(key="") != True  # noqa: E712
    assert PipeSet.VALIDATE_INPUTS(key="a") is True


def test_pipe_remove():
    p = PipeValue({"a": 1, "b": 2}, {"a": "INT", "b": "INT"})
    (p2,) = PipeRemove().execute(p, "a")
    assert list(p2.keys()) == ["b"]
    with pytest.raises(PipeError, match="not present"):
        PipeRemove().execute(p2, "a")


def test_pipe_get():
    p = PipeValue({"a": 1}, {"a": "INT"})
    assert PipeGet().execute(p, "a", _value_type="INT") == (1,)
    with pytest.raises(PipeError, match="not present"):
        PipeGet().execute(p, "b")
    with pytest.raises(PipeError, match="expected type MODEL"):
        PipeGet().execute(p, "a", _value_type="MODEL")


# ---------------------------------------------------------------------------
# PipeMerge
# ---------------------------------------------------------------------------

def test_pipe_merge_policies():
    a = PipeValue({"x": 1, "y": 2}, {"x": "INT", "y": "INT"})
    b = PipeValue({"y": 20, "z": 30}, {"y": "FLOAT", "z": "INT"})

    (right,) = PipeMerge().execute(a, b, "right_wins")
    assert right.values == {"x": 1, "y": 20, "z": 30}
    assert right.types["y"] == "FLOAT"

    (left,) = PipeMerge().execute(a, b, "left_wins")
    assert left.values == {"x": 1, "y": 2, "z": 30}
    assert left.types["y"] == "INT"

    with pytest.raises(PipeError, match="present in both"):
        PipeMerge().execute(a, b, "error")


def test_pipe_merge_disjoint_error_ok():
    a = PipeValue({"x": 1}, {"x": "INT"})
    b = PipeValue({"y": 2}, {"y": "INT"})
    (m,) = PipeMerge().execute(a, b, "error")
    assert m.values == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# end-to-end chain
# ---------------------------------------------------------------------------

def test_chain_create_set_remove_out():
    manifest = json.dumps([["model", "MODEL"], ["clip", "CLIP"]])
    (p,) = PipeCreate().execute(_manifest=manifest, model="M", clip="C")
    (p,) = PipeSet().execute(p, "vae", "V", _value_type="VAE")
    (p,) = PipeRemove().execute(p, "clip")
    out = PipeOut().execute(
        p, _manifest=json.dumps([["model", "MODEL"], ["vae", "VAE"]])
    )
    assert out == ("M", "V")
