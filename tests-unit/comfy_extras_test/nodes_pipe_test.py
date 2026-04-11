"""Unit tests for the Pipe node pack.

Covers the pure-Python ``Pipe`` value type and the execute paths of the six
pipe nodes. The tests stub out ``nodes`` and ``server`` imports the same way
``nodes_math_test.py`` does so the module loads without a full ComfyUI
runtime.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# comfy_extras.nodes_pipe does not reference ComfyUI's root `nodes` module at
# import time, so we can import it without the patch.dict dance other tests
# (nodes_math_test.py) need. Using patch.dict here would pop the module from
# sys.modules on unwind, forcing re-imports on later lookups - those re-imports
# fail because torch doesn't survive re-initialization in this environment.
from comfy_extras.nodes_pipe import (
    MAX_PIPE_OUT_SLOTS,
    Pipe,
    PipeGet,
    PipeMerge,
    PipeOut,
    PipeRemove,
    PipeSet,
    PipeSource,
    _check_type,
)


class TestPipe:
    def test_empty_construction(self):
        p = Pipe()
        assert p.entries == {}
        assert p.manifest == {}
        assert not p.has("x")
        assert p.get("x") is None
        assert p.get("x", default=7) == 7

    def test_construction_from_dicts(self):
        p = Pipe({"model": "M", "clip": "C"}, {"model": "MODEL", "clip": "CLIP"})
        assert p.has("model") and p.has("clip")
        assert p.get("model") == "M"
        assert p.type_of("model") == "MODEL"
        assert p.type_of("missing") == "*"

    def test_set_returns_new_pipe_and_does_not_mutate(self):
        p = Pipe({"a": 1}, {"a": "INT"})
        p2 = p.set("b", 2, "INT")
        assert p is not p2
        assert p.entries == {"a": 1}
        assert p.manifest == {"a": "INT"}
        assert p2.entries == {"a": 1, "b": 2}
        assert p2.manifest == {"a": "INT", "b": "INT"}

    def test_set_replaces_existing_key_and_type(self):
        p = Pipe({"x": 1}, {"x": "INT"})
        p2 = p.set("x", "hello", "STRING")
        assert p2.get("x") == "hello"
        assert p2.type_of("x") == "STRING"

    def test_remove(self):
        p = Pipe({"a": 1, "b": 2}, {"a": "INT", "b": "INT"})
        p2 = p.remove("a")
        assert p.has("a")  # original untouched
        assert not p2.has("a")
        assert p2.has("b")
        assert list(p2.manifest) == ["b"]

    def test_merge_left_wins(self):
        a = Pipe({"x": 1, "y": 2}, {"x": "A", "y": "A"})
        b = Pipe({"y": 99, "z": 3}, {"y": "B", "z": "B"})
        merged = a.merge(b, right_wins=False)
        assert merged.get("x") == 1
        assert merged.get("y") == 2  # left wins
        assert merged.get("z") == 3
        assert merged.type_of("y") == "A"

    def test_merge_right_wins(self):
        a = Pipe({"x": 1, "y": 2}, {"x": "A", "y": "A"})
        b = Pipe({"y": 99, "z": 3}, {"y": "B", "z": "B"})
        merged = a.merge(b, right_wins=True)
        assert merged.get("y") == 99
        assert merged.type_of("y") == "B"

    def test_repr_includes_manifest(self):
        p = Pipe({"a": 1}, {"a": "INT"})
        assert "a" in repr(p)
        assert "INT" in repr(p)


class TestCheckType:
    def test_wildcards_always_pass(self):
        assert _check_type("*", "MODEL", "k") is None
        assert _check_type("MODEL", "*", "k") is None
        assert _check_type("*", "*", "k") is None

    def test_matching_types_pass(self):
        assert _check_type("MODEL", "MODEL", "model") is None

    def test_mismatch_returns_message_with_key(self):
        err = _check_type("MODEL", "CLIP", "model")
        assert err is not None
        assert "'model'" in err
        assert "MODEL" in err
        assert "CLIP" in err


class TestPipeOutExecute:
    def test_emits_values_in_manifest_order_when_no_snapshot(self):
        pipe = Pipe(
            {"model": "M", "clip": "C", "vae": "V"},
            {"model": "MODEL", "clip": "CLIP", "vae": "VAE"},
        )
        out = PipeOut.execute(pipe)
        assert out.args[0] == "M"
        assert out.args[1] == "C"
        assert out.args[2] == "V"
        assert out.args[3] is None
        assert len(out.args) == MAX_PIPE_OUT_SLOTS

    def test_empty_pipe_returns_all_none(self):
        out = PipeOut.execute(Pipe())
        assert len(out.args) == MAX_PIPE_OUT_SLOTS
        assert all(v is None for v in out.args)

    def test_snapshot_drives_output_order(self):
        # Pipe insertion order is A, B, C but snapshot fixes order as C, A, B.
        pipe = Pipe(
            {"a": "alpha", "b": "beta", "c": "gamma"},
            {"a": "T", "b": "T", "c": "T"},
        )
        snapshot = json.dumps([["c", "T"], ["a", "T"], ["b", "T"]])
        out = PipeOut.execute(pipe, snapshot)
        assert out.args[0] == "gamma"
        assert out.args[1] == "alpha"
        assert out.args[2] == "beta"

    def test_snapshot_missing_key_raises(self):
        pipe = Pipe({"a": 1}, {"a": "INT"})
        snapshot = json.dumps([["a", "INT"], ["missing", "STRING"]])
        with pytest.raises(ValueError, match="missing key 'missing'"):
            PipeOut.execute(pipe, snapshot)

    def test_snapshot_type_mismatch_raises(self):
        pipe = Pipe({"a": "hi"}, {"a": "STRING"})
        snapshot = json.dumps([["a", "INT"]])
        with pytest.raises(ValueError, match="expected type 'INT'"):
            PipeOut.execute(pipe, snapshot)

    def test_overflow_raises_with_clear_message(self):
        entries = {f"k{i}": i for i in range(MAX_PIPE_OUT_SLOTS + 1)}
        pipe = Pipe(entries, {k: "INT" for k in entries})
        with pytest.raises(ValueError, match=f"at most {MAX_PIPE_OUT_SLOTS}"):
            PipeOut.execute(pipe)

    def test_malformed_snapshot_falls_back_to_pipe_manifest(self):
        pipe = Pipe({"a": 1}, {"a": "INT"})
        out = PipeOut.execute(pipe, "not json at all")
        assert out.args[0] == 1


class TestPipeSetExecute:
    def _hidden(self, unique_id=None, prompt=None):
        hidden = MagicMock()
        hidden.prompt = prompt
        hidden.unique_id = unique_id
        return hidden

    def test_set_adopts_upstream_type_from_prompt(self):
        pipe = Pipe({"a": 1}, {"a": "INT"})
        with patch.object(PipeSet, "hidden", self._hidden()), patch(
            "comfy_extras.nodes_pipe._resolve_input_types_from_prompt",
            return_value={"value": "STRING"},
        ):
            out = PipeSet.execute(pipe, "b", "hi")
        assert out.args[0].get("b") == "hi"
        assert out.args[0].type_of("b") == "STRING"

    def test_set_without_prompt_falls_back_to_wildcard(self):
        pipe = Pipe()
        with patch.object(PipeSet, "hidden", self._hidden()):
            out = PipeSet.execute(pipe, "k", 42)
        assert out.args[0].get("k") == 42
        assert out.args[0].type_of("k") == "*"


class TestResolveInputTypesFromPrompt:
    def test_returns_empty_when_prompt_is_none(self):
        from comfy_extras.nodes_pipe import _resolve_input_types_from_prompt

        assert _resolve_input_types_from_prompt(None, "1") == {}

    def test_returns_empty_when_unique_id_not_in_prompt(self):
        from comfy_extras.nodes_pipe import _resolve_input_types_from_prompt

        assert _resolve_input_types_from_prompt({}, "missing") == {}

    def test_resolves_through_node_class_mappings(self):
        from comfy_extras import nodes_pipe

        upstream = MagicMock()
        upstream.RETURN_TYPES = ("MODEL", "CLIP")
        prompt = {
            "10": {"inputs": {"pipe": ["5", 1]}},
            "5": {"class_type": "MultiOutNode", "inputs": {}},
        }
        with patch.object(
            nodes_pipe,
            "_node_class_mappings",
            return_value={"MultiOutNode": upstream},
        ):
            result = nodes_pipe._resolve_input_types_from_prompt(prompt, "10")
        assert result == {"pipe": "CLIP"}

    def test_missing_nodes_module_degrades_to_empty(self):
        from comfy_extras import nodes_pipe

        prompt = {
            "10": {"inputs": {"pipe": ["5", 0]}},
            "5": {"class_type": "Foo", "inputs": {}},
        }
        with patch.object(nodes_pipe, "_node_class_mappings", return_value=None):
            assert nodes_pipe._resolve_input_types_from_prompt(prompt, "10") == {}

    def test_out_of_range_slot_ignored(self):
        from comfy_extras import nodes_pipe

        upstream = MagicMock()
        upstream.RETURN_TYPES = ("MODEL",)
        prompt = {
            "1": {"inputs": {"pipe": ["2", 99]}},
            "2": {"class_type": "OneOut", "inputs": {}},
        }
        with patch.object(
            nodes_pipe, "_node_class_mappings", return_value={"OneOut": upstream}
        ):
            assert nodes_pipe._resolve_input_types_from_prompt(prompt, "1") == {}

    def test_ignores_widget_inputs(self):
        from comfy_extras.nodes_pipe import _resolve_input_types_from_prompt

        prompt = {"1": {"inputs": {"text": "hello", "count": 3}}}
        assert _resolve_input_types_from_prompt(prompt, "1") == {}


class TestPipeRemoveExecute:
    def test_remove_key(self):
        pipe = Pipe({"a": 1, "b": 2}, {"a": "INT", "b": "INT"})
        out = PipeRemove.execute(pipe, "a")
        assert not out.args[0].has("a")
        assert out.args[0].has("b")

    def test_remove_missing_key_raises(self):
        pipe = Pipe({"a": 1}, {"a": "INT"})
        with pytest.raises(ValueError, match="not present"):
            PipeRemove.execute(pipe, "zzz")


class TestPipeGetExecute:
    def test_get_matching_type(self):
        pipe = Pipe({"model": "M"}, {"model": "MODEL"})
        out = PipeGet.execute(pipe, "model", "MODEL")
        assert out.args[0] == "M"

    def test_get_wildcard_expected(self):
        pipe = Pipe({"model": "M"}, {"model": "MODEL"})
        out = PipeGet.execute(pipe, "model", "*")
        assert out.args[0] == "M"

    def test_get_missing_key_raises(self):
        with pytest.raises(ValueError, match="not present"):
            PipeGet.execute(Pipe(), "x", "*")

    def test_get_type_mismatch_raises(self):
        pipe = Pipe({"x": 1}, {"x": "INT"})
        with pytest.raises(ValueError, match="expected type 'STRING'"):
            PipeGet.execute(pipe, "x", "STRING")


class TestPipeMergeExecute:
    def test_merge_left_wins(self):
        a = Pipe({"x": 1, "y": 2}, {"x": "A", "y": "A"})
        b = Pipe({"y": 99}, {"y": "B"})
        out = PipeMerge.execute(a, b, "left_wins")
        assert out.args[0].get("y") == 2

    def test_merge_right_wins(self):
        a = Pipe({"y": 2}, {"y": "A"})
        b = Pipe({"y": 99}, {"y": "B"})
        out = PipeMerge.execute(a, b, "right_wins")
        assert out.args[0].get("y") == 99

    def test_merge_error_on_collision(self):
        a = Pipe({"x": 1, "y": 2}, {"x": "A", "y": "A"})
        b = Pipe({"y": 99, "z": 3}, {"y": "B", "z": "B"})
        with pytest.raises(ValueError, match="colliding keys"):
            PipeMerge.execute(a, b, "error")

    def test_merge_error_with_no_collision_succeeds(self):
        a = Pipe({"x": 1}, {"x": "A"})
        b = Pipe({"z": 3}, {"z": "B"})
        out = PipeMerge.execute(a, b, "error")
        assert out.args[0].get("x") == 1
        assert out.args[0].get("z") == 3


class TestPipeSourceValidation:
    def test_empty_key_rejected(self):
        assert PipeSource.validate_inputs(**{"": 1, "valid": 2}) != True  # noqa: E712

    def test_all_non_empty_keys_accepted(self):
        assert PipeSource.validate_inputs(a=1, b=2, c=3) is True

    def test_no_inputs_accepted(self):
        assert PipeSource.validate_inputs() is True
