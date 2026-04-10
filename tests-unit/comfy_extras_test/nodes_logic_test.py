"""Unit tests for DynamicSwitchNode in comfy_extras.nodes_logic."""
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock all heavy native dependencies before any ComfyUI imports
# ---------------------------------------------------------------------------
for _mod in [
    "torch",
    "numpy",
    "av",
    "av.container",
    "av.subtitles",
    "av.subtitles.stream",
    "comfy",
    "comfy.sd",
    "comfy.cli_args",
    "folder_paths",
    "server",
    "nodes",
]:
    sys.modules[_mod] = MagicMock()

_pil = MagicMock()
_pil.PngImagePlugin = MagicMock()
_pil.Image = MagicMock()
sys.modules["PIL"] = _pil
sys.modules["PIL.Image"] = _pil.Image
sys.modules["PIL.PngImagePlugin"] = _pil.PngImagePlugin

from comfy_extras.nodes_logic import DynamicSwitchNode  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exec(switch: bool, **kwargs):
    """Call execute and return the NodeOutput."""
    return DynamicSwitchNode.execute(switch, **kwargs)


def _lazy(switch: bool, **kwargs):
    """Call check_lazy_status and return the result (list or None)."""
    return DynamicSwitchNode.check_lazy_status(switch, **kwargs)


# ---------------------------------------------------------------------------
# execute() tests
# ---------------------------------------------------------------------------

class TestDynamicSwitchNodeExecute:

    def test_switch_true_returns_on_true_values(self):
        result = _exec(True, on_true_0="T0", on_false_0="F0",
                              on_true_1="T1", on_false_1="F1")
        assert result.result[0] == "T0"
        assert result.result[1] == "T1"

    def test_switch_false_returns_on_false_values(self):
        result = _exec(False, on_true_0="T0", on_false_0="F0",
                               on_true_1="T1", on_false_1="F1")
        assert result.result[0] == "F0"
        assert result.result[1] == "F1"

    def test_unconnected_slot_returns_none(self):
        # only slot 0 is connected
        result = _exec(True, on_true_0="T0", on_false_0="F0")
        assert result.result[0] == "T0"
        assert result.result[1] is None  # slot 1 not connected

    def test_all_slots_true(self):
        kwargs = {}
        for i in range(DynamicSwitchNode.MAX_SLOTS):
            kwargs[f"on_true_{i}"] = f"true_{i}"
            kwargs[f"on_false_{i}"] = f"false_{i}"
        result = _exec(True, **kwargs)
        for i in range(DynamicSwitchNode.MAX_SLOTS):
            assert result.result[i] == f"true_{i}", f"slot {i} mismatch"

    def test_all_slots_false(self):
        kwargs = {}
        for i in range(DynamicSwitchNode.MAX_SLOTS):
            kwargs[f"on_true_{i}"] = f"true_{i}"
            kwargs[f"on_false_{i}"] = f"false_{i}"
        result = _exec(False, **kwargs)
        for i in range(DynamicSwitchNode.MAX_SLOTS):
            assert result.result[i] == f"false_{i}", f"slot {i} mismatch"

    def test_output_count_equals_max_slots(self):
        result = _exec(True)
        assert len(result.result) == DynamicSwitchNode.MAX_SLOTS

    def test_mixed_types_per_slot(self):
        """Different types for different slots should pass through unchanged."""
        result = _exec(True,
                       on_true_0=42,
                       on_false_0=0,
                       on_true_1=3.14,
                       on_false_1=0.0,
                       on_true_2=["a", "b"],
                       on_false_2=[])
        assert result.result[0] == 42
        assert result.result[1] == 3.14
        assert result.result[2] == ["a", "b"]

    def test_no_inputs_all_none(self):
        result = _exec(True)
        assert all(v is None for v in result.result)


# ---------------------------------------------------------------------------
# check_lazy_status() tests
# ---------------------------------------------------------------------------

class TestDynamicSwitchNodeLazy:

    def test_requests_on_true_when_switch_true_and_unevaluated(self):
        # on_true_0 present but None → needs evaluation
        needed = _lazy(True, on_true_0=None, on_false_0=None)
        assert "on_true_0" in needed
        # on_false_0 should NOT be requested when switch is True
        assert "on_false_0" not in needed

    def test_requests_on_false_when_switch_false_and_unevaluated(self):
        needed = _lazy(False, on_true_0=None, on_false_0=None)
        assert "on_false_0" in needed
        assert "on_true_0" not in needed

    def test_does_not_request_already_evaluated(self):
        # on_true_0 already has a real value → should not be requested again
        needed = _lazy(True, on_true_0="already_evaluated", on_false_0=None)
        assert needed is None or "on_true_0" not in (needed or [])

    def test_returns_none_when_nothing_needed(self):
        # All on_true inputs already evaluated
        result = _lazy(True, on_true_0="val0", on_true_1="val1")
        assert result is None

    def test_requests_multiple_unevaluated_true_slots(self):
        needed = _lazy(True, on_true_0=None, on_true_1=None, on_false_0=None)
        assert "on_true_0" in needed
        assert "on_true_1" in needed
        assert "on_false_0" not in needed

    def test_requests_multiple_unevaluated_false_slots(self):
        needed = _lazy(False, on_false_0=None, on_false_1=None, on_true_0=None)
        assert "on_false_0" in needed
        assert "on_false_1" in needed
        assert "on_true_0" not in needed

    def test_unconnected_inputs_not_requested(self):
        # on_true_2 is not in kwargs at all (not connected) → should not appear
        needed = _lazy(True, on_true_0=None)
        assert "on_true_0" in needed
        assert "on_true_2" not in (needed or [])


# ---------------------------------------------------------------------------
# Schema definition tests
# ---------------------------------------------------------------------------

class TestDynamicSwitchNodeSchema:

    def setup_method(self):
        self.schema = DynamicSwitchNode.define_schema()

    def test_schema_has_correct_node_id(self):
        assert self.schema.node_id == "ComfyDynamicSwitchNode"

    def test_schema_has_switch_input(self):
        input_ids = [inp.id for inp in self.schema.inputs]
        assert "switch" in input_ids

    def test_schema_has_correct_number_of_inputs(self):
        # 1 switch + 2 per slot (on_true and on_false)
        expected = 1 + DynamicSwitchNode.MAX_SLOTS * 2
        assert len(self.schema.inputs) == expected

    def test_schema_has_correct_number_of_outputs(self):
        assert len(self.schema.outputs) == DynamicSwitchNode.MAX_SLOTS

    def test_all_slot_inputs_are_optional(self):
        for inp in self.schema.inputs:
            if inp.id == "switch":
                continue
            assert inp.optional, f"Input '{inp.id}' should be optional"

    def test_all_slot_inputs_are_lazy(self):
        for inp in self.schema.inputs:
            if inp.id == "switch":
                continue
            assert inp.lazy, f"Input '{inp.id}' should be lazy"

    def test_each_slot_has_unique_template_id(self):
        template_ids = set()
        for inp in self.schema.inputs:
            if inp.id == "switch":
                continue
            tid = inp.template.template_id
            # Both on_true and on_false for the same slot share one template id
            template_ids.add(tid)
        # Should have exactly MAX_SLOTS unique template ids
        assert len(template_ids) == DynamicSwitchNode.MAX_SLOTS

    def test_true_and_false_inputs_share_template_per_slot(self):
        """on_true_i and on_false_i must use the same template so types match."""
        for i in range(DynamicSwitchNode.MAX_SLOTS):
            true_inp = next(x for x in self.schema.inputs if x.id == f"on_true_{i}")
            false_inp = next(x for x in self.schema.inputs if x.id == f"on_false_{i}")
            assert true_inp.template.template_id == false_inp.template.template_id

    def test_outputs_use_same_template_as_corresponding_inputs(self):
        for i in range(DynamicSwitchNode.MAX_SLOTS):
            true_inp = next(x for x in self.schema.inputs if x.id == f"on_true_{i}")
            output = self.schema.outputs[i]
            assert true_inp.template.template_id == output.template.template_id

    def test_different_slots_have_different_templates(self):
        """Ensure cross-slot type isolation: slot 0 template ≠ slot 1 template."""
        inp0 = next(x for x in self.schema.inputs if x.id == "on_true_0")
        inp1 = next(x for x in self.schema.inputs if x.id == "on_true_1")
        assert inp0.template.template_id != inp1.template.template_id
