from __future__ import annotations
from typing import TypedDict
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io

# sentinel for missing inputs
MISSING = object()


class SwitchNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="ComfySwitchNode",
            display_name="Switch",
            category="logic",
            is_experimental=True,
            inputs=[
                io.Boolean.Input("switch"),
                io.MatchType.Input("on_false", template=template, lazy=True),
                io.MatchType.Input("on_true", template=template, lazy=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, switch, on_false=None, on_true=None):
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]

    @classmethod
    def execute(cls, switch, on_true, on_false) -> io.NodeOutput:
        return io.NodeOutput(on_true if switch else on_false)


class SoftSwitchNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="ComfySoftSwitchNode",
            display_name="Soft Switch",
            category="logic",
            is_experimental=True,
            inputs=[
                io.Boolean.Input("switch"),
                io.MatchType.Input("on_false", template=template, lazy=True, optional=True),
                io.MatchType.Input("on_true", template=template, lazy=True, optional=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
            ],
        )

    @classmethod
    def check_lazy_status(cls, switch, on_false=MISSING, on_true=MISSING):
        # We use MISSING instead of None, as None is passed for connected-but-unevaluated inputs.
        # This trick allows us to ignore the value of the switch and still be able to run execute().

        # One of the inputs may be missing, in which case we need to evaluate the other input
        if on_false is MISSING:
            return ["on_true"]
        if on_true is MISSING:
            return ["on_false"]
        # Normal lazy switch operation
        if switch and on_true is None:
            return ["on_true"]
        if not switch and on_false is None:
            return ["on_false"]

    @classmethod
    def validate_inputs(cls, switch, on_false=MISSING, on_true=MISSING):
        # This check happens before check_lazy_status(), so we can eliminate the case where
        # both inputs are missing.
        if on_false is MISSING and on_true is MISSING:
            return "At least one of on_false or on_true must be connected to Switch node"
        return True

    @classmethod
    def execute(cls, switch, on_true=MISSING, on_false=MISSING) -> io.NodeOutput:
        if on_true is MISSING:
            return io.NodeOutput(on_false)
        if on_false is MISSING:
            return io.NodeOutput(on_true)
        return io.NodeOutput(on_true if switch else on_false)


class DynamicSwitchNode(io.ComfyNode):
    """
    A switch node with a growing number of input pairs.

    For each slot i, connects on_false_i and on_true_i (both optional).
    When ``switch`` is True all on_true_i values are forwarded to output_i;
    when False all on_false_i values are forwarded.

    Each slot pair has its own independent MatchType template so different
    slots may carry different data types.  Inputs grow as you connect them:
    the first empty pair at the end acts as the "next available" slot.
    """

    MAX_SLOTS = 10

    @classmethod
    def define_schema(cls):
        inputs = [io.Boolean.Input("switch")]
        outputs = []

        for i in range(cls.MAX_SLOTS):
            template = io.MatchType.Template(f"slot_{i}")
            inputs.append(
                io.MatchType.Input(f"on_false_{i}", template=template, optional=True, lazy=True)
            )
            inputs.append(
                io.MatchType.Input(f"on_true_{i}", template=template, optional=True, lazy=True)
            )
            outputs.append(
                io.MatchType.Output(template=template, display_name=f"output_{i}")
            )

        return io.Schema(
            node_id="ComfyDynamicSwitchNode",
            display_name="Dynamic Switch",
            category="logic",
            is_experimental=True,
            inputs=inputs,
            outputs=outputs,
        )

    @classmethod
    def check_lazy_status(cls, switch, **kwargs):
        """Request only the inputs on the active side that haven't been evaluated yet."""
        needed = []
        for i in range(cls.MAX_SLOTS):
            key = f"on_true_{i}" if switch else f"on_false_{i}"
            # key present with None → connected but not yet evaluated (lazy)
            if key in kwargs and kwargs[key] is None:
                needed.append(key)
        return needed if needed else None

    @classmethod
    def execute(cls, switch, **kwargs) -> io.NodeOutput:
        outputs = []
        for i in range(cls.MAX_SLOTS):
            key = f"on_true_{i}" if switch else f"on_false_{i}"
            outputs.append(kwargs.get(key))
        return io.NodeOutput(*outputs)


class CustomComboNode(io.ComfyNode):
    """
    Frontend node that allows user to write their own options for a combo.
    This is here to make sure the node has a backend-representation to avoid some annoyances.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CustomCombo",
            display_name="Custom Combo",
            category="utils",
            is_experimental=True,
            inputs=[io.Combo.Input("choice", options=[])],
            outputs=[
                io.String.Output(display_name="STRING"),
                io.Int.Output(display_name="INDEX"),
            ],
            accept_all_inputs=True,
        )

    @classmethod
    def validate_inputs(cls, choice: io.Combo.Type, index: int = 0, **kwargs) -> bool:
        # NOTE: DO NOT DO THIS unless you want to skip validation entirely on the node's inputs.
        # I am doing that here because the widgets (besides the combo dropdown) on this node are fully frontend defined.
        # I need to skip checking that the chosen combo option is in the options list, since those are defined by the user.
        return True

    @classmethod
    def execute(cls, choice: io.Combo.Type, index: int = 0, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(choice, index)


class DCTestNode(io.ComfyNode):
    class DCValues(TypedDict):
        combo: str
        string: str
        integer: int
        image: io.Image.Type
        subcombo: dict[str]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DCTestNode",
            display_name="DCTest",
            category="logic",
            is_output_node=True,
            inputs=[io.DynamicCombo.Input("combo", options=[
                io.DynamicCombo.Option("option1", [io.String.Input("string")]),
                io.DynamicCombo.Option("option2", [io.Int.Input("integer")]),
                io.DynamicCombo.Option("option3", [io.Image.Input("image")]),
                io.DynamicCombo.Option("option4", [
                    io.DynamicCombo.Input("subcombo", options=[
                        io.DynamicCombo.Option("opt1", [io.Float.Input("float_x"), io.Float.Input("float_y")]),
                        io.DynamicCombo.Option("opt2", [io.Mask.Input("mask1", optional=True)]),
                    ])
                ])]
            )],
            outputs=[io.AnyType.Output()],
        )

    @classmethod
    def execute(cls, combo: DCValues) -> io.NodeOutput:
        combo_val = combo["combo"]
        if combo_val == "option1":
            return io.NodeOutput(combo["string"])
        elif combo_val == "option2":
            return io.NodeOutput(combo["integer"])
        elif combo_val == "option3":
            return io.NodeOutput(combo["image"])
        elif combo_val == "option4":
            return io.NodeOutput(f"{combo['subcombo']}")
        else:
            raise ValueError(f"Invalid combo: {combo_val}")


class AutogrowNamesTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _io.Autogrow.TemplateNames(input=io.Float.Input("float"), names=["a", "b", "c"])
        return io.Schema(
            node_id="AutogrowNamesTestNode",
            display_name="AutogrowNamesTest",
            category="logic",
            inputs=[
                _io.Autogrow.Input("autogrow", template=template)
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, autogrow: _io.Autogrow.Type) -> io.NodeOutput:
        vals = list(autogrow.values())
        combined = ",".join([str(x) for x in vals])
        return io.NodeOutput(combined)

class AutogrowPrefixTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = _io.Autogrow.TemplatePrefix(input=io.Float.Input("float"), prefix="float", min=1, max=10)
        return io.Schema(
            node_id="AutogrowPrefixTestNode",
            display_name="AutogrowPrefixTest",
            category="logic",
            inputs=[
                _io.Autogrow.Input("autogrow", template=template)
            ],
            outputs=[io.String.Output()],
        )

    @classmethod
    def execute(cls, autogrow: _io.Autogrow.Type) -> io.NodeOutput:
        vals = list(autogrow.values())
        combined = ",".join([str(x) for x in vals])
        return io.NodeOutput(combined)

class ComboOutputTestNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComboOptionTestNode",
            display_name="ComboOptionTest",
            category="logic",
            inputs=[io.Combo.Input("combo", options=["option1", "option2", "option3"]),
                    io.Combo.Input("combo2", options=["option4", "option5", "option6"])],
            outputs=[io.Combo.Output(), io.Combo.Output()],
        )

    @classmethod
    def execute(cls, combo: io.Combo.Type, combo2: io.Combo.Type) -> io.NodeOutput:
        return io.NodeOutput(combo, combo2)

class ConvertStringToComboNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConvertStringToComboNode",
            search_aliases=["string to dropdown", "text to combo"],
            display_name="Convert String to Combo",
            category="logic",
            inputs=[io.String.Input("string")],
            outputs=[io.Combo.Output()],
        )

    @classmethod
    def execute(cls, string: str) -> io.NodeOutput:
        return io.NodeOutput(string)

class InvertBooleanNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="InvertBooleanNode",
            search_aliases=["not", "toggle", "negate", "flip boolean"],
            display_name="Invert Boolean",
            category="logic",
            inputs=[io.Boolean.Input("boolean")],
            outputs=[io.Boolean.Output()],
        )

    @classmethod
    def execute(cls, boolean: bool) -> io.NodeOutput:
        return io.NodeOutput(not boolean)

class LogicExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SwitchNode,
            DynamicSwitchNode,
            CustomComboNode,
            # SoftSwitchNode,
            # ConvertStringToComboNode,
            # DCTestNode,
            # AutogrowNamesTestNode,
            # AutogrowPrefixTestNode,
            # ComboOutputTestNode,
            # InvertBooleanNode,
        ]

async def comfy_entrypoint() -> LogicExtension:
    return LogicExtension()
