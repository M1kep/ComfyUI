from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

# Frontend JS that propagates slot types across the PIPE wire lives here.
WEB_DIRECTORY = "./js"

# Bundled type carried on a single wire between PipeIn and PipeOut.
Pipe = io.Custom("PIPE")

# Number of value slots on each node. Must match NUM_SLOTS in js/pipe.js.
NUM_SLOTS = 5


class PipeIn(io.ComfyNode):
    """
    Bundles up to NUM_SLOTS values of any type into a single PIPE wire.

    Slot types are wildcards on the backend; the pipe.js frontend extension
    records the concrete type of each connected slot and propagates it to
    downstream PipeOut nodes on connect/disconnect.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PipeIn",
            display_name="Pipe In",
            category="utils/pipe",
            is_experimental=True,
            inputs=[
                io.AnyType.Input(f"value_{i}", optional=True)
                for i in range(NUM_SLOTS)
            ],
            outputs=[Pipe.Output(display_name="pipe")],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(
            {f"value_{i}": kwargs.get(f"value_{i}") for i in range(NUM_SLOTS)}
        )


class PipeOut(io.ComfyNode):
    """
    Unbundles a PIPE wire back into its individual values.

    Output slots mirror the inputs of the upstream PipeIn; their displayed
    types are kept in sync by the pipe.js frontend extension.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PipeOut",
            display_name="Pipe Out",
            category="utils/pipe",
            is_experimental=True,
            inputs=[Pipe.Input("pipe")],
            outputs=[
                io.AnyType.Output(id=f"value_{i}", display_name=f"value_{i}")
                for i in range(NUM_SLOTS)
            ],
        )

    @classmethod
    def execute(cls, pipe: dict) -> io.NodeOutput:
        return io.NodeOutput(*(pipe.get(f"value_{i}") for i in range(NUM_SLOTS)))


class PipeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PipeIn,
            PipeOut,
        ]


async def comfy_entrypoint() -> PipeExtension:
    return PipeExtension()
