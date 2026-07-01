import { app } from "../../scripts/app.js";

const PIPE_IN = "PipeIn";
const PIPE_OUT = "PipeOut";
// Must match NUM_SLOTS in nodes_pipe/__init__.py
const NUM_SLOTS = 5;
const WILDCARD = "*";

function slotName(i) {
	return `value_${i}`;
}

function wildcardTypes() {
	return Array.from({ length: NUM_SLOTS }, () => ({ type: WILDCARD, label: null }));
}

// Read the concrete type of each connected PipeIn input from the origin
// node's output slot.
function readTypesFromPipeIn(node) {
	const types = wildcardTypes();
	if (!node.graph) return types;
	for (let i = 0; i < NUM_SLOTS; i++) {
		const input = node.inputs?.find((inp) => inp.name === slotName(i));
		const linkId = input?.link;
		if (linkId == null) continue;
		const link = node.graph.links?.[linkId];
		if (!link) continue;
		const originNode = node.graph.getNodeById(link.origin_id);
		const originSlot = originNode?.outputs?.[link.origin_slot];
		if (originSlot) {
			types[i] = {
				type: originSlot.type ?? WILDCARD,
				label: originSlot.label ?? originSlot.name ?? null,
			};
		}
	}
	return types;
}

function applyTypesToPipeInInputs(node, types) {
	for (let i = 0; i < NUM_SLOTS; i++) {
		const input = node.inputs?.find((inp) => inp.name === slotName(i));
		if (!input) continue;
		input.type = types[i].type;
		input.label = types[i].type === WILDCARD ? null : `${slotName(i)} (${types[i].type})`;
	}
}

function applyTypesToPipeOutOutputs(node, types) {
	for (let i = 0; i < NUM_SLOTS; i++) {
		const output = node.outputs?.[i];
		if (!output) continue;
		output.type = types[i].type;
		output.label = types[i].type === WILDCARD ? slotName(i) : `${slotName(i)} (${types[i].type})`;
	}
	node.properties.pipeTypes = types;
	node.setDirtyCanvas?.(true, true);
}

// Recompute a PipeIn's slot types and push them to every PipeOut connected
// to its pipe output.
function propagateFromPipeIn(node) {
	const types = readTypesFromPipeIn(node);
	applyTypesToPipeInInputs(node, types);
	node.properties.pipeTypes = types;
	node.setDirtyCanvas?.(true, true);
	const pipeOutput = node.outputs?.[0];
	if (!pipeOutput?.links || !node.graph) return;
	for (const linkId of pipeOutput.links) {
		const link = node.graph.links?.[linkId];
		if (!link) continue;
		const target = node.graph.getNodeById(link.target_id);
		if (target?.type === PIPE_OUT) {
			applyTypesToPipeOutOutputs(target, types);
		}
	}
}

// Resolve a PipeOut's output types from the PipeIn feeding its pipe input.
function pullTypesFromUpstream(node) {
	let types = null;
	if (node.graph) {
		const input = node.inputs?.find((inp) => inp.name === "pipe");
		const linkId = input?.link;
		if (linkId != null) {
			const link = node.graph.links?.[linkId];
			const origin = link ? node.graph.getNodeById(link.origin_id) : null;
			if (origin?.type === PIPE_IN) {
				types = origin.properties?.pipeTypes ?? readTypesFromPipeIn(origin);
			}
		}
	}
	applyTypesToPipeOutOutputs(node, types ?? wildcardTypes());
}

app.registerExtension({
	name: "Comfy.PipeTypeSync",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name === PIPE_IN) {
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (...args) {
				onConnectionsChange?.apply(this, args);
				// Defer so litegraph's link tables reflect the change regardless
				// of whether the callback fires before or after the mutation.
				setTimeout(() => propagateFromPipeIn(this), 0);
			};
			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function (...args) {
				onConfigure?.apply(this, args);
				if (this.properties?.pipeTypes) {
					applyTypesToPipeInInputs(this, this.properties.pipeTypes);
				}
			};
		} else if (nodeData.name === PIPE_OUT) {
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (...args) {
				onConnectionsChange?.apply(this, args);
				setTimeout(() => pullTypesFromUpstream(this), 0);
			};
			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function (...args) {
				onConfigure?.apply(this, args);
				if (this.properties?.pipeTypes) {
					// Reapply saved types without clobbering them via the
					// upstream lookup (nodes may configure in any order).
					for (let i = 0; i < NUM_SLOTS; i++) {
						const output = this.outputs?.[i];
						const t = this.properties.pipeTypes[i];
						if (!output || !t) continue;
						output.type = t.type;
						output.label = t.type === WILDCARD ? slotName(i) : `${slotName(i)} (${t.type})`;
					}
				}
			};
		}
	},
});
