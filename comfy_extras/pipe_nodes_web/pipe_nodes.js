// ComfyUI Pipe node pack - editor-time extension.
//
// Walks upstream to resolve each pipe's manifest, reshapes Pipe Out / Pipe Get
// slots to match, and keeps Pipe source slots growing as the user connects.

import { app } from "../../scripts/app.js";

const PIPE_TYPE = "PIPE";
const MAX_PIPE_OUT_SLOTS = 32;
const SLOT_KIND_INPUT = 1;
const DEFAULT_SLOT_NAME_RE = /^slot_\d+$/;
const MANIFEST_PROPERTY = "pipe_manifest";

const PIPE_NODE_TYPES = new Set([
    "Pipe",
    "PipeOut",
    "PipeSet",
    "PipeRemove",
    "PipeGet",
    "PipeMerge",
]);

const defaultSlotName = (i) => `slot_${i}`;

function findWidget(node, name) {
    return node?.widgets?.find((w) => w.name === name);
}

function findSlot(slots, name) {
    if (!slots) return null;
    for (const slot of slots) {
        if (slot.name === name) return slot;
    }
    return null;
}

function upstreamNode(node, inputName) {
    const slot = findSlot(node.inputs, inputName);
    if (!slot || slot.link == null) return null;
    const link = app.graph.links[slot.link];
    if (!link) return null;
    return app.graph.getNodeById(link.origin_id) || null;
}

// Compute the manifest emitted by a pipe-producing node as an array of
// [key, type] pairs. `memo` is a per-propagation cache keyed by node id.
function computeManifest(node, memo) {
    if (!node) return [];
    if (memo.has(node.id)) return memo.get(node.id);
    // Place a sentinel to break cycles in pathological graphs.
    memo.set(node.id, []);

    let result;
    switch (node.type) {
        case "Pipe":
            result = [];
            for (const slot of node.inputs || []) {
                if (slot.link != null && slot.name) {
                    result.push([slot.name, slot.type || "*"]);
                }
            }
            break;
        case "PipeSet": {
            const upstream = manifestFromInput(node, "pipe", memo);
            const key = findWidget(node, "key")?.value;
            if (!key) { result = upstream; break; }
            const valueSlot = findSlot(node.inputs, "value");
            const type = valueSlot?.link != null ? (valueSlot.type || "*") : "*";
            result = upstream.filter(([k]) => k !== key);
            result.push([key, type]);
            break;
        }
        case "PipeRemove": {
            const upstream = manifestFromInput(node, "pipe", memo);
            const key = findWidget(node, "key")?.value;
            result = key ? upstream.filter(([k]) => k !== key) : upstream;
            break;
        }
        case "PipeMerge": {
            const a = manifestFromInput(node, "a", memo);
            const b = manifestFromInput(node, "b", memo);
            const collision = findWidget(node, "collision")?.value || "left_wins";
            const byKey = new Map();
            const [first, second] = collision === "right_wins" ? [a, b] : [b, a];
            for (const [k, t] of first) byKey.set(k, t);
            for (const [k, t] of second) byKey.set(k, t);
            result = Array.from(byKey.entries());
            break;
        }
        default:
            result = loadPersistedManifest(node);
            break;
    }

    memo.set(node.id, result);
    return result;
}

function manifestFromInput(node, inputName, memo) {
    const up = upstreamNode(node, inputName);
    return up ? computeManifest(up, memo) : [];
}

function persistManifest(node, manifest) {
    if (!node.properties) node.properties = {};
    const serialized = JSON.stringify(manifest);
    if (node.properties[MANIFEST_PROPERTY] === serialized) return false;
    node.properties[MANIFEST_PROPERTY] = serialized;
    return true;
}

function loadPersistedManifest(node) {
    const raw = node?.properties?.[MANIFEST_PROPERTY];
    if (!raw) return [];
    try {
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed.filter(Array.isArray) : [];
    } catch (err) {
        console.warn(`[comfyui.pipe] malformed persisted manifest on node ${node?.id}:`, err);
        return [];
    }
}

function disconnectOutput(node, slotIndex) {
    const out = node.outputs?.[slotIndex];
    if (!out?.links) return;
    for (const linkId of [...out.links]) {
        app.graph.removeLink(linkId);
    }
}

function retypeOutputSlot(node, slotIndex, name, type) {
    const slot = node.outputs[slotIndex];
    if (slot.name === name && slot.type === type) return;
    if (slot.type !== type && slot.links?.length) {
        disconnectOutput(node, slotIndex);
    }
    slot.name = name;
    slot.label = name;
    slot.type = type;
}

function reshapePipeOut(node, manifest) {
    if (!node.outputs) node.outputs = [];
    const current = node.outputs;
    const limit = Math.min(manifest.length, MAX_PIPE_OUT_SLOTS);

    for (let i = 0; i < limit; i++) {
        const [key, type] = manifest[i];
        if (!current[i]) {
            node.addOutput(key, type);
        } else {
            retypeOutputSlot(node, i, key, type);
        }
    }
    while (current.length > limit) {
        const lastIdx = current.length - 1;
        disconnectOutput(node, lastIdx);
        node.removeOutput(lastIdx);
    }

    const changed = persistManifest(node, manifest.slice(0, limit));
    if (changed) node.setDirtyCanvas(true, true);
}

function reshapePipeGet(node, memo) {
    const upstream = manifestFromInput(node, "pipe", memo || new Map());
    const key = findWidget(node, "key")?.value || "";
    const entry = upstream.find(([k]) => k === key);
    if (!node.outputs?.[0]) return;
    const name = entry ? entry[0] : "value";
    const type = entry ? entry[1] : "*";
    retypeOutputSlot(node, 0, name, type);

    const expected = findWidget(node, "expected_type");
    if (expected) expected.value = type;
    node.setDirtyCanvas(true, true);
}

const isPipeConsumer = (node) => node?.type === "PipeOut" || node?.type === "PipeGet";

function propagateDownstream(node) {
    const memo = new Map();
    const visited = new Set();
    const walk = (n) => {
        if (!n || visited.has(n.id)) return;
        visited.add(n.id);
        for (let i = 0; i < (n.outputs?.length || 0); i++) {
            const out = n.outputs[i];
            if (out.type !== PIPE_TYPE) continue;
            for (const linkId of out.links || []) {
                const link = app.graph.links[linkId];
                if (!link) continue;
                const target = app.graph.getNodeById(link.target_id);
                if (!target) continue;
                if (target.type === "PipeOut") {
                    reshapePipeOut(target, manifestFromInput(target, "pipe", memo));
                } else if (target.type === "PipeGet") {
                    reshapePipeGet(target, memo);
                }
                if (PIPE_NODE_TYPES.has(target.type) && !isPipeConsumer(target)) {
                    walk(target);
                }
            }
        }
    };
    walk(node);
}

// Ensure a Pipe source node has exactly one trailing empty input slot.
function ensurePipeSourceSlots(node) {
    if (!node.inputs) node.inputs = [];
    while (node.inputs.length >= 2) {
        const a = node.inputs[node.inputs.length - 2];
        const b = node.inputs[node.inputs.length - 1];
        if (a.link == null && b.link == null) {
            node.removeInput(node.inputs.length - 1);
        } else {
            break;
        }
    }
    const last = node.inputs[node.inputs.length - 1];
    if (!last || last.link != null) {
        node.addInput(defaultSlotName(node.inputs.length), "*");
    }
}

function uniquifyKey(node, proposed, ownIndex) {
    const taken = new Set();
    for (let i = 0; i < (node.inputs?.length || 0); i++) {
        if (i !== ownIndex && node.inputs[i]?.name) {
            taken.add(node.inputs[i].name);
        }
    }
    if (!taken.has(proposed)) return proposed;
    let n = 2;
    while (taken.has(`${proposed}_${n}`)) n++;
    return `${proposed}_${n}`;
}

function safeWrap(label, fn) {
    return function (...args) {
        try {
            return fn.apply(this, args);
        } catch (err) {
            console.error(`[comfyui.pipe] ${label} error`, err);
        }
    };
}

app.registerExtension({
    name: "comfyui.pipe",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!PIPE_NODE_TYPES.has(nodeData.name)) return;

        const origConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotKind, slotIndex, connected, linkInfo, ioSlot) {
            const r = origConnectionsChange?.apply(this, arguments);
            safeWrap("onConnectionsChange", handleConnectionsChange).call(this, this, slotKind, slotIndex, connected, linkInfo);
            return r;
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = origConfigure?.apply(this, arguments);
            safeWrap("onConfigure", handleConfigure).call(this, this);
            return r;
        };
    },

    nodeCreated(node) {
        if (!PIPE_NODE_TYPES.has(node.type)) return;
        if (node.type === "Pipe") {
            setTimeout(() => ensurePipeSourceSlots(node), 0);
        }
        if (node.type === "PipeOut") {
            // Restore slots from persisted manifest so downstream wires bind
            // before upstream resolves on workflow reload.
            setTimeout(() => {
                const persisted = loadPersistedManifest(node);
                if (persisted.length) reshapePipeOut(node, persisted);
            }, 0);
        }
        attachWidgetCallbacks(node);
    },
});

function attachWidgetCallbacks(node) {
    const reshapeAndPropagate = () => {
        if (node.type === "PipeGet") reshapePipeGet(node);
        propagateDownstream(node);
    };
    for (const widgetName of ["key", "collision"]) {
        const widget = findWidget(node, widgetName);
        if (!widget) continue;
        const orig = widget.callback;
        widget.callback = function (value) {
            const r = orig?.apply(this, arguments);
            reshapeAndPropagate();
            return r;
        };
    }
}

function handleConfigure(node) {
    if (node.type === "Pipe") {
        ensurePipeSourceSlots(node);
    } else if (node.type === "PipeOut") {
        const persisted = loadPersistedManifest(node);
        if (persisted.length) reshapePipeOut(node, persisted);
    } else if (node.type === "PipeGet") {
        reshapePipeGet(node);
    }
}

function handleConnectionsChange(node, slotKind, slotIndex, connected, linkInfo) {
    if (slotKind !== SLOT_KIND_INPUT) return;

    if (node.type === "Pipe") {
        const slot = node.inputs[slotIndex];
        if (connected && linkInfo && slot) {
            const upstream = app.graph.getNodeById(linkInfo.origin_id);
            const upOutput = upstream?.outputs?.[linkInfo.origin_slot];
            if (upOutput) {
                slot.type = upOutput.type || "*";
                if (DEFAULT_SLOT_NAME_RE.test(slot.name)) {
                    const suggestion = upOutput.name || upOutput.label || slot.type.toLowerCase();
                    slot.name = uniquifyKey(node, suggestion, slotIndex);
                    slot.label = slot.name;
                }
            }
        } else if (!connected && slot) {
            slot.type = "*";
            slot.name = defaultSlotName(slotIndex);
            slot.label = slot.name;
        }
        ensurePipeSourceSlots(node);
        propagateDownstream(node);
        return;
    }

    if (node.type === "PipeOut") {
        reshapePipeOut(node, manifestFromInput(node, "pipe", new Map()));
    } else if (node.type === "PipeGet") {
        reshapePipeGet(node);
    }
    propagateDownstream(node);
}
