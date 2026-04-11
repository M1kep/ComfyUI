// ComfyUI Pipe node pack - editor-time extension.
//
// Walks upstream pipe-producing nodes to compute a manifest of {key: type},
// reshapes Pipe Out / Pipe Get output slots to match, keeps the Pipe source
// input slots growing, and persists each Pipe Out's expected manifest to
// its `expected_manifest` widget so downstream wires bind on workflow reload
// before upstream resolves.
//
// Conventions confirmed against the real frontend source:
//   - onConnectionsChange(type, index, connected, link, ioSlot)
//     type is LiteGraph.INPUT (1) or LiteGraph.OUTPUT (2).
//   - app.configuringGraph is true during graph load.
//   - onAfterGraphConfigured is the per-node post-load hook.
//   - setDirtyCanvas(true, true) triggers repaint.
//   - Widgets with serialize_widgets=true (ComfyUI default) are auto-persisted
//     through workflow JSON via widgets_values.

import { app } from "../../scripts/app.js";

// LiteGraph is exposed on window by the frontend bundle (verified in the
// installed comfyui-frontend-package: GraphView bundle sets window.LiteGraph).
const LiteGraph = /** @type {any} */ (globalThis).LiteGraph;

const PIPE_TYPE = "PIPE";
const MAX_PIPE_OUT_SLOTS = 32;
const WILDCARD = "*";
const DEFAULT_SLOT_NAME_RE = /^slot_\d+$/;

const PIPE_NODE_TYPES = new Set([
    "Pipe",
    "PipeOut",
    "PipeSet",
    "PipeRemove",
    "PipeGet",
    "PipeMerge",
]);

const PIPE_PRODUCERS = new Set(["Pipe", "PipeSet", "PipeRemove", "PipeMerge"]);

const defaultSlotName = (i) => `slot_${i}`;

function chain(original, ...extras) {
    return function chained(...args) {
        const r = typeof original === "function" ? original.apply(this, args) : undefined;
        for (const fn of extras) fn.apply(this, args);
        return r;
    };
}

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

// ---- Manifest computation ----------------------------------------------------

// Compute the manifest emitted by a pipe-producing node as an array of
// [key, type] pairs. `memo` is a per-propagation cache keyed by node id.
function computeManifest(node, memo) {
    if (!node) return [];
    if (memo.has(node.id)) return memo.get(node.id);
    memo.set(node.id, []);

    let result;
    switch (node.type) {
        case "Pipe":
            result = [];
            for (const slot of node.inputs || []) {
                if (slot.link != null && slot.name) {
                    result.push([slot.name, slot.type || WILDCARD]);
                }
            }
            break;
        case "PipeSet": {
            const upstream = manifestFromInput(node, "pipe", memo);
            const key = findWidget(node, "key")?.value;
            if (!key) { result = upstream; break; }
            const valueSlot = findSlot(node.inputs, "value");
            const type = valueSlot?.link != null ? (valueSlot.type || WILDCARD) : WILDCARD;
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
            // Foreign pipe producer (e.g. subgraph IO proxy, another extension).
            // Read its last-known persisted manifest if it exposes one.
            result = loadExpectedManifest(node);
            break;
    }

    memo.set(node.id, result);
    return result;
}

function manifestFromInput(node, inputName, memo) {
    const up = upstreamNode(node, inputName);
    return up ? computeManifest(up, memo) : [];
}

// ---- Manifest persistence on PipeOut ---------------------------------------

// The `expected_manifest` STRING widget on PipeOut is the round-tripped
// source of truth. ComfyUI serializes widget values through workflow JSON
// automatically, so we just keep it in sync.

function loadExpectedManifest(node) {
    const widget = findWidget(node, "expected_manifest");
    const raw = widget?.value;
    if (typeof raw !== "string" || !raw) return [];
    try {
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];
        return parsed.filter((e) => Array.isArray(e) && e.length >= 2);
    } catch (err) {
        console.warn(`[comfyui.pipe] malformed expected_manifest on node ${node?.id}:`, err);
        return [];
    }
}

function persistExpectedManifest(node, manifest) {
    const widget = findWidget(node, "expected_manifest");
    if (!widget) return false;
    const serialized = JSON.stringify(manifest);
    if (widget.value === serialized) return false;
    widget.value = serialized;
    return true;
}

function hideExpectedManifestWidget(node) {
    const widget = findWidget(node, "expected_manifest");
    if (!widget) return;
    // Litegraph honors type="hidden" by skipping the widget in layout/draw.
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
}

// ---- Slot reshape -----------------------------------------------------------

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
    if (manifest.length > MAX_PIPE_OUT_SLOTS) {
        console.warn(
            `[comfyui.pipe] Pipe Out node ${node.id}: upstream manifest has ${manifest.length} keys; trimming to ${MAX_PIPE_OUT_SLOTS}. Insert a Pipe Remove to stay within the limit.`
        );
    }
    const limit = Math.min(manifest.length, MAX_PIPE_OUT_SLOTS);
    const current = node.outputs;

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

    const changed = persistExpectedManifest(node, manifest.slice(0, limit));
    if (changed) {
        node.setSize(node.computeSize());
        node.setDirtyCanvas(true, true);
    }
}

function reshapePipeGet(node, memo) {
    const upstream = manifestFromInput(node, "pipe", memo || new Map());
    const key = findWidget(node, "key")?.value || "";
    const entry = upstream.find(([k]) => k === key);
    const out = node.outputs?.[0];
    if (!out) return;
    const name = entry ? entry[0] : "value";
    const type = entry ? entry[1] : WILDCARD;
    retypeOutputSlot(node, 0, name, type);

    const expected = findWidget(node, "expected_type");
    if (expected) expected.value = type;
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

// ---- Downstream propagation -------------------------------------------------

const isPipeConsumer = (node) => node?.type === "PipeOut" || node?.type === "PipeGet";

function propagateDownstream(node) {
    if (app.configuringGraph) return; // onAfterGraphConfigured will handle load-time reshape.
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
                if (PIPE_PRODUCERS.has(target.type)) walk(target);
            }
        }
    };
    walk(node);
}

// ---- Reconcile-on-load ------------------------------------------------------

function reconcileOnLoad(node) {
    const persisted = loadExpectedManifest(node);
    const fresh = manifestFromInput(node, "pipe", new Map());

    if (!persisted.length) {
        // First-time load with no snapshot - take the fresh shape.
        if (fresh.length) reshapePipeOut(node, fresh);
        return;
    }

    // Apply persisted so downstream wires bind, regardless of whether upstream
    // resolved yet.
    reshapePipeOut(node, persisted);

    if (fresh.length && !manifestsEqual(persisted, fresh)) {
        markDrift(node, persisted, fresh);
    } else {
        clearDrift(node);
    }
}

function manifestsEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i][0] !== b[i][0] || a[i][1] !== b[i][1]) return false;
    }
    return true;
}

// Drift indicator: prefix the title and flip bgcolor. A context-menu entry
// lets the user re-sync on demand.
function markDrift(node, persisted, fresh) {
    node.__pipe_drift = { persisted, fresh };
    node.bgcolor = "#6b3030";
    if (!node.title.startsWith("\u26A0 ")) {
        node.__pipe_original_title = node.title;
        node.title = `\u26A0 ${node.title}`;
    }
    console.warn(
        `[comfyui.pipe] Pipe Out node ${node.id}: upstream manifest drift detected.`,
        { persisted, fresh }
    );
    node.setDirtyCanvas(true, true);
}

function clearDrift(node) {
    if (!node.__pipe_drift) return;
    delete node.__pipe_drift;
    node.bgcolor = undefined;
    if (node.__pipe_original_title != null) {
        node.title = node.__pipe_original_title;
        delete node.__pipe_original_title;
    }
    node.setDirtyCanvas(true, true);
}

// ---- Pipe source slot management -------------------------------------------

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
        node.addInput(defaultSlotName(node.inputs.length), WILDCARD);
    }
    // A duplicate slot name silently collapses server-side when the prompt
    // dict is built from the workflow inputs array, dropping a value. Catch
    // it in the editor.
    deduplicateSourceKeys(node);
}

function deduplicateSourceKeys(node) {
    const seen = new Set();
    for (let i = 0; i < (node.inputs?.length || 0); i++) {
        const slot = node.inputs[i];
        if (!slot?.name) continue;
        if (!seen.has(slot.name)) {
            seen.add(slot.name);
            continue;
        }
        const unique = uniquifyKey(node, slot.name, i);
        console.warn(
            `[comfyui.pipe] Pipe node ${node.id}: duplicate key ${JSON.stringify(slot.name)} on slot ${i}; renamed to ${JSON.stringify(unique)}.`
        );
        slot.name = unique;
        slot.label = unique;
        seen.add(unique);
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

// ---- Key suggestion menu (lightweight "dropdown" UX) -----------------------

function attachKeySuggestions(node) {
    const keyWidget = findWidget(node, "key");
    if (!keyWidget) return;
    const origMouse = keyWidget.mouse;
    keyWidget.mouse = function (event, pos, owningNode) {
        if (event.type === "pointerdown" && event.button === 2) {
            showKeySuggestionMenu(owningNode, event);
            return true;
        }
        return typeof origMouse === "function" ? origMouse.call(this, event, pos, owningNode) : false;
    };
}

function showKeySuggestionMenu(node, event) {
    const inputName = node.type === "PipeMerge" ? "a" : "pipe";
    const suggestions = manifestFromInput(node, inputName, new Map());
    if (!suggestions.length) {
        new LiteGraph.ContextMenu(["(no keys on upstream pipe)"], {
            event,
            title: "Upstream keys",
        });
        return;
    }
    const entries = suggestions.map(([k, t]) => ({
        content: `${k}  ·  ${t}`,
        key: k,
    }));
    new LiteGraph.ContextMenu(entries, {
        event,
        title: "Upstream keys",
        callback: (entry) => {
            const keyWidget = findWidget(node, "key");
            if (!keyWidget || !entry) return;
            keyWidget.value = entry.key;
            keyWidget.callback?.(entry.key);
        },
    });
}

// ---- Connection change handler ----------------------------------------------

function handleConnectionsChange(node, type, slotIndex, connected, linkInfo) {
    if (type !== LiteGraph.INPUT) return;

    if (node.type === "Pipe") {
        const slot = node.inputs[slotIndex];
        if (connected && linkInfo && slot) {
            const upstream = app.graph.getNodeById(linkInfo.origin_id);
            const upOutput = upstream?.outputs?.[linkInfo.origin_slot];
            if (upOutput) {
                slot.type = upOutput.type || WILDCARD;
                if (DEFAULT_SLOT_NAME_RE.test(slot.name)) {
                    const suggestion = upOutput.name || upOutput.label || slot.type.toLowerCase();
                    slot.name = uniquifyKey(node, suggestion, slotIndex);
                    slot.label = slot.name;
                }
            }
        } else if (!connected && slot) {
            slot.type = WILDCARD;
            slot.name = defaultSlotName(slotIndex);
            slot.label = slot.name;
        }
        ensurePipeSourceSlots(node);
        propagateDownstream(node);
        return;
    }

    if (node.type === "PipeOut") {
        reshapePipeOut(node, manifestFromInput(node, "pipe", new Map()));
        clearDrift(node);
    } else if (node.type === "PipeGet") {
        reshapePipeGet(node);
    }
    propagateDownstream(node);
}

// ---- Extension registration -------------------------------------------------

app.registerExtension({
    name: "comfyui.pipe",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!PIPE_NODE_TYPES.has(nodeData.name)) return;

        nodeType.prototype.onConnectionsChange = chain(
            nodeType.prototype.onConnectionsChange,
            function (type, slotIndex, connected, linkInfo) {
                try { handleConnectionsChange(this, type, slotIndex, connected, linkInfo); }
                catch (err) { console.error("[comfyui.pipe] onConnectionsChange", err); }
            }
        );

        if (nodeData.name === "PipeOut") {
            nodeType.prototype.onAfterGraphConfigured = chain(
                nodeType.prototype.onAfterGraphConfigured,
                function () {
                    try { reconcileOnLoad(this); }
                    catch (err) { console.error("[comfyui.pipe] onAfterGraphConfigured", err); }
                }
            );

            nodeType.prototype.getExtraMenuOptions = chain(
                nodeType.prototype.getExtraMenuOptions,
                function (_canvas, options) {
                    if (!this.__pipe_drift) return;
                    options.unshift({
                        content: "Pipe: refresh from upstream",
                        callback: () => {
                            reshapePipeOut(this, manifestFromInput(this, "pipe", new Map()));
                            clearDrift(this);
                        },
                    });
                }
            );
        }
    },

    nodeCreated(node) {
        if (!PIPE_NODE_TYPES.has(node.type)) return;
        if (node.type === "Pipe") {
            setTimeout(() => ensurePipeSourceSlots(node), 0);
        }
        if (node.type === "PipeOut") {
            hideExpectedManifestWidget(node);
            setTimeout(() => {
                const persisted = loadExpectedManifest(node);
                if (persisted.length) reshapePipeOut(node, persisted);
            }, 0);
        }
        attachKeySuggestions(node);
        attachPropagationWidgetCallbacks(node);
    },
});

function attachPropagationWidgetCallbacks(node) {
    const widgets = ["key", "collision"];
    for (const name of widgets) {
        const widget = findWidget(node, name);
        if (!widget) continue;
        widget.callback = chain(widget.callback, () => {
            if (node.type === "PipeGet") reshapePipeGet(node);
            propagateDownstream(node);
        });
    }
}
