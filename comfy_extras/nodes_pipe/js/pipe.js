// Editor-side support for the Pipe node family.
//
// The backend nodes are intentionally thin; this extension owns:
//   - dynamic input slots on `Pipe` (grow-on-connect, named by the user)
//   - manifest propagation (walk PIPE wires upstream to discover {key: type})
//   - dynamic output slots on `Pipe Out` / `Pipe Get`
//   - persisting the manifest into hidden `_manifest` / `_value_type` widgets
//     so the backend knows slot order and types at execution time, and so
//     workflow JSON round-trips without dropping wires.

import { app } from "../../scripts/app.js";

const PIPE_TYPE = "PIPE";
const ANY_TYPE = "*";

const NODE_PIPE = "PipeCreate";
const NODE_OUT = "PipeOut";
const NODE_SET = "PipeSet";
const NODE_REMOVE = "PipeRemove";
const NODE_GET = "PipeGet";
const NODE_MERGE = "PipeMerge";

const PIPE_NODES = new Set([
    NODE_PIPE, NODE_OUT, NODE_SET, NODE_REMOVE, NODE_GET, NODE_MERGE,
]);

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function hideWidget(node, name) {
    const w = findWidget(node, name);
    if (!w) return;
    w.type = "hidden";
    w.hidden = true;
    w.computeSize = () => [0, -4];
    // The frontend creates a widget-backed input slot for every declared
    // input so users can convert widgets to sockets. We don't want that for
    // these internal carriers — drop the slot so dynamic slot indexing is
    // stable and the user can't accidentally wire into them.
    const idx = node.inputs?.findIndex((i) => i.widget?.name === name);
    if (idx != null && idx >= 0) node.removeInput(idx);
}

function setWidget(node, name, value) {
    const w = findWidget(node, name);
    if (w) w.value = value;
}

function getWidget(node, name) {
    return findWidget(node, name)?.value;
}

function getGraph(node) {
    return node.graph ?? app.graph;
}

function originOf(node, slotIndex) {
    const link_id = node.inputs?.[slotIndex]?.link;
    if (link_id == null) return null;
    const link = getGraph(node)?.links?.[link_id];
    if (!link) return null;
    const origin = getGraph(node)?.getNodeById?.(link.origin_id);
    if (!origin) return null;
    return { node: origin, slot: link.origin_slot };
}

function originOfNamed(node, inputName) {
    const idx = node.inputs?.findIndex((i) => i.name === inputName);
    if (idx == null || idx < 0) return null;
    return originOf(node, idx);
}

// ---------------------------------------------------------------------------
// manifest computation
// ---------------------------------------------------------------------------

// Walk upstream along a PIPE wire and return an ordered manifest
// [[key, type], ...]. `seen` guards against cycles via reroutes.
function computeManifest(node, inputName, seen) {
    seen = seen ?? new Set();
    const origin = originOfNamed(node, inputName);
    if (!origin) return [];
    return manifestOfOutput(origin.node, origin.slot, seen);
}

// Manifest entry shape: [key, type] or [key, "PIPE", nestedManifest].
// The optional 3rd element lets a downstream Pipe Out unpack a nested pipe
// without re-walking the whole upstream chain (which would yield the OUTER
// manifest, not the inner one).
function entryFor(name, type, nested) {
    return type === PIPE_TYPE && nested ? [name, type, nested] : [name, type];
}

function nestedOf(entry) {
    return entry[1] === PIPE_TYPE ? entry[2] ?? [] : null;
}

function manifestOfOutput(node, slot, seen) {
    if (!node || seen.has(node.id)) return [];
    seen.add(node.id);

    const type = node.type ?? node.comfyClass;

    if (type === NODE_PIPE) {
        // Clone so callers can mutate (e.g. PipeSet appends entries).
        return (node.properties?.pipe_manifest ?? []).map((e) => [...e]);
    }
    if (type === NODE_SET) {
        const m = computeManifest(node, "pipe", seen);
        const key = getWidget(node, "key");
        const vtype = getWidget(node, "_value_type") || ANY_TYPE;
        if (key) {
            const nested = vtype === PIPE_TYPE
                ? computeManifest(node, "value", new Set(seen))
                : null;
            const entry = entryFor(key, vtype, nested);
            const i = m.findIndex(([k]) => k === key);
            if (i >= 0) m[i] = entry;
            else m.push(entry);
        }
        return m;
    }
    if (type === NODE_REMOVE) {
        const m = computeManifest(node, "pipe", seen);
        const key = getWidget(node, "key");
        return key ? m.filter(([k]) => k !== key) : m;
    }
    if (type === NODE_MERGE) {
        const a = computeManifest(node, "a", new Set(seen));
        const b = computeManifest(node, "b", new Set(seen));
        const collision = getWidget(node, "collision") || "right_wins";
        const out = a.map((e) => [...e]);
        for (const e of b) {
            const i = out.findIndex(([ok]) => ok === e[0]);
            if (i >= 0) {
                if (collision === "left_wins") continue;
                out[i] = [...e];
            } else {
                out.push([...e]);
            }
        }
        return out;
    }
    if (type === NODE_OUT || type === NODE_GET) {
        // Each output of Pipe Out / Pipe Get is one upstream manifest entry's
        // unpacked value. If that entry is itself a PIPE, return its nested
        // manifest so a downstream Pipe Out can reshape correctly.
        const outName = node.outputs?.[slot]?.name;
        const upstream = computeManifest(node, "pipe", seen);
        const entry = upstream.find(([k]) => k === outName);
        return entry ? (nestedOf(entry) ?? []) : [];
    }

    // Pass-through: reroutes, subgraph IO proxies, or any unknown node that
    // has exactly one PIPE-typed input — recurse through it.
    const passthrough = node.inputs?.findIndex(
        (i) => i.type === PIPE_TYPE || i.type === ANY_TYPE,
    );
    if (passthrough != null && passthrough >= 0) {
        return computeManifest(node, node.inputs[passthrough].name, seen);
    }
    return [];
}

function manifestKeys(m) {
    return m.map(([k]) => k);
}

function sameManifest(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i][0] !== b[i][0] || a[i][1] !== b[i][1]) return false;
        // Only the surface (key, type) drives slot rendering; nested manifests
        // are tracked separately on PipeCreate.properties so a no-op rebuild
        // here doesn't mask deep changes.
    }
    return true;
}

// ---------------------------------------------------------------------------
// PipeCreate: dynamic inputs
// ---------------------------------------------------------------------------

function isPipeSlot(inp) {
    return inp && !inp.widget;
}

function pipeCreateSync(node) {
    // Ensure one trailing empty ANY slot, prune disconnected non-trailing
    // slots, and rebuild the manifest from the current connected inputs.
    // Widget-backed inputs (e.g. `key` on PipeSet if user converts it) are
    // never treated as pipe slots.
    let trailing = -1;
    for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--) {
        if (isPipeSlot(node.inputs[i])) { trailing = i; break; }
    }
    for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--) {
        const inp = node.inputs[i];
        if (isPipeSlot(inp) && inp.link == null && i < trailing) {
            node.removeInput(i);
            trailing--;
        }
    }
    const manifest = [];
    for (const inp of node.inputs ?? []) {
        if (!isPipeSlot(inp) || inp.link == null) continue;
        const t = inp.type === ANY_TYPE ? ANY_TYPE : inp.type;
        let nested = null;
        if (t === PIPE_TYPE) {
            const link = getGraph(node)?.links?.[inp.link];
            const src = link && getGraph(node)?.getNodeById?.(link.origin_id);
            if (src) nested = manifestOfOutput(src, link.origin_slot, new Set());
        }
        manifest.push(entryFor(inp.name, t, nested));
    }
    const last = trailing >= 0 ? node.inputs[trailing] : null;
    if (!last || last.link != null) {
        node.addInput("+", ANY_TYPE);
    }
    node.properties.pipe_manifest = manifest;
    setWidget(node, "_manifest", JSON.stringify(manifest));
}

function pipeCreateOnConnect(node, slotIndex, link_info) {
    const inp = node.inputs[slotIndex];
    if (!isPipeSlot(inp)) return;
    const origin = getGraph(node)?.getNodeById?.(link_info.origin_id);
    const otype = origin?.outputs?.[link_info.origin_slot]?.type ?? ANY_TYPE;
    inp.type = otype;
    if (inp.name === "+" || inp.name === "") {
        const oname = origin?.outputs?.[link_info.origin_slot]?.name || otype;
        inp.name = uniqueKey(node, slugify(oname));
    }
    inp.removable = true;
}

function slugify(s) {
    return String(s || "value")
        .toLowerCase()
        .replace(/[^a-z0-9_]+/g, "_")
        .replace(/^_+|_+$/g, "") || "value";
}

function uniqueKey(node, base) {
    const taken = new Set((node.inputs ?? []).map((i) => i.name));
    if (!taken.has(base)) return base;
    let n = 2;
    while (taken.has(`${base}_${n}`)) n++;
    return `${base}_${n}`;
}

function pipeCreateMenu(node, options) {
    options.push(null);
    for (let i = 0; i < (node.inputs?.length ?? 0); i++) {
        const inp = node.inputs[i];
        if (inp.link == null) continue;
        options.push({
            content: `Rename pipe key '${inp.name}'`,
            callback: () => {
                const v = prompt("Pipe key name", inp.name);
                if (v && v !== inp.name) {
                    inp.name = uniqueKey(node, slugify(v));
                    pipeCreateSync(node);
                    refreshDownstream(node);
                    node.setDirtyCanvas(true, true);
                }
            },
        });
    }
}

// ---------------------------------------------------------------------------
// PipeOut: dynamic outputs
// ---------------------------------------------------------------------------

function pipeOutReshape(node, manifest) {
    const prev = node.properties.pipe_manifest ?? [];
    node.properties.pipe_manifest = manifest;
    setWidget(node, "_manifest", JSON.stringify(manifest));

    // Manifest unchanged: outputs are already in the right positions, but
    // workflow load may have reset slot names/types to the nodeDef default —
    // patch them in place so existing links survive.
    if (sameManifest(prev, manifest)
            && (node.outputs?.length ?? 0) === Math.max(1, manifest.length)) {
        for (let i = 0; i < manifest.length; i++) {
            node.outputs[i].name = manifest[i][0];
            node.outputs[i].type = manifest[i][1];
        }
        return;
    }

    // Snapshot existing downstream link targets by key name so we can
    // reattach after rebuilding outputs (link objects are freed on remove).
    const graph = getGraph(node);
    const oldLinks = {};
    for (const out of node.outputs ?? []) {
        for (const lid of out.links ?? []) {
            const l = graph?.links?.[lid];
            if (l) {
                (oldLinks[out.name] ??= []).push({
                    target_id: l.target_id, target_slot: l.target_slot,
                });
            }
        }
    }
    while (node.outputs?.length) node.removeOutput(0);

    for (const [key, type] of manifest) {
        node.addOutput(key, type);
    }
    if (manifest.length === 0) {
        node.addOutput("*", ANY_TYPE);
    }

    // Reattach links whose key+type still match.
    for (let i = 0; i < manifest.length; i++) {
        const [key, type] = manifest[i];
        for (const l of oldLinks[key] ?? []) {
            const target = graph?.getNodeById?.(l.target_id);
            const tslot = target?.inputs?.[l.target_slot];
            if (tslot && (tslot.type === type || tslot.type === ANY_TYPE || type === ANY_TYPE)) {
                node.connect(i, target, l.target_slot);
            }
        }
    }
    node.setSize(node.computeSize());
}

function pipeGetReshape(node, manifest) {
    const key = getWidget(node, "key");
    const entry = manifest.find(([k]) => k === key);
    const type = entry ? entry[1] : ANY_TYPE;
    if (node.outputs?.[0]) {
        node.outputs[0].type = type;
        node.outputs[0].name = key || "value";
    }
    setWidget(node, "_value_type", type);
    node.properties.pipe_manifest = manifest;
}

// ---------------------------------------------------------------------------
// downstream refresh
// ---------------------------------------------------------------------------

function refreshNode(node, seen) {
    const type = node.type ?? node.comfyClass;
    if (type === NODE_OUT) {
        pipeOutReshape(node, computeManifest(node, "pipe"));
        refreshDownstream(node, seen);
    } else if (type === NODE_GET) {
        pipeGetReshape(node, computeManifest(node, "pipe"));
        refreshDownstream(node, seen);
    } else if (type === NODE_PIPE) {
        // A PIPE wired into another PipeCreate (pipe-in-pipe): re-snapshot the
        // nested manifest so the OUTER pipe's stored manifest stays in sync
        // with later edits to the inner one.
        pipeCreateSync(node);
        refreshDownstream(node, seen);
    } else if (type === NODE_REMOVE || type === NODE_SET || type === NODE_MERGE) {
        refreshDownstream(node, seen);
    }
}

function refreshDownstream(node, seen) {
    seen = seen ?? new Set();
    if (seen.has(node.id)) return;
    seen.add(node.id);
    for (const out of node.outputs ?? []) {
        if (out.type !== PIPE_TYPE) continue;
        for (const lid of out.links ?? []) {
            const l = getGraph(node)?.links?.[lid];
            const target = l && getGraph(node)?.getNodeById?.(l.target_id);
            if (target) {
                refreshNode(target, seen);
                if (!PIPE_NODES.has(target.type ?? target.comfyClass)) {
                    refreshDownstream(target, seen);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// extension
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "Comfy.PipeNodes",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!PIPE_NODES.has(nodeData.name)) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            this.properties = this.properties ?? {};
            this.properties.pipe_manifest = this.properties.pipe_manifest ?? [];
            hideWidget(this, "_manifest");
            hideWidget(this, "_value_type");

            if (nodeData.name === NODE_PIPE) {
                pipeCreateSync(this);
            }
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origConfigure?.apply(this, arguments);
            // Reconcile persisted manifest -> widget so backend sees it even
            // if widgets_values was reordered on load.
            const m = this.properties?.pipe_manifest;
            if (Array.isArray(m)) {
                setWidget(this, "_manifest", JSON.stringify(m));
            }
            if (nodeData.name === NODE_PIPE) {
                pipeCreateSync(this);
            }
        };

        const origConn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (kind, slot, connected, link_info) {
            origConn?.apply(this, arguments);
            if (app.configuringGraph || this.__pipeConfiguring) return;
            this.__pipeConfiguring = true;
            try {
                if (nodeData.name === NODE_PIPE && kind === 1) {
                    if (connected && link_info) {
                        pipeCreateOnConnect(this, slot, link_info);
                    } else if (!connected && this.inputs?.[slot]) {
                        this.inputs[slot].type = ANY_TYPE;
                    }
                    pipeCreateSync(this);
                    refreshDownstream(this);
                } else if (nodeData.name === NODE_SET && kind === 1) {
                    if (this.inputs?.[slot]?.name === "value") {
                        const o = connected && link_info
                            ? getGraph(this)?.getNodeById?.(link_info.origin_id)
                                ?.outputs?.[link_info.origin_slot]
                            : null;
                        setWidget(this, "_value_type", o?.type ?? ANY_TYPE);
                    }
                    refreshDownstream(this);
                } else if (
                    (nodeData.name === NODE_OUT || nodeData.name === NODE_GET)
                    && kind === 1 && this.inputs?.[slot]?.name === "pipe"
                ) {
                    refreshNode(this);
                } else if (
                    (nodeData.name === NODE_REMOVE || nodeData.name === NODE_MERGE)
                    && kind === 1
                ) {
                    refreshDownstream(this);
                }
            } finally {
                this.__pipeConfiguring = false;
            }
        };

        if (nodeData.name === NODE_PIPE) {
            const origMenu = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
                origMenu?.apply(this, arguments);
                pipeCreateMenu(this, options);
            };
        }

        if (
            nodeData.name === NODE_SET || nodeData.name === NODE_REMOVE
            || nodeData.name === NODE_GET || nodeData.name === NODE_MERGE
        ) {
            const origAdded = nodeType.prototype.onAdded;
            nodeType.prototype.onAdded = function () {
                origAdded?.apply(this, arguments);
                const trigger = nodeData.name === NODE_MERGE ? "collision" : "key";
                const w = findWidget(this, trigger);
                if (w) {
                    const cb = w.callback;
                    w.callback = (...args) => {
                        const r = cb?.apply(this, args);
                        if (nodeData.name === NODE_GET) refreshNode(this);
                        else refreshDownstream(this);
                        return r;
                    };
                }
            };
        }
    },

    async afterConfigureGraph() {
        // Reconcile pass: recompute manifests from the live graph and warn on
        // drift vs the persisted manifest (e.g. upstream edited externally).
        for (const node of app.graph?._nodes ?? []) {
            const type = node.type ?? node.comfyClass;
            if (type === NODE_OUT || type === NODE_GET) {
                const fresh = computeManifest(node, "pipe");
                const stored = node.properties?.pipe_manifest ?? [];
                if (fresh.length && !sameManifest(fresh, stored)) {
                    console.warn(
                        `[pipe] ${type}#${node.id}: persisted manifest `
                        + `(${manifestKeys(stored).join(",")}) differs from `
                        + `upstream (${manifestKeys(fresh).join(",")})`,
                    );
                }
                if (type === NODE_OUT) pipeOutReshape(node, fresh);
                else pipeGetReshape(node, fresh);
            }
        }
    },
});
