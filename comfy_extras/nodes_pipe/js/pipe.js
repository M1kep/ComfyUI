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
const NODE_PICK = "PipePick";
const NODE_SET = "PipeSet";
const NODE_REMOVE = "PipeRemove";
const NODE_GET = "PipeGet";
const NODE_MERGE = "PipeMerge";

const PIPE_NODES = new Set([
    NODE_PIPE, NODE_OUT, NODE_PICK, NODE_SET, NODE_REMOVE, NODE_GET, NODE_MERGE,
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
    if (type === NODE_OUT || type === NODE_PICK || type === NODE_GET) {
        const upstream = computeManifest(node, "pipe", seen);
        // Slot 0 on Pipe Out / Pipe Pick is the PIPE passthrough.
        if ((type === NODE_OUT || type === NODE_PICK) && slot === 0) return upstream;
        // Other outputs are unpacked entries; if one is itself a PIPE, return
        // its nested manifest so a downstream Pipe Out can reshape correctly.
        const outName = node.outputs?.[slot]?.name;
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

function isDeclared(node, inp) {
    return (node.properties?.pipe_declared ?? []).includes(inp.name);
}

function pipeCreateSync(node) {
    // Ensure one trailing empty ANY slot, prune disconnected non-trailing
    // ad-hoc slots (declared slots are kept even when empty), and rebuild the
    // manifest from the current connected inputs. Widget-backed inputs are
    // never treated as pipe slots.
    let trailing = -1;
    for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--) {
        if (isPipeSlot(node.inputs[i])) { trailing = i; break; }
    }
    for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--) {
        const inp = node.inputs[i];
        if (isPipeSlot(inp) && inp.link == null && i < trailing
                && !isDeclared(node, inp)) {
            node.removeInput(i);
            trailing--;
        }
    }
    const manifest = [];
    for (const inp of node.inputs ?? []) {
        if (!isPipeSlot(inp) || inp.link == null) continue;
        let nested = null;
        if (inp.type === PIPE_TYPE) {
            const link = getGraph(node)?.links?.[inp.link];
            const src = link && getGraph(node)?.getNodeById?.(link.origin_id);
            if (src) nested = manifestOfOutput(src, link.origin_slot, new Set());
        }
        manifest.push(entryFor(inp.name, inp.type, nested));
    }
    const last = trailing >= 0 ? node.inputs[trailing] : null;
    if (!last || last.link != null || isDeclared(node, last)) {
        node.addInput("+", ANY_TYPE);
    }
    commitManifest(node, manifest);
}

function pipeCreateOnConnect(node, slotIndex, link_info) {
    const inp = node.inputs[slotIndex];
    if (!isPipeSlot(inp)) return;
    const origin = getGraph(node)?.getNodeById?.(link_info.origin_id);
    const otype = origin?.outputs?.[link_info.origin_slot]?.type ?? ANY_TYPE;
    // Declared slots keep their authored type; ad-hoc slots adopt upstream's.
    if (!isDeclared(node, inp)) inp.type = otype;
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
    // "pipe" is the passthrough output's reserved name on Pipe Out — avoid
    // auto-naming a key "pipe" so the two never collide.
    const taken = new Set((node.inputs ?? []).map((i) => i.name));
    taken.add("pipe");
    if (!taken.has(base)) return base;
    let n = 2;
    while (taken.has(`${base}_${n}`)) n++;
    return `${base}_${n}`;
}

function pipeCreateAddDeclared(node, name, type) {
    name = uniqueKey(node, slugify(name || type));
    type = String(type || ANY_TYPE).trim().toUpperCase();
    node.properties.pipe_declared = [
        ...(node.properties.pipe_declared ?? []), name,
    ];
    // Insert before the trailing "+" so the grow slot stays last.
    const plus = (node.inputs ?? []).findIndex((i) => i.name === "+");
    node.addInput(name, type);
    if (plus >= 0 && plus < node.inputs.length - 1) {
        const [slot] = node.inputs.splice(node.inputs.length - 1, 1);
        node.inputs.splice(plus, 0, slot);
    }
    pipeCreateSync(node);
    refreshDownstream(node);
    node.setDirtyCanvas(true, true);
}

function pipeCreateMenu(node, options) {
    options.push(null, {
        content: "Add typed input…",
        callback: () => {
            const type = prompt("Slot type (e.g. MODEL, CLIP, VAE)", "");
            if (!type) return;
            const name = prompt("Slot name", slugify(type));
            pipeCreateAddDeclared(node, name, type);
        },
    });
    for (let i = 0; i < (node.inputs?.length ?? 0); i++) {
        const inp = node.inputs[i];
        if (!isPipeSlot(inp) || inp.name === "+") continue;
        const declared = isDeclared(node, inp);
        options.push({
            content: `Rename pipe key '${inp.name}'${declared ? " (declared)" : ""}`,
            callback: () => {
                const v = prompt("Pipe key name", inp.name);
                if (v && v !== inp.name) {
                    const old = inp.name;
                    inp.name = uniqueKey(node, slugify(v));
                    if (declared) {
                        node.properties.pipe_declared =
                            node.properties.pipe_declared.map(
                                (n) => (n === old ? inp.name : n),
                            );
                    }
                    pipeCreateSync(node);
                    refreshDownstream(node);
                    node.setDirtyCanvas(true, true);
                }
            },
        });
        if (declared) {
            options.push({
                content: `Remove declared input '${inp.name}'`,
                callback: () => {
                    node.properties.pipe_declared =
                        node.properties.pipe_declared.filter((n) => n !== inp.name);
                    node.removeInput(i);
                    pipeCreateSync(node);
                    refreshDownstream(node);
                    node.setDirtyCanvas(true, true);
                },
            });
        }
    }
}

// ---------------------------------------------------------------------------
// PipeOut: dynamic outputs
// ---------------------------------------------------------------------------

function commitManifest(node, manifest) {
    node.properties.pipe_manifest = manifest;
    setWidget(node, "_manifest", JSON.stringify(manifest));
}

// Rebuild a node's outputs to match `want` (list of [name, type]) while
// preserving downstream links by output name. If multiple `want` entries share
// a name with old outputs, links are matched positionally within that name so
// duplicates don't all collapse onto the first slot.
function rebuildOutputs(node, want) {
    if ((node.outputs?.length ?? 0) === want.length
            && want.every(([k, t], i) =>
                node.outputs[i]?.name === k && node.outputs[i]?.type === t)) {
        return;
    }
    const graph = getGraph(node);
    const old = (node.outputs ?? []).map((o) => ({
        name: o.name,
        type: o.type,
        links: (o.links ?? []).map((lid) => {
            const l = graph?.links?.[lid];
            return l && { target_id: l.target_id, target_slot: l.target_slot };
        }).filter(Boolean),
    }));
    while (node.outputs?.length) node.removeOutput(0);
    for (const [k, t] of want) node.addOutput(k, t);

    const used = new Set();
    for (let i = 0; i < want.length; i++) {
        const [name, type] = want[i];
        const j = old.findIndex((o, idx) => o.name === name && !used.has(idx));
        if (j < 0) continue;
        used.add(j);
        for (const l of old[j].links) {
            const target = graph?.getNodeById?.(l.target_id);
            const tslot = target?.inputs?.[l.target_slot];
            if (tslot && (tslot.type === type || tslot.type === ANY_TYPE || type === ANY_TYPE)) {
                node.connect(i, target, l.target_slot);
            }
        }
    }
    node.setSize(node.computeSize());
}

function pipeOutReshape(node, manifest) {
    const prev = node.properties.pipe_manifest ?? [];
    commitManifest(node, manifest);

    const want = [["pipe", PIPE_TYPE], ...manifest.map(([k, t]) => [k, t])];

    // Manifest unchanged: outputs are already in the right positions, but
    // workflow load may have reset slot names/types to the nodeDef default —
    // patch them in place so existing links survive.
    if (sameManifest(prev, manifest)
            && (node.outputs?.length ?? 0) === want.length) {
        for (let i = 0; i < want.length; i++) {
            node.outputs[i].name = want[i][0];
            node.outputs[i].type = want[i][1];
        }
        return;
    }
    rebuildOutputs(node, want);
}

function pipeGetReshape(node, manifest) {
    populateKeyDropdown(node, manifest);
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

// Convert the `key` text widget on PipeRemove / PipeGet into a combo whose
// options reflect the upstream manifest. Preserves the current value (so a
// stale key from a workflow load is still visible / editable).
function populateKeyDropdown(node, manifest) {
    const w = findWidget(node, "key");
    if (!w) return;
    const keys = manifestKeys(manifest);
    if (w.value && !keys.includes(w.value)) keys.push(w.value);
    w.type = "combo";
    w.options = { ...(w.options ?? {}), values: keys };
    if (!w.value && keys.length) w.value = keys[0];
}

// ---------------------------------------------------------------------------
// downstream refresh
// ---------------------------------------------------------------------------

function refreshNode(node, seen) {
    const type = node.type ?? node.comfyClass;
    if (type === NODE_OUT) {
        pipeOutReshape(node, computeManifest(node, "pipe"));
    } else if (type === NODE_PICK) {
        pipePickRefresh(node);
    } else if (type === NODE_GET) {
        pipeGetReshape(node, computeManifest(node, "pipe"));
    } else if (type === NODE_PIPE) {
        // A PIPE wired into another PipeCreate (pipe-in-pipe): re-snapshot the
        // nested manifest so the OUTER pipe's stored manifest stays in sync
        // with later edits to the inner one.
        pipeCreateSync(node);
    } else if (type === NODE_REMOVE) {
        populateKeyDropdown(node, computeManifest(node, "pipe"));
    } else if (type !== NODE_SET && type !== NODE_MERGE) {
        return;
    }
    refreshDownstream(node, seen);
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
// PipePick: growing key dropdowns
// ---------------------------------------------------------------------------

function pipePickKeys(node) {
    const seen = new Set();
    const keys = [];
    for (const w of node.widgets ?? []) {
        if (!w.name?.startsWith("key_") || !w.value) continue;
        if (seen.has(w.value)) continue;
        seen.add(w.value);
        keys.push(w.value);
    }
    return keys;
}

function pipePickAddCombo(node, value) {
    const idx = (node.widgets ?? []).filter((w) => w.name?.startsWith("key_")).length + 1;
    const upstream = node.properties?.pipe_upstream ?? [];
    const w = node.addWidget(
        "combo",
        `key_${idx}`,
        value ?? "",
        () => pipePickSync(node),
        { values: ["", ...manifestKeys(upstream)] },
    );
    w.serialize = false;
    return w;
}

function pipePickSync(node) {
    const upstream = node.properties?.pipe_upstream ?? [];
    const keys = pipePickKeys(node);
    const selected = keys
        .map((k) => {
            const e = upstream.find(([uk]) => uk === k);
            return e ? entryFor(k, e[1], nestedOf(e)) : [k, ANY_TYPE];
        });

    node.properties.pipe_pick_keys = keys;
    commitManifest(node, selected);

    // Ensure exactly one trailing blank combo for the next pick.
    const combos = (node.widgets ?? []).filter((w) => w.name?.startsWith("key_"));
    if (!combos.length || combos[combos.length - 1].value) {
        pipePickAddCombo(node, "");
    }
    // Refresh dropdown options on every combo.
    const opts = ["", ...manifestKeys(upstream)];
    for (const c of (node.widgets ?? [])) {
        if (c.name?.startsWith("key_")) c.options = { ...c.options, values: opts };
    }

    rebuildOutputs(node, [["pipe", PIPE_TYPE], ...selected.map(([k, t]) => [k, t])]);
}

function pipePickRefresh(node) {
    node.properties.pipe_upstream = computeManifest(node, "pipe");
    pipePickSync(node);
}

// ---------------------------------------------------------------------------
// "Bundle into Pipe" — canvas selection helper
// ---------------------------------------------------------------------------

function bundleSelectionIntoPipe(canvas) {
    const graph = canvas.graph;
    const selected = Object.values(canvas.selected_nodes ?? {})
        .filter((n) => (n.outputs?.length ?? 0) > 0);
    if (!selected.length) return;

    // Place the new Pipe to the right of the selection bounding box.
    let maxX = -Infinity, minY = Infinity;
    for (const n of selected) {
        maxX = Math.max(maxX, n.pos[0] + n.size[0]);
        minY = Math.min(minY, n.pos[1]);
    }
    const pipe = window.LiteGraph.createNode(NODE_PIPE, undefined, {});
    pipe.pos = [maxX + 60, minY];
    graph.add(pipe);

    for (const n of selected) {
        const idx = (pipe.inputs ?? []).findIndex(
            (i) => isPipeSlot(i) && i.link == null,
        );
        if (idx < 0) break;
        n.connect(0, pipe, idx);
    }
    canvas.setDirty(true, true);
}

// ---------------------------------------------------------------------------
// extension
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "Comfy.PipeNodes",

    __pipeTest: { addDeclared: pipeCreateAddDeclared },

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
            } else if (nodeData.name === NODE_PICK) {
                this.properties.pipe_upstream = [];
                this.properties.pipe_pick_keys = [];
                pipePickAddCombo(this, "");
                pipePickSync(this);
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
            } else if (nodeData.name === NODE_PICK) {
                // Restore the dynamic combos from persisted keys.
                const keys = this.properties?.pipe_pick_keys ?? [];
                this.widgets = (this.widgets ?? [])
                    .filter((w) => !w.name?.startsWith("key_"));
                for (const k of keys) pipePickAddCombo(this, k);
                pipePickSync(this);
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
                    } else if (!connected && this.inputs?.[slot]
                            && !isDeclared(this, this.inputs[slot])) {
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
                    (nodeData.name === NODE_OUT || nodeData.name === NODE_PICK
                        || nodeData.name === NODE_GET)
                    && kind === 1 && this.inputs?.[slot]?.name === "pipe"
                ) {
                    refreshNode(this);
                } else if (
                    (nodeData.name === NODE_REMOVE || nodeData.name === NODE_MERGE)
                    && kind === 1
                ) {
                    if (nodeData.name === NODE_REMOVE
                            && this.inputs?.[slot]?.name === "pipe") {
                        populateKeyDropdown(this, computeManifest(this, "pipe"));
                    }
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

    getCanvasMenuItems(canvas) {
        const selected = Object.values(canvas?.selected_nodes ?? {});
        if (!selected.length) return [];
        return [
            null,
            {
                content: `Bundle ${selected.length} output${selected.length > 1 ? "s" : ""} into Pipe`,
                callback: () => bundleSelectionIntoPipe(canvas),
            },
        ];
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
            } else if (type === NODE_PICK) {
                pipePickRefresh(node);
            } else if (type === NODE_REMOVE) {
                populateKeyDropdown(node, computeManifest(node, "pipe"));
            }
        }
    },
});
