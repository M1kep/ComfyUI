import { app } from "../../scripts/app.js";

/**
 * Pipe node frontend extension.
 *
 * Manages dynamic input/output slots on Pipe-family nodes and propagates a
 * `{name, type}` manifest along PIPE wires at editor time so that `Pipe Out` /
 * `Pipe Get` can expose properly-typed outputs without `*` wildcards.
 *
 * The manifest is mirrored into a hidden `_manifest` STRING widget on each
 * node so the backend sees the same schema and so workflow JSON round-trips.
 */

const MAX_PIPE_KEYS = 20;

const PIPE_NODE_TYPES = new Set([
  "Pipe",
  "PipeOut",
  "PipeSet",
  "PipeRemove",
  "PipeGet",
  "PipeMerge",
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
}

function getWidgetValue(node, name, fallback) {
  const w = findWidget(node, name);
  return w ? w.value : fallback;
}

function setWidgetValue(node, name, value) {
  const w = findWidget(node, name);
  if (w && w.value !== value) w.value = value;
}

function readManifestWidget(node) {
  try {
    const v = JSON.parse(getWidgetValue(node, "_manifest", "[]"));
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

function writeManifestWidget(node, entries) {
  setWidgetValue(node, "_manifest", JSON.stringify(entries));
}

function manifestEquals(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].name !== b[i].name || a[i].type !== b[i].type) return false;
  }
  return true;
}

function originOf(node, inputName) {
  const slot = node.inputs?.find((i) => i.name === inputName);
  if (!slot || slot.link == null) return null;
  const link = node.graph?.links?.[slot.link];
  if (!link) return null;
  const origin = node.graph.getNodeById(link.origin_id);
  if (!origin) return null;
  return { node: origin, slot: link.origin_slot };
}

/** Follow 1-in/1-out passthroughs (Reroute etc.) to the real producing node. */
function resolvePassthrough(node, slot) {
  let hops = 0;
  while (node && hops++ < 64) {
    if (PIPE_NODE_TYPES.has(node.type ?? node.comfyClass)) return { node, slot };
    if (node.inputs?.length !== 1 || node.outputs?.length !== 1) break;
    const link = node.graph?.links?.[node.inputs[0].link];
    if (!link) break;
    node = node.graph.getNodeById(link.origin_id);
    slot = link.origin_slot;
  }
  return { node, slot };
}

// ---------------------------------------------------------------------------
// manifest computation
// ---------------------------------------------------------------------------

/**
 * Walk upstream from a PIPE-typed input and compute the effective manifest
 * (ordered list of {name, type}). Returns null if no upstream pipe is
 * reachable, in which case callers fall back to the persisted manifest.
 */
function computeUpstreamManifest(node, inputName, seen = new Set()) {
  const origin = originOf(node, inputName);
  if (!origin) return null;

  const { node: src } = resolvePassthrough(origin.node, origin.slot);
  if (!src || seen.has(src.id)) return null;
  seen = new Set(seen).add(src.id);

  switch (src.type ?? src.comfyClass) {
    case "Pipe":
      return readManifestWidget(src);

    case "PipeSet": {
      const base = computeUpstreamManifest(src, "pipe", seen) ?? [];
      const k = getWidgetValue(src, "key", "");
      const vt = getWidgetValue(src, "_value_type", "*");
      if (!k) return base;
      const out = base.filter((e) => e.name !== k);
      out.push({ name: k, type: vt });
      return out;
    }

    case "PipeRemove": {
      const base = computeUpstreamManifest(src, "pipe", seen) ?? [];
      const k = getWidgetValue(src, "key", "");
      return k ? base.filter((e) => e.name !== k) : base;
    }

    case "PipeMerge": {
      const a = computeUpstreamManifest(src, "a", seen) ?? [];
      const b = computeUpstreamManifest(src, "b", seen) ?? [];
      const collision = getWidgetValue(src, "collision", "left_wins");
      const primary = collision === "right_wins" ? b : a;
      const secondary = collision === "right_wins" ? a : b;
      const names = new Set(primary.map((e) => e.name));
      return [...primary, ...secondary.filter((e) => !names.has(e.name))];
    }

    default:
      // Unknown producer (e.g. subgraph output proxy). Prefer a manifest
      // persisted on that node if it carries one.
      return src.properties?.pipeManifest ?? null;
  }
}

// ---------------------------------------------------------------------------
// downstream refresh fan-out
// ---------------------------------------------------------------------------

/**
 * Refresh `node` and every node reachable downstream along PIPE wires, once.
 * `__pipeRefresh` implementations are node-local; they must not call this.
 */
function refreshFrom(node) {
  const seen = new Set();
  const queue = [node];
  while (queue.length) {
    const n = queue.shift();
    if (!n || seen.has(n.id)) continue;
    seen.add(n.id);
    n.__pipeRefresh?.();
    for (const out of n.outputs ?? []) {
      if (out.type !== "PIPE" || !out.links) continue;
      for (const linkId of out.links) {
        const link = n.graph?.links?.[linkId];
        if (link) queue.push(n.graph.getNodeById(link.target_id));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Pipe (source) — dynamic input slots
// ---------------------------------------------------------------------------

function isValueInput(slot) {
  return slot && slot.__pipeValue;
}

function setupPipeSource(node) {
  hideWidget(node, "_manifest");

  node.__pipeAddValueInput = function (name, type) {
    const input = this.addInput(name ?? "", type ?? "*");
    input.__pipeValue = true;
    input.color_off = "#666";
    return input;
  };

  node.__pipeEnsureTrailingSlot = function () {
    const valueInputs = (this.inputs ?? []).filter(isValueInput);
    const last = valueInputs[valueInputs.length - 1];
    if ((!last || last.link != null) && valueInputs.length < MAX_PIPE_KEYS) {
      this.__pipeAddValueInput();
    }
  };

  node.__pipeRefresh = function () {
    const entries = [];
    const seen = new Set();
    for (const slot of this.inputs ?? []) {
      if (!isValueInput(slot) || slot.link == null) continue;
      const dup = !slot.name || seen.has(slot.name);
      if (dup) slot.color_on = "#c61010";
      else delete slot.color_on;
      seen.add(slot.name);
      entries.push({ name: slot.name, type: slot.type === "*" ? "*" : slot.type });
    }
    writeManifestWidget(this, entries);
  };

  // Restore slots from persisted manifest on load (slots reconstruct before
  // LiteGraph reattaches links by index).
  const existing = (node.inputs ?? []).filter((s) => s.name !== "_manifest");
  if (existing.length === 0) {
    for (const e of readManifestWidget(node)) node.__pipeAddValueInput(e.name, e.type);
  } else {
    for (const slot of existing) slot.__pipeValue = true;
  }
  node.__pipeEnsureTrailingSlot();

  const origMenu = node.getExtraMenuOptions;
  node.getExtraMenuOptions = function (canvas, options) {
    origMenu?.call(this, canvas, options);
    options.push({
      content: "Rename pipe key...",
      callback: () => {
        const slots = (this.inputs ?? []).filter((s) => isValueInput(s) && s.link != null);
        if (!slots.length) return;
        const idx = parseInt(
          prompt(
            `Slot index to rename? (0-${slots.length - 1})\n[${slots.map((s) => s.name).join(", ")}]`,
            "0"
          ),
          10
        );
        if (Number.isNaN(idx) || !slots[idx]) return;
        const name = prompt("New key name:", slots[idx].name || "");
        if (name == null) return;
        slots[idx].name = name;
        slots[idx].label = name;
        refreshFrom(this);
        this.setDirtyCanvas(true, true);
      },
    });
  };
}

function pipeSourceOnConnectionsChange(node, type, index, connected, linkInfo) {
  if (type !== LiteGraph.INPUT) return;
  const slot = node.inputs?.[index];
  if (!isValueInput(slot)) return;

  if (connected && linkInfo) {
    const origin = node.graph?.getNodeById(linkInfo.origin_id);
    const originSlot = origin?.outputs?.[linkInfo.origin_slot];
    const originType = originSlot?.type ?? linkInfo.type ?? "*";
    slot.type = originType;
    if (!slot.name) {
      const base = (originSlot?.name || originType || "value").toLowerCase();
      const used = new Set((node.inputs ?? []).filter(isValueInput).map((s) => s.name));
      let name = base;
      let i = 2;
      while (used.has(name)) name = `${base}${i++}`;
      slot.name = name;
      slot.label = name;
    }
    node.__pipeEnsureTrailingSlot();
  } else if (!connected) {
    const valueInputs = (node.inputs ?? []).filter(isValueInput);
    if (valueInputs.length > 1) {
      node.removeInput(index);
    } else {
      slot.name = "";
      slot.label = "";
      slot.type = "*";
    }
    node.__pipeEnsureTrailingSlot();
  }
  refreshFrom(node);
}

// ---------------------------------------------------------------------------
// PipeOut — dynamic output slots
// ---------------------------------------------------------------------------

function setupPipeOut(node) {
  hideWidget(node, "_manifest");

  node.__pipeRefresh = function () {
    const computed = computeUpstreamManifest(this, "pipe");
    const manifest = (computed ?? readManifestWidget(this)).slice(0, MAX_PIPE_KEYS);
    if (computed != null) writeManifestWidget(this, manifest);
    reshapeOutputs(this, manifest);
  };
}

function reshapeOutputs(node, manifest) {
  const current = (node.outputs ?? []).map((o) => ({ name: o.name, type: o.type }));
  if (manifestEquals(current, manifest)) return;

  // Capture downstream targets before tearing down outputs (removeOutput
  // deletes the link entries from graph.links).
  const oldLinksByName = {};
  for (const out of node.outputs ?? []) {
    if (!out.links?.length) continue;
    const targets = [];
    for (const linkId of out.links) {
      const link = node.graph?.links?.[linkId];
      if (link) targets.push({ id: link.target_id, slot: link.target_slot });
    }
    oldLinksByName[out.name] = { type: out.type, targets };
  }

  while (node.outputs?.length) node.removeOutput(0);

  for (const e of manifest) {
    node.addOutput(e.name, e.type);
    const prev = oldLinksByName[e.name];
    if (!prev || prev.type !== e.type) continue;
    const slotIdx = node.outputs.length - 1;
    for (const t of prev.targets) {
      const target = node.graph?.getNodeById(t.id);
      if (target) node.connect(slotIdx, target, t.slot);
    }
  }
  node.setSize(node.computeSize());
}

// ---------------------------------------------------------------------------
// PipeSet / PipeGet / PipeRemove — key dropdown + type sync
// ---------------------------------------------------------------------------

function wrapKeyWidget(node) {
  const keyWidget = findWidget(node, "key");
  if (!keyWidget) return;
  const origCallback = keyWidget.callback;
  keyWidget.callback = function (v) {
    origCallback?.call(this, v);
    refreshFrom(node);
  };
  node.__pipeUpdateKeyOptions = function () {
    const manifest = computeUpstreamManifest(this, "pipe") ?? [];
    keyWidget.options = keyWidget.options || {};
    keyWidget.options.values = manifest.map((e) => e.name);
  };
}

function setupPipeKeyed(node, hideValueType) {
  if (hideValueType) hideWidget(node, "_value_type");
  wrapKeyWidget(node);
  node.__pipeRefresh = function () {
    this.__pipeUpdateKeyOptions?.();
  };
}

function pipeSetOnConnectionsChange(node, type, index, connected, linkInfo) {
  if (type !== LiteGraph.INPUT) return;
  const slot = node.inputs?.[index];
  if (slot?.name === "value") {
    if (connected && linkInfo) {
      const origin = node.graph?.getNodeById(linkInfo.origin_id);
      const originType = origin?.outputs?.[linkInfo.origin_slot]?.type ?? linkInfo.type ?? "*";
      slot.type = originType;
      setWidgetValue(node, "_value_type", originType);
    } else {
      slot.type = "*";
      setWidgetValue(node, "_value_type", "*");
    }
  }
  refreshFrom(node);
}

function setupPipeGet(node) {
  hideWidget(node, "_value_type");
  wrapKeyWidget(node);

  node.__pipeRefresh = function () {
    this.__pipeUpdateKeyOptions?.();
    const manifest = computeUpstreamManifest(this, "pipe") ?? [];
    const key = getWidgetValue(this, "key", "");
    const entry = manifest.find((e) => e.name === key);
    const vt = entry?.type ?? getWidgetValue(this, "_value_type", "*");
    setWidgetValue(this, "_value_type", vt);
    if (this.outputs?.[0]) {
      this.outputs[0].type = vt;
      this.outputs[0].name = key || "value";
      this.outputs[0].label = key || "value";
    }
  };
}

function setupPipeMerge(node) {
  const w = findWidget(node, "collision");
  if (w) {
    const orig = w.callback;
    w.callback = function (v) {
      orig?.call(this, v);
      refreshFrom(node);
    };
  }
  node.__pipeRefresh = function () {};
}

// ---------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------

const SETUP = {
  Pipe: setupPipeSource,
  PipeOut: setupPipeOut,
  PipeSet: (n) => setupPipeKeyed(n, true),
  PipeRemove: (n) => setupPipeKeyed(n, false),
  PipeGet: setupPipeGet,
  PipeMerge: setupPipeMerge,
};

const CONNECTION_HANDLERS = {
  Pipe: pipeSourceOnConnectionsChange,
  PipeSet: pipeSetOnConnectionsChange,
};

function ensureSetup(node, name) {
  if (node.__pipeSetupDone) return;
  node.__pipeSetupDone = true;
  SETUP[name]?.(node);
}

app.registerExtension({
  name: "Comfy.PipeNodes",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!PIPE_NODE_TYPES.has(nodeData.name)) return;

    const origCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origCreated?.apply(this, arguments);
      ensureSetup(this, nodeData.name);
      return r;
    };

    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = origConfigure?.apply(this, arguments);
      ensureSetup(this, nodeData.name);
      return r;
    };

    const origConn = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
      const r = origConn?.apply(this, arguments);
      const handler = CONNECTION_HANDLERS[nodeData.name];
      if (handler) {
        handler(this, type, index, connected, linkInfo);
      } else if (type === LiteGraph.INPUT) {
        refreshFrom(this);
      }
      return r;
    };
  },

  loadedGraphNode(node) {
    if (PIPE_NODE_TYPES.has(node.type ?? node.comfyClass)) {
      queueMicrotask(() => node.__pipeRefresh?.());
    }
  },
});
