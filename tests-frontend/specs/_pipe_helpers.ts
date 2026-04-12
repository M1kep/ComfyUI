// Shared helpers for the Pipe browser specs.
import type { Page } from '@playwright/test'

export async function addNode(page: Page, type: string): Promise<number> {
  return page.evaluate((type) => {
    const n = window.LiteGraph!.createNode(type, undefined, {})
    window.app!.graph.add(n)
    return Number(n!.id)
  }, type)
}

export async function connect(
  page: Page,
  fromId: number,
  fromSlot: number,
  toId: number,
  toSlot: number
) {
  await page.evaluate(
    ({ fromId, fromSlot, toId, toSlot }) => {
      const a = window.app!.graph.getNodeById(fromId)!
      const b = window.app!.graph.getNodeById(toId)!
      a.connect(fromSlot, b, toSlot)
    },
    { fromId, fromSlot, toId, toSlot }
  )
}

export async function setWidget(
  page: Page,
  id: number,
  name: string,
  value: unknown
) {
  await page.evaluate(
    ({ id, name, value }) => {
      const n = window.app!.graph.getNodeById(id)!
      const w = n.widgets?.find((w) => w.name === name)
      if (w) {
        w.value = value
        // @ts-expect-error — optional callback used by some widget types
        w.callback?.(value)
      }
    },
    { id, name, value }
  )
}

export async function nodeInfo(page: Page, id: number) {
  return page.evaluate((id) => {
    const n = window.app!.graph.getNodeById(id)
    if (!n) return null
    const outputs = (n.outputs ?? []).map((o) => ({
      name: o.name,
      type: o.type,
      linkCount: o.links?.length ?? 0
    }))
    return {
      id: n.id,
      // Only socket inputs (widget-backed slots are an implementation detail
      // of the frontend and the pipe extension strips/ignores them).
      inputs: (n.inputs ?? [])
        .filter((i) => !i.widget)
        .map((i) => ({
          name: i.name,
          type: i.type,
          linked: i.link != null
        })),
      outputs,
      // PipeOut/PipePick output[0] is the PIPE passthrough; key outputs are 1+.
      keyOutputs:
        (n.type === 'PipeOut' || n.type === 'PipePick')
        && outputs[0]?.type === 'PIPE'
          ? outputs.slice(1)
          : outputs,
      manifest: (n.properties ?? {}).pipe_manifest ?? null
    }
  }, id)
}

export async function slotIndex(
  page: Page,
  id: number,
  name: string
): Promise<number> {
  return page.evaluate(
    ({ id, name }) => {
      const n = window.app!.graph.getNodeById(id)!
      return (n.inputs ?? []).findIndex((i) => i.name === name)
    },
    { id, name }
  )
}

export async function outSlot(
  page: Page,
  id: number,
  name: string
): Promise<number> {
  return page.evaluate(
    ({ id, name }) => {
      const n = window.app!.graph.getNodeById(id)!
      return (n.outputs ?? []).findIndex((o) => o.name === name)
    },
    { id, name }
  )
}

/** Invoke the "Bundle into Pipe" canvas-menu callback for the given node ids
 *  (or report it absent). Returns the new PipeCreate's id, or null. */
export async function bundle(
  page: Page,
  ids: number[]
): Promise<{ pipeId: number | null; menuPresent: boolean }> {
  return page.evaluate((ids) => {
    const canvas = window.app!.canvas
    canvas.selected_nodes = Object.fromEntries(
      ids.map((id) => [id, window.app!.graph.getNodeById(id)])
    )
    const items: any[] = []
    // @ts-expect-error — frontend extension hook
    for (const ext of window.app!.extensions ?? []) {
      if (typeof ext.getCanvasMenuItems === 'function') {
        items.push(...(ext.getCanvasMenuItems(canvas) ?? []))
      }
    }
    const item = items.find((i) => i && /Bundle .* into Pipe/.test(i.content))
    if (!item) return { pipeId: null, menuPresent: false }
    const before = new Set(
      window.app!.graph.nodes
        .filter((n) => n.type === 'PipeCreate')
        .map((n) => n.id)
    )
    item.callback()
    const created = window.app!.graph.nodes.find(
      (n) => n.type === 'PipeCreate' && !before.has(n.id)
    )
    return { pipeId: created ? Number(created.id) : null, menuPresent: true }
  }, ids)
}
