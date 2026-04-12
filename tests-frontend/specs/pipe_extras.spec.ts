// Browser tests for the higher-level Pipe helpers (Pipe Pick, Bundle into
// Pipe). Split out from pipe.spec.ts so the per-server page-load budget in
// the local sandbox isn't exhausted by a single file.
import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '@e2e/fixtures/ComfyPage'

import {
  addNode,
  bundle,
  connect,
  nodeInfo,
  outSlot,
  setWidget,
  slotIndex
} from './_pipe_helpers'

test.describe('Pipe Pick', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.page.evaluate(() => window.app!.graph.clear())
  })

  test('grows a key combo per selection and reshapes outputs', async ({
    comfyPage
  }) => {
    const { page } = comfyPage
    const sA = await addNode(page, 'PrimitiveString')
    const sB = await addNode(page, 'PrimitiveString')
    const sC = await addNode(page, 'PrimitiveString')
    const pipeId = await addNode(page, 'PipeCreate')
    const pickId = await addNode(page, 'PipePick')

    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, sB, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, sC, 0, pipeId, await slotIndex(page, pipeId, '+'))
    const keys = ((await nodeInfo(page, pipeId))!.manifest as [string, string][])
      .map(([k]) => k)
    await connect(page, pipeId, 0, pickId, await slotIndex(page, pickId, 'pipe'))

    const combo = (name: string) => page.evaluate(({ id, name }) => {
      const n = window.app!.graph.getNodeById(id)!
      const w = n.widgets!.find((w) => w.name === name)
      return w ? { value: w.value, values: (w.options as any)?.values } : null
    }, { id: pickId, name })

    let c1 = await combo('key_1')
    expect(c1!.values).toEqual(['', ...keys])
    expect((await nodeInfo(page, pickId))!.outputs.map((o) => o.name))
      .toEqual(['pipe'])

    await setWidget(page, pickId, 'key_1', keys[2])
    expect(await combo('key_2')).not.toBeNull()
    await setWidget(page, pickId, 'key_2', keys[0])
    expect(await combo('key_3')).not.toBeNull()

    const out = await nodeInfo(page, pickId)
    expect(out!.outputs.map((o) => o.name)).toEqual(['pipe', keys[2], keys[0]])
    for (const o of out!.outputs.slice(1)) expect(o.type).toBe('STRING')
  })

  test('clearing a key drops its output but preserves the surviving wire', async ({
    comfyPage
  }) => {
    const { page } = comfyPage
    const sA = await addNode(page, 'PrimitiveString')
    const sB = await addNode(page, 'PrimitiveString')
    const pipeId = await addNode(page, 'PipeCreate')
    const pickId = await addNode(page, 'PipePick')
    const sink = await addNode(page, 'PreviewAny')

    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, sB, 0, pipeId, await slotIndex(page, pipeId, '+'))
    const keys = ((await nodeInfo(page, pipeId))!.manifest as [string, string][])
      .map(([k]) => k)
    await connect(page, pipeId, 0, pickId, await slotIndex(page, pickId, 'pipe'))

    await setWidget(page, pickId, 'key_1', keys[0])
    await setWidget(page, pickId, 'key_2', keys[1])
    await connect(page, pickId, await outSlot(page, pickId, keys[1]), sink, 0)

    await setWidget(page, pickId, 'key_1', '')
    const out = await nodeInfo(page, pickId)
    expect(out!.outputs.map((o) => o.name)).toEqual(['pipe', keys[1]])
    const sinkLink = await page.evaluate(
      (id) => window.app!.graph.getNodeById(id)!.inputs[0].link,
      sink
    )
    expect(sinkLink).not.toBeNull()
  })

  test('round-trip restores selected keys and outputs', async ({
    comfyPage
  }) => {
    const { page } = comfyPage
    const sA = await addNode(page, 'PrimitiveString')
    const sB = await addNode(page, 'PrimitiveString')
    const pipeId = await addNode(page, 'PipeCreate')
    const pickId = await addNode(page, 'PipePick')

    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, sB, 0, pipeId, await slotIndex(page, pipeId, '+'))
    const keys = ((await nodeInfo(page, pipeId))!.manifest as [string, string][])
      .map(([k]) => k)
    await connect(page, pipeId, 0, pickId, await slotIndex(page, pickId, 'pipe'))
    await setWidget(page, pickId, 'key_1', keys[1])

    const before = (await nodeInfo(page, pickId))!.outputs.map((o) => o.name)
    const ser = await page.evaluate(() =>
      JSON.parse(JSON.stringify(window.app!.graph.serialize()))
    )
    await page.evaluate(async (s) => { await window.app!.loadGraphData(s) }, ser)

    const picks = await comfyPage.nodeOps.getNodeRefsByType('PipePick')
    const after = await nodeInfo(page, Number(picks[0].id))
    expect(after!.outputs.map((o) => o.name)).toEqual(before)
  })
})

test.describe('Bundle into Pipe', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.page.evaluate(() => window.app!.graph.clear())
  })

  test('wires each selected node\'s first output into a fresh PipeCreate', async ({
    comfyPage
  }) => {
    const { page } = comfyPage
    const sA = await addNode(page, 'PrimitiveString')
    const iA = await addNode(page, 'PrimitiveInt')

    const { pipeId, menuPresent } = await bundle(page, [sA, iA])
    expect(menuPresent).toBe(true)
    expect(pipeId).not.toBeNull()

    const info = await nodeInfo(page, pipeId!)
    expect((info!.manifest as unknown[]).length).toBe(2)
    const types = (info!.manifest as [string, string][])
      .map(([, t]) => t).sort()
    expect(types).toEqual(['INT', 'STRING'])

    const [pipeX, srcRight] = await page.evaluate(
      ({ pipeId, sA }) => {
        const p = window.app!.graph.getNodeById(pipeId)!
        const s = window.app!.graph.getNodeById(sA)!
        return [p.pos[0], s.pos[0] + s.size[0]]
      },
      { pipeId: pipeId!, sA }
    )
    expect(pipeX).toBeGreaterThan(srcRight)
  })

  test('skips selected nodes that have no outputs', async ({ comfyPage }) => {
    const { page } = comfyPage
    const sA = await addNode(page, 'PrimitiveString')
    const sink = await addNode(page, 'PreviewAny')

    const { pipeId } = await bundle(page, [sA, sink])
    expect(pipeId).not.toBeNull()
    const m = (await nodeInfo(page, pipeId!))!.manifest as [string, string][]
    expect(m).toHaveLength(1)
    expect(m[0][1]).toBe('STRING')
  })

  test('bundling a PipeCreate produces a PIPE-typed key with a nested manifest', async ({
    comfyPage
  }) => {
    const { page } = comfyPage
    const sA = await addNode(page, 'PrimitiveString')
    const inner = await addNode(page, 'PipeCreate')
    await connect(page, sA, 0, inner, await slotIndex(page, inner, '+'))

    const { pipeId } = await bundle(page, [inner])
    expect(pipeId).not.toBeNull()
    const m = (await nodeInfo(page, pipeId!))!.manifest as unknown[][]
    expect(m).toHaveLength(1)
    expect(m[0][1]).toBe('PIPE')
    expect((m[0][2] as unknown[]).length).toBe(1)
  })

  test('does not steal an existing wire on the bundled output', async ({
    comfyPage
  }) => {
    const { page } = comfyPage
    const sA = await addNode(page, 'PrimitiveString')
    const sink = await addNode(page, 'PreviewAny')
    await connect(page, sA, 0, sink, 0)

    const { pipeId } = await bundle(page, [sA])
    expect(pipeId).not.toBeNull()
    const sinkLink = await page.evaluate(
      (id) => window.app!.graph.getNodeById(id)!.inputs[0].link,
      sink
    )
    expect(sinkLink).not.toBeNull()
    expect(((await nodeInfo(page, pipeId!))!.manifest as unknown[]).length).toBe(1)
  })

  test('menu item is absent when nothing is selected', async ({ comfyPage }) => {
    const { page } = comfyPage
    const { menuPresent, pipeId } = await bundle(page, [])
    expect(menuPresent).toBe(false)
    expect(pipeId).toBeNull()
  })
})
