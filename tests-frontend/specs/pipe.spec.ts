// Browser-level tests for the Pipe node family.
//
// These exercise the JS extension in `comfy_extras/nodes_pipe/js/pipe.js`:
//   - PipeCreate: grow-on-connect input slots with per-slot naming
//   - PipeOut / PipeGet: dynamic output reshape driven by an upstream
//     manifest computed by walking Pipe* nodes
//   - PipeSet / PipeRemove: manifest propagation
//   - End-to-end queue through the backend so runtime values match the
//     frontend-declared manifest
//
// Graphs are built programmatically rather than loaded from JSON so the
// full connect path (onConnectionsChange, reshape, serialize) is what we
// assert against.
import { expect } from '@playwright/test'
import type { Page } from '@playwright/test'

import { comfyPageFixture as test } from '@e2e/fixtures/ComfyPage'

async function addNode(page: Page, type: string): Promise<number> {
  return page.evaluate((type) => {
    const n = window.LiteGraph!.createNode(type, undefined, {})
    window.app!.graph.add(n)
    return Number(n!.id)
  }, type)
}

async function connect(
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

async function setWidget(page: Page, id: number, name: string, value: unknown) {
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

async function nodeInfo(page: Page, id: number) {
  return page.evaluate((id) => {
    const n = window.app!.graph.getNodeById(id)
    if (!n) return null
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
      outputs: (n.outputs ?? []).map((o) => ({
        name: o.name,
        type: o.type,
        linkCount: o.links?.length ?? 0
      })),
      manifest: (n.properties ?? {}).pipe_manifest ?? null
    }
  }, id)
}

async function slotIndex(page: Page, id: number, name: string): Promise<number> {
  return page.evaluate(
    ({ id, name }) => {
      const n = window.app!.graph.getNodeById(id)!
      return (n.inputs ?? []).findIndex((i) => i.name === name)
    },
    { id, name }
  )
}

test.describe('Pipe nodes (frontend)', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.page.evaluate(() => window.app!.graph.clear())
  })

  test('PipeCreate grows a trailing slot and names each key after the upstream', async ({
    comfyPage
  }) => {
    const { page } = comfyPage

    const pipeId = await addNode(page, 'PipeCreate')
    const sA = await addNode(page, 'PrimitiveString')
    const sB = await addNode(page, 'PrimitiveString')

    // Fresh Pipe starts with exactly one empty trailing slot.
    let info = await nodeInfo(page, pipeId)
    expect(info!.inputs).toHaveLength(1)
    expect(info!.inputs[0].linked).toBe(false)
    expect(info!.inputs[0].type).toBe('*')

    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))
    info = await nodeInfo(page, pipeId)
    expect(info!.inputs).toHaveLength(2)
    expect(info!.inputs[0].linked).toBe(true)
    expect(info!.inputs[0].type).toBe('STRING')
    expect(info!.inputs[1].linked).toBe(false) // fresh trailing slot

    await connect(page, sB, 0, pipeId, await slotIndex(page, pipeId, '+'))
    info = await nodeInfo(page, pipeId)
    expect(info!.inputs).toHaveLength(3)
    expect(info!.inputs.filter((i) => i.linked)).toHaveLength(2)
    // Keys are unique (slugified from upstream output name / type).
    const keys = info!.inputs.filter((i) => i.linked).map((i) => i.name)
    expect(new Set(keys).size).toBe(keys.length)
    // Manifest is persisted on properties for workflow JSON round-trip.
    expect(info!.manifest).toHaveLength(2)
    for (const [, type] of info!.manifest as [string, string][]) {
      expect(type).toBe('STRING')
    }
  })

  test('PipeOut reshapes outputs from the upstream manifest', async ({
    comfyPage
  }) => {
    const { page } = comfyPage

    const pipeId = await addNode(page, 'PipeCreate')
    const outId = await addNode(page, 'PipeOut')
    const sA = await addNode(page, 'PrimitiveString')
    const sB = await addNode(page, 'PrimitiveString')

    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, sB, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, pipeId, 0, outId, await slotIndex(page, outId, 'pipe'))

    const out = await nodeInfo(page, outId)
    expect(out!.outputs.length).toBeGreaterThanOrEqual(2)
    // Output names match the pipe manifest keys, output types STRING.
    const pipeInfo = await nodeInfo(page, pipeId)
    const manifestKeys = (pipeInfo!.manifest as [string, string][]).map(
      ([k]) => k
    )
    expect(out!.outputs.map((o) => o.name)).toEqual(manifestKeys)
    for (const o of out!.outputs) expect(o.type).toBe('STRING')
  })

  test('PipeSet adds and PipeRemove removes keys downstream', async ({
    comfyPage
  }) => {
    const { page } = comfyPage

    const pipeId = await addNode(page, 'PipeCreate')
    const setId = await addNode(page, 'PipeSet')
    const rmId = await addNode(page, 'PipeRemove')
    const outId = await addNode(page, 'PipeOut')
    const sA = await addNode(page, 'PrimitiveString')
    const sB = await addNode(page, 'PrimitiveString')

    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, pipeId, 0, setId, await slotIndex(page, setId, 'pipe'))
    await setWidget(page, setId, 'key', 'extra')
    await connect(page, sB, 0, setId, await slotIndex(page, setId, 'value'))
    await connect(page, setId, 0, rmId, await slotIndex(page, rmId, 'pipe'))

    // Remove the ORIGINAL key (auto-named from the first PrimitiveString).
    const orig = (await nodeInfo(page, pipeId))!.manifest as [string, string][]
    const origKey = orig[0][0]
    await setWidget(page, rmId, 'key', origKey)
    await connect(page, rmId, 0, outId, await slotIndex(page, outId, 'pipe'))

    const out = await nodeInfo(page, outId)
    expect(out!.outputs.map((o) => o.name)).toEqual(['extra'])
    expect(out!.outputs[0].type).toBe('STRING')
  })

  test('queuing a PipeCreate→PipeGet→PreviewAny graph unpacks the value', async ({
    comfyPage
  }) => {
    const { page } = comfyPage

    const sA = await addNode(page, 'PrimitiveString')
    await setWidget(page, sA, 'value', 'hello-from-pipe')

    const pipeId = await addNode(page, 'PipeCreate')
    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))

    const getId = await addNode(page, 'PipeGet')
    await connect(page, pipeId, 0, getId, await slotIndex(page, getId, 'pipe'))
    // Set the key to whatever the Pipe auto-named the connected slot.
    const pipeInfo = await nodeInfo(page, pipeId)
    const key = (pipeInfo!.manifest as [string, string][])[0][0]
    await setWidget(page, getId, 'key', key)

    const previewId = await addNode(page, 'PreviewAny')
    await connect(page, getId, 0, previewId, 0)

    await comfyPage.command.executeCommand('Comfy.QueuePrompt')

    const preview = await comfyPage.nodeOps.getNodeRefById(previewId)
    await expect
      .poll(async () => (await preview.getWidget(0)).getValue(), {
        timeout: 15_000
      })
      .toBe('hello-from-pipe')
  })

  test('workflow round-trip preserves slots and manifest', async ({
    comfyPage
  }) => {
    const { page } = comfyPage

    const pipeId = await addNode(page, 'PipeCreate')
    const outId = await addNode(page, 'PipeOut')
    const sA = await addNode(page, 'PrimitiveString')
    const sB = await addNode(page, 'PrimitiveString')
    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, sB, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, pipeId, 0, outId, await slotIndex(page, outId, 'pipe'))

    const before = {
      pipe: await nodeInfo(page, pipeId),
      out: await nodeInfo(page, outId)
    }

    // Serialize → clear → load via the same path workflow upload takes
    // (sets app.configuringGraph and fires afterConfigureGraph).
    const serialized = await page.evaluate(() =>
      JSON.parse(JSON.stringify(window.app!.graph.serialize()))
    )
    await page.evaluate(async (s) => {
      await window.app!.loadGraphData(s)
    }, serialized)

    const pipeNodes = await comfyPage.nodeOps.getNodeRefsByType('PipeCreate')
    const outNodes = await comfyPage.nodeOps.getNodeRefsByType('PipeOut')
    expect(pipeNodes).toHaveLength(1)
    expect(outNodes).toHaveLength(1)

    const after = {
      pipe: await nodeInfo(page, Number(pipeNodes[0].id)),
      out: await nodeInfo(page, Number(outNodes[0].id))
    }

    // Same manifest, same named outputs — no wires should have dropped.
    expect(after.pipe!.manifest).toEqual(before.pipe!.manifest)
    expect(after.out!.outputs.map((o) => o.name)).toEqual(
      before.out!.outputs.map((o) => o.name)
    )
  })

  test('nested pipes: outer Pipe Out exposes a PIPE output that an inner Pipe Out unpacks correctly', async ({
    comfyPage
  }) => {
    const { page } = comfyPage

    // Data flow under test:
    //   strA, strB --> innerPipe (PipeCreate)
    //   strM --> outerPipe (PipeCreate slot 0)
    //   innerPipe --> outerPipe (PipeCreate slot 1, type=PIPE)
    //   outerPipe --> outerOut (PipeOut)            -> outputs [STRING, PIPE]
    //   outerOut.<inner> --> innerOut (PipeOut)     -> outputs [STRING, STRING]
    //   innerOut.0 --> previewA, innerOut.1 --> previewB
    const strA = await addNode(page, 'PrimitiveString')
    await setWidget(page, strA, 'value', 'A')
    const strB = await addNode(page, 'PrimitiveString')
    await setWidget(page, strB, 'value', 'B')
    const strM = await addNode(page, 'PrimitiveString')
    await setWidget(page, strM, 'value', 'M')

    const innerPipe = await addNode(page, 'PipeCreate')
    await connect(page, strA, 0, innerPipe, await slotIndex(page, innerPipe, '+'))
    await connect(page, strB, 0, innerPipe, await slotIndex(page, innerPipe, '+'))

    const outerPipe = await addNode(page, 'PipeCreate')
    await connect(page, strM, 0, outerPipe, await slotIndex(page, outerPipe, '+'))
    await connect(page, innerPipe, 0, outerPipe, await slotIndex(page, outerPipe, '+'))

    // Outer manifest: [(modelKey, STRING), (innerKey, PIPE, [(aKey, STRING), (bKey, STRING)])]
    const outerManifest = (await nodeInfo(page, outerPipe))!.manifest as unknown[][]
    expect(outerManifest.map((e) => e[1])).toEqual(['STRING', 'PIPE'])
    const nested = outerManifest[1][2] as [string, string][]
    expect(nested.map((e) => e[1])).toEqual(['STRING', 'STRING'])

    // Outer Pipe Out: 2 outputs of types [STRING, PIPE]
    const outerOut = await addNode(page, 'PipeOut')
    await connect(page, outerPipe, 0, outerOut, await slotIndex(page, outerOut, 'pipe'))
    const outerOutInfo = await nodeInfo(page, outerOut)
    expect(outerOutInfo!.outputs.map((o) => o.type)).toEqual(['STRING', 'PIPE'])

    // Inner Pipe Out: connect outerOut's PIPE-typed output → expects 2 STRING outputs
    const pipeSlotOnOuterOut = outerOutInfo!.outputs.findIndex((o) => o.type === 'PIPE')
    const innerOut = await addNode(page, 'PipeOut')
    await connect(page, outerOut, pipeSlotOnOuterOut, innerOut, await slotIndex(page, innerOut, 'pipe'))
    const innerOutInfo = await nodeInfo(page, innerOut)
    expect(innerOutInfo!.outputs).toHaveLength(2)
    for (const o of innerOutInfo!.outputs) expect(o.type).toBe('STRING')
    // Names should match the inner manifest keys (preserved through the nest).
    expect(innerOutInfo!.outputs.map((o) => o.name)).toEqual(nested.map((e) => e[0]))

    // End-to-end: queue and verify the unpacked nested values reach previews.
    const previewA = await addNode(page, 'PreviewAny')
    const previewB = await addNode(page, 'PreviewAny')
    await connect(page, innerOut, 0, previewA, 0)
    await connect(page, innerOut, 1, previewB, 0)

    await comfyPage.command.executeCommand('Comfy.QueuePrompt')

    const pa = await comfyPage.nodeOps.getNodeRefById(previewA)
    const pb = await comfyPage.nodeOps.getNodeRefById(previewB)
    await expect
      .poll(async () => (await pa.getWidget(0)).getValue(), { timeout: 15_000 })
      .toBe('A')
    await expect
      .poll(async () => (await pb.getWidget(0)).getValue(), { timeout: 15_000 })
      .toBe('B')
  })

  test('disconnecting a PipeCreate input prunes the slot and updates manifest downstream', async ({
    comfyPage
  }) => {
    const { page } = comfyPage

    const sA = await addNode(page, 'PrimitiveString')
    const sB = await addNode(page, 'PrimitiveString')
    const pipeId = await addNode(page, 'PipeCreate')
    const outId = await addNode(page, 'PipeOut')

    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, sB, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, pipeId, 0, outId, await slotIndex(page, outId, 'pipe'))

    expect((await nodeInfo(page, outId))!.outputs).toHaveLength(2)

    // Disconnect sA from the pipe.
    await page.evaluate(({ pipeId }) => {
      const n = window.app!.graph.getNodeById(pipeId)!
      const idx = n.inputs.findIndex((i) => !i.widget && i.link != null)
      n.disconnectInput(idx)
    }, { pipeId })

    const after = await nodeInfo(page, pipeId)
    // One linked + one trailing "+" remain; the disconnected slot is pruned.
    expect(after!.inputs.filter((i) => i.linked)).toHaveLength(1)
    expect(after!.inputs).toHaveLength(2)
    expect((after!.manifest as unknown[]).length).toBe(1)
    // Downstream PipeOut reshaped to one output.
    expect((await nodeInfo(page, outId))!.outputs).toHaveLength(1)
  })

  test('PipeOut keeps surviving downstream wires when an upstream key is removed', async ({
    comfyPage
  }) => {
    const { page } = comfyPage

    const sA = await addNode(page, 'PrimitiveString')
    const sB = await addNode(page, 'PrimitiveString')
    const pipeId = await addNode(page, 'PipeCreate')
    const outId = await addNode(page, 'PipeOut')
    const sinkA = await addNode(page, 'PreviewAny')
    const sinkB = await addNode(page, 'PreviewAny')

    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, sB, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, pipeId, 0, outId, await slotIndex(page, outId, 'pipe'))
    await connect(page, outId, 0, sinkA, 0)
    await connect(page, outId, 1, sinkB, 0)

    const before = await nodeInfo(page, outId)
    expect(before!.outputs.map((o) => o.linkCount)).toEqual([1, 1])
    const survivorKey = before!.outputs[1].name

    // Disconnect the FIRST upstream key on PipeCreate.
    await page.evaluate(({ pipeId }) => {
      const n = window.app!.graph.getNodeById(pipeId)!
      const idx = n.inputs.findIndex((i) => !i.widget && i.link != null)
      n.disconnectInput(idx)
    }, { pipeId })

    const after = await nodeInfo(page, outId)
    // Only the surviving key remains, and its downstream wire is preserved.
    expect(after!.outputs).toHaveLength(1)
    expect(after!.outputs[0].name).toBe(survivorKey)
    expect(after!.outputs[0].linkCount).toBe(1)
    // sinkA's input link is dropped (its key vanished).
    const sinkAInfo = await page.evaluate((id) => {
      const n = window.app!.graph.getNodeById(id)!
      return n.inputs[0].link
    }, sinkA)
    expect(sinkAInfo).toBeNull()
  })

  test('manifest passes through a Reroute node', async ({ comfyPage }) => {
    const { page } = comfyPage

    const sA = await addNode(page, 'PrimitiveString')
    const pipeId = await addNode(page, 'PipeCreate')
    const reroute = await addNode(page, 'Reroute')
    const outId = await addNode(page, 'PipeOut')

    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, pipeId, 0, reroute, 0)
    await connect(page, reroute, 0, outId, await slotIndex(page, outId, 'pipe'))

    const out = await nodeInfo(page, outId)
    expect(out!.outputs).toHaveLength(1)
    expect(out!.outputs[0].type).toBe('STRING')
  })

  test('changing the inner pipe propagates to the outer nested manifest', async ({
    comfyPage
  }) => {
    const { page } = comfyPage

    const sA = await addNode(page, 'PrimitiveString')
    const sB = await addNode(page, 'PrimitiveString')
    const inner = await addNode(page, 'PipeCreate')
    const outer = await addNode(page, 'PipeCreate')
    const outerOut = await addNode(page, 'PipeOut')
    const innerOut = await addNode(page, 'PipeOut')

    // inner:{a} -> outer -> outerOut[PIPE] -> innerOut
    await connect(page, sA, 0, inner, await slotIndex(page, inner, '+'))
    await connect(page, inner, 0, outer, await slotIndex(page, outer, '+'))
    await connect(page, outer, 0, outerOut, await slotIndex(page, outerOut, 'pipe'))
    const pipeSlot = (await nodeInfo(page, outerOut))!.outputs.findIndex(
      (o) => o.type === 'PIPE'
    )
    await connect(page, outerOut, pipeSlot, innerOut, await slotIndex(page, innerOut, 'pipe'))

    expect((await nodeInfo(page, innerOut))!.outputs).toHaveLength(1)

    // Now extend INNER with a second key — the chain should propagate so
    // innerOut grows a second output.
    await connect(page, sB, 0, inner, await slotIndex(page, inner, '+'))

    const refreshed = await nodeInfo(page, innerOut)
    expect(refreshed!.outputs).toHaveLength(2)
    for (const o of refreshed!.outputs) expect(o.type).toBe('STRING')
  })

  test('PipeMerge unions manifests with the chosen collision policy', async ({
    comfyPage
  }) => {
    const { page } = comfyPage

    // left:  {string: STRING}             — one PrimitiveString
    // right: {string: STRING, int: INT}   — colliding "string" + a unique "int"
    // (key names come from slugify(upstream output name))
    const sL = await addNode(page, 'PrimitiveString')
    const sR = await addNode(page, 'PrimitiveString')
    const iR = await addNode(page, 'PrimitiveInt')
    const left = await addNode(page, 'PipeCreate')
    const right = await addNode(page, 'PipeCreate')
    const merge = await addNode(page, 'PipeMerge')
    const out = await addNode(page, 'PipeOut')

    await connect(page, sL, 0, left, await slotIndex(page, left, '+'))
    await connect(page, sR, 0, right, await slotIndex(page, right, '+'))
    await connect(page, iR, 0, right, await slotIndex(page, right, '+'))

    const lKeys = ((await nodeInfo(page, left))!.manifest as [string, string][])
      .map(([k]) => k)
    const rKeys = ((await nodeInfo(page, right))!.manifest as [string, string][])
      .map(([k]) => k)
    expect(lKeys).toHaveLength(1)
    expect(rKeys).toHaveLength(2)
    // Collision: left's only key is also right's first key.
    expect(rKeys).toContain(lKeys[0])

    await connect(page, left, 0, merge, await slotIndex(page, merge, 'a'))
    await connect(page, right, 0, merge, await slotIndex(page, merge, 'b'))
    await connect(page, merge, 0, out, await slotIndex(page, out, 'pipe'))

    // Union (right_wins by default) — keys are the union, in left-then-right
    // order.
    const expected = [...new Set([...lKeys, ...rKeys])]
    let outInfo = await nodeInfo(page, out)
    expect(outInfo!.outputs.map((o) => o.name)).toEqual(expected)

    // Flip the collision policy and confirm the downstream re-walk fires
    // (surface keys don't change here since both colliding entries are STRING).
    await setWidget(page, merge, 'collision', 'left_wins')
    outInfo = await nodeInfo(page, out)
    expect(outInfo!.outputs.map((o) => o.name)).toEqual(expected)
  })

  test('key widget on PipeRemove/PipeGet becomes a dropdown of upstream keys', async ({
    comfyPage
  }) => {
    const { page } = comfyPage

    const sA = await addNode(page, 'PrimitiveString')
    const sB = await addNode(page, 'PrimitiveString')
    const pipeId = await addNode(page, 'PipeCreate')
    const rmId = await addNode(page, 'PipeRemove')
    const getId = await addNode(page, 'PipeGet')

    await connect(page, sA, 0, pipeId, await slotIndex(page, pipeId, '+'))
    await connect(page, sB, 0, pipeId, await slotIndex(page, pipeId, '+'))
    const keys = ((await nodeInfo(page, pipeId))!.manifest as [string, string][])
      .map(([k]) => k)

    await connect(page, pipeId, 0, rmId, await slotIndex(page, rmId, 'pipe'))
    await connect(page, pipeId, 0, getId, await slotIndex(page, getId, 'pipe'))

    const widgetInfo = (id: number) => page.evaluate((id) => {
      const n = window.app!.graph.getNodeById(id)!
      const w = n.widgets!.find((w) => w.name === 'key')!
      return { type: w.type, values: (w.options as any)?.values, value: w.value }
    }, id)

    const rm = await widgetInfo(rmId)
    expect(rm.type).toBe('combo')
    expect(rm.values).toEqual(keys)
    expect(keys).toContain(rm.value)

    const get = await widgetInfo(getId)
    expect(get.type).toBe('combo')
    expect(get.values).toEqual(keys)
  })
})
