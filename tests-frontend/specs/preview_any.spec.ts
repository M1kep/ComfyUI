import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '@e2e/fixtures/ComfyPage'

test.describe('PreviewAny node', () => {
  test('renders backend output in the frontend widget', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('backend/preview_any')

    const preview = await comfyPage.nodeOps.getNodeRefById(2)
    await expect
      .poll(async () => (await preview.getWidget(0)).getValue())
      .toBe('')

    await comfyPage.command.executeCommand('Comfy.QueuePrompt')

    await expect
      .poll(async () => (await preview.getWidget(0)).getValue())
      .toBe('hello from backend test')
  })
})
