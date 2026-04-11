# Frontend (browser) tests

Browser-level workflow tests for nodes that have both a Python backend
implementation in this repo **and** frontend JavaScript behaviour (widgets,
previews, etc.) provided by `comfyui-frontend-package`.

These tests drive a real ComfyUI server + frontend in headless Chromium via
Playwright, reusing the [`ComfyUI_frontend`](https://github.com/Comfy-Org/ComfyUI_frontend)
repo's `ComfyPage` fixture so you don't have to reinvent canvas/node helpers.

## Layout

```
tests-frontend/
  setup.sh              one-time setup (clone frontend, install deps, devtools)
  run-tests.sh          boot server + run Playwright against specs/
  playwright.config.ts  Playwright config (copied into the frontend checkout)
  specs/                *.spec.ts files for backend nodes
  assets/               workflow .json files loadable as `backend/<name>`
  ComfyUI_frontend/     git-ignored checkout created by setup.sh
```

## Prerequisites

- Python deps installed (`pip install -r requirements.txt` plus a torch build)
- Node.js 24 with `pnpm` (`corepack enable`)

## Setup

```bash
./tests-frontend/setup.sh
```

This clones `ComfyUI_frontend` into `tests-frontend/ComfyUI_frontend/`, runs
`pnpm install`, installs the Playwright Chromium browser, copies the
`ComfyUI_devtools` test nodes into `custom_nodes/`, and links `assets/` into
the checkout so the frontend's `@e2e/*` fixtures resolve.

Override the checkout location, repo, or ref with `FRONTEND_DIR`,
`FRONTEND_REPO`, or `FRONTEND_REF`.

## Running

```bash
./tests-frontend/run-tests.sh                 # all specs in tests-frontend/specs
./tests-frontend/run-tests.sh preview_any     # filter by file/name
./tests-frontend/run-tests.sh --ui            # Playwright UI mode
SKIP_SERVER=1 ./tests-frontend/run-tests.sh   # use an already-running server
```

The script starts `python main.py --cpu --multi-user`, waits for it to come up,
then runs `playwright test --project=chromium browser_tests/tests/backend`
inside the frontend checkout. Server logs go to `tests-frontend/server.log`.

## Writing a test for a new node

1. Export a minimal workflow that exercises the node from the ComfyUI UI
   (Menu → *Save*) and drop the `.json` into `tests-frontend/assets/`.
2. Add a spec in `tests-frontend/specs/`:

   ```ts
   import { expect } from '@playwright/test'
   import { comfyPageFixture as test } from '@e2e/fixtures/ComfyPage'

   test('my node does the thing', async ({ comfyPage }) => {
     await comfyPage.workflow.loadWorkflow('backend/my_workflow')
     await comfyPage.command.executeCommand('Comfy.QueuePrompt')

     const node = await comfyPage.nodeOps.getNodeRefById(2)
     await expect
       .poll(async () => (await node.getWidget(0)).getValue())
       .toBe('expected value')
   })
   ```

   The `@e2e/*` import alias and helpers (`comfyPage.nodeOps`,
   `comfyPage.workflow`, `comfyPage.command`, …) come from the frontend repo;
   see its [`browser_tests/README.md`](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/browser_tests/README.md)
   for the full helper catalogue.

3. Run `./tests-frontend/run-tests.sh <name>`.

Avoid screenshot assertions here — they are platform-sensitive and the frontend
repo manages its own Linux baselines. Prefer functional assertions on widget
values, node state, or `/history` results.
