# Web graph rendering: observation, diagnosis, and presentation options

This note records how to inspect MLTorch's Model Explorer output as a visual
artifact, what was observed for `mobilenetv2_050`, and which changes to the
exported graph materially improve its presentation. It is about whether the
graph is readable and visually useful. The existing correctness, lifecycle,
and session-validation tests answer different questions.

## Rendering path

The assembled application lives in `webapp-dist/`. `web/app/renderer.js`
creates a `model-explorer-visualizer` custom element, assigns the
session's `graphCollections`, and waits for `modelGraphProcessed`. The element
renders its graph in a WebGL canvas inside its shadow tree. Ordinary DOM
assertions can establish that the element exists and is visible, but cannot say
that its fitted graph is legible or attractive.

The graph seen by Model Explorer is largely determined before it reaches the
browser:

- `lib/model_explorer_export/me_source.ml` projects the exported-program graph.
- `lib/model_explorer_export/me_build.ml` projects the native graphs.
- Both exporters turn graph inputs, including captured parameters, into
  ordinary boundary nodes.
- Constant boundary nodes currently have an empty namespace. They therefore
  sit at the root rather than inside the module or structural group which uses
  them.
- Each consumer receives a normal incoming edge from the constant node.

Model Explorer performs the final layout, but the exporter determines the
topology and hierarchy on which that layout operates.

## How to observe the real UI

Build the browser application when its inputs have changed:

```sh
make webapp.build
```

Serve the assembled output using the same server as the Playwright gate:

```sh
cd web
npm run serve:webapp
```

The application is then available at `http://127.0.0.1:8124/index.html`.
Use the repository's downloaded browser so local observation and CI use the
same Chromium build:

```sh
cd web
PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright-browsers" \
  npx playwright test -c playwright.webapp.config.ts
```

For visual investigation, use a fixed viewport and device scale, wait for the
application's real completion condition, and screenshot the current visualizer
slot rather than the whole application. A useful diagnostic viewport is
1600 by 1000 with `deviceScaleFactor: 1`.

```ts
const page = await browser.newPage({
  viewport: { width: 1600, height: 1000 },
  deviceScaleFactor: 1,
});

await page.goto('http://127.0.0.1:8124/index.html');
await page.waitForFunction(
  () => document.querySelector('#status')?.textContent === 'Model loaded',
  null,
  { timeout: 120_000 },
);
await page.evaluate(() => document.fonts.ready);

await page.locator(
  '#visualizer .visualizer-slot--current',
).screenshot({ path: '/tmp/mltorch-source.png' });
```

When changing views, do not use a pre-existing status string as the completion
signal. `Model loaded` or `View changed` can describe the preceding render.
Record the current slot or custom-element identity before selecting the view,
then wait until a different element becomes current and the replacement has
settled. The mutation-history helpers in `web/test/webapp.spec.ts` already
encode this lifecycle accurately.

During an investigation, collect more than screenshots:

- page exceptions and console errors;
- failed asset and worker responses;
- current view id, graph id, node count, and layer count;
- a focused screenshot of the visualizer;
- a full-page screenshot when shell sizing or side panels may be involved;
- a Playwright trace for a render that intermittently differs.

The focused screenshot is the main artifact for judging graph composition.
The full page explains whether the host application gave Model Explorer a
reasonable canvas. In the current application it does: the visualizer is about
70 percent of the viewport height and fills the available width. The excessive
graph width comes from the graph representation, not from a narrow host
container.

## What to judge visually

Review the initial fitted graph before zooming or expanding groups. It should
communicate the model's main flow immediately. Check:

1. Node labels at the initial fit are readable.
2. The major path has a clear direction and uses the canvas well.
3. Root-level fan-out does not dominate the composition.
4. Long dependency edges do not cross the entire graph unnecessarily.
5. Collapsed module or structural groups form a useful architectural summary.
6. Expanding one group reveals its local computation without introducing a
   model-wide row of unrelated nodes.
7. Inputs and outputs are visually distinct and easy to locate.
8. Comparison and flow views remain balanced after the same representation
   change.

Screenshot regression tests are useful after a presentation has been chosen,
but a pixel match is not itself an aesthetic judgment. Establish the baseline
through human review, then use Playwright to detect unintended changes. Keep
Chromium, the Linux image, fonts, viewport, device scale, color scheme, and
model fixture fixed. Upload expected, actual, and diff images on failure.

## MobileNet observation

The assembled web application was rendered headlessly at a 1600-pixel-wide
viewport using the repository's pinned Chromium. The Exported Program view was
an extremely thin horizontal line. Most labels were unreadable at initial fit.
This reproduces the reported problem.

The exported session explains the shape:

| Graph | Nodes | Constant nodes | Root/source nodes | Edges |
|---|---:|---:|---:|---:|
| `pt2/root` | 468 | 314 | 315 | 425 |
| `g/native/000` | 731 | 314 | 315 | 688 |
| `g/native/001` | 208 | 106 | 107 | 217 |
| `g/symbolic/000` | 208 | 106 | 107 | 217 |
| `g/native4d/000` | 208 | 106 | 107 | 217 |
| `g/kernel/000` | 208 | 106 | 107 | 217 |

For `pt2/root`, 314 independent constant nodes have no incoming edges and an
empty namespace. A layered layout places this large source set on the same
rank. Fitting that rank into the canvas determines the zoom level for the whole
graph, reducing the actual compute path to nearly invisible marks.

## Representation experiments

Variants were built from the same exported `pt2/root` graph and rendered by the
same Model Explorer bundle. Only the representation supplied to the custom
element changed.

### Reordering nodes

Reversing the node array and moving constants after operation nodes made no
material difference to graph dimensions or readability. Array order may break
ties within a rank, but it does not change the fact that hundreds of constants
occupy one source rank. Sorting is not the primary remedy.

### Smaller layout spacing

Setting very small `nodeSep`, `rankSep`, and `edgeSep` values left the graph as
an unreadable horizontal line. Spacing can tune a good topology; it cannot make
a rank of 314 nodes fit legibly.

### `nodeLabelsToHide = ["constant"]`

Using Model Explorer's label-hiding facility produced a blank or unusable view
in this graph. Hiding producer nodes is not equivalent to projecting constants
out while retaining their consumers. Do not use this as compact mode without a
separate upstream fix and a verified renderer contract.

### One root `parameters` group

Putting every constant in one `parameters` namespace made the graph compact,
but the collapsed group still fed operations throughout the model. The result
contained long fan-out arcs spanning much of the diagram. It was more legible
than the baseline but visually noisy and did not express parameter ownership.

### Constants as consumer metadata

Removing constant nodes and their edges produced the cleanest high-level
layout: a centered vertical compute path from input to output. Constant name,
shape, dtype, and transformation information can be retained in consumer
attributes or `inputsMetadata`.

This representation changes topology. Constant node ids no longer exist, so
comparison mappings, selection, node data sets, diagnostics, and detail links
must all define what happens to those ids. It is a useful possible mode, but it
is not the safest first presentation change.

### Constants in consumer namespaces

Assigning each constant to the namespace of its consumer gave the best balance.
At initial fit the graph became a clear vertical architectural summary:

```text
input
  |
conv_stem
  |
bn1
  |
blocks
  |
conv_head
  |
bn2
  |
global_pool
  |
classifier
  |
output
```

The constant nodes, their ids, metadata, and dependency edges remained in the
graph. Model Explorer's collapsed hierarchy absorbed them into the module which
uses them. Expanding a module can still reveal its parameter dependencies.

For a constant with multiple consumers, use the longest common namespace of
all consumers. If the consumers have no non-empty common namespace, place it in
a dedicated root `parameters` namespace or leave it at the root. This rule is
deterministic and reflects ownership better than choosing the first consumer.

## Suggested user-selectable presentation

Add a presentation control near the existing View and Comparison controls:

```text
Constants:  Grouped  |  Explicit
```

`Grouped` should be the default because it produces a useful initial overview.
It assigns constants to their consumers' longest common namespace. `Explicit`
preserves today's root-level dependency graph for users investigating individual
weights or exporter details.

This first version can be presentation-only and instant:

- retain the full session in memory;
- derive graph collections with adjusted constant namespaces for `Grouped`;
- preserve every node id, edge, attribute, metadata item, and mapping;
- construct a new visualizer candidate through the existing renderer lifecycle;
- do not post another worker request;
- store the choice in the URL, for example `constants=grouped` or
  `constants=explicit`;
- restore the choice across view changes, comparisons, history navigation, and
  reloads.

The renderer should not identify constants solely from user-visible label text
forever. The present exporters control stable `const:` boundary ids, but a
dedicated boundary-kind attribute in the session would make the contract
explicit and would cover future id schemes. Until that field exists, keep the
classification in one adapter function and test it against source and native
graphs.

A later `Metadata only` choice could remove constant nodes and attach their
information to consuming operations. Treat that as an export/schema feature,
not merely a namespace transformation, because it changes topology and every
id-addressed facility must specify its behavior.

## Validation for the selectable mode

Use a small semantic test and a reviewed visual suite.

The semantic Playwright test should prove that switching Grouped/Explicit:

- sends no application-worker message;
- replaces the visualizer through the existing safe candidate lifecycle;
- keeps the same graph and view selected;
- preserves node counts and known constant node ids;
- updates and restores the URL value;
- survives back/forward navigation;
- leaves no page or renderer errors.

The visual suite should capture at least:

- MobileNet Exported Program, Grouped and Explicit;
- Initial Native and Canonical Native in Grouped mode;
- one expanded module in Grouped mode;
- one comparison view;
- the flow view, to prove the presentation adapter does not transform a graph
  with no constant boundaries.

The primary acceptance criterion for Grouped MobileNet is qualitative and
reviewable: the initial fitted screenshot shows the architectural path with
readable group labels and without a model-wide constant row. A simple automated
guard can supplement the screenshot by checking that the exported session has
no large set of constant nodes left in the root namespace after the Grouped
projection. That structural proxy is stable and directly targets the cause of
the bad composition.

## Recommended sequence

1. Implement constant classification and longest-common-consumer namespace as
   a pure, unit-tested presentation transformation.
2. Render the transformed source and native graphs against the real pinned
   Model Explorer element.
3. Review fixed-viewport screenshots before choosing the final default.
4. Add the Grouped/Explicit control, URL state, and history behavior.
5. Add the semantic Playwright assertions and a small reviewed screenshot set.
6. Consider metadata-only constants separately after comparison and detail
   semantics have been designed.
