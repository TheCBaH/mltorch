/* §5.8's gate: the session `native_graph visualize` ACTUALLY produces, rendered
 * by the pinned element.
 *
 * Distinct from the 0A spike next door, and the difference is the point. That
 * one drives a hand-authored fixture to settle questions about the RENDERER.
 * This one drives real output to settle a question about US: whether a document
 * `Me_session.validate` accepts is a document the renderer accepts. Our
 * validator is our own reading of the schema, and a projection can satisfy it
 * and still be rejected — a graph id the renderer cannot address, a namespace it
 * splits somewhere we did not expect, a node data set keyed by an id no node
 * carries.
 *
 * It records rather than asserts, exactly as the spike does, except for the one
 * thing that is a defect on our side however the renderer behaves: a graph that
 * never finishes processing.
 */

const MOUNT = document.getElementById('mount');
const result = { graphs: [], errors: [] };
window.__gate = result;

/* [graphId], when given, is load-bearing rather than cosmetic: an element
 * that already processed one graph and is then asked to process a second
 * still carries whatever `modelGraphProcessed` events the FIRST graph's own
 * processing (layout, node reveal) has queued but not yet delivered, and an
 * unfiltered `{once: true}` listener registered after the second request can
 * resolve on one of those instead of on the graph actually asked for --
 * `selectGraphInPane` also no-ops (fires nothing) when asked to reselect the
 * graph already active, which an unfiltered wait cannot tell apart from a
 * real, still-pending switch. The event's `detail.modelGraph.id` is Angular
 * Elements' own bridging of the `modelGraphProcessed` `@Output()`, so this is
 * exactly what fired, not a guess about it. */
function awaitProcessed(el, ms, graphId) {
  return new Promise((resolve) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        el.removeEventListener('modelGraphProcessed', onEvent);
        resolve({ fired: false, timedOut: true });
      }
    }, ms);
    const onEvent = (e) => {
      if (graphId != null && e.detail?.modelGraph?.id !== graphId) return;
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        el.removeEventListener('modelGraphProcessed', onEvent);
        resolve({ fired: true, timedOut: false });
      }
    };
    el.addEventListener('modelGraphProcessed', onEvent);
  });
}

async function run() {
  const r = await fetch('session.json');
  if (!r.ok) throw new Error(`session.json: ${r.status}`);
  const session = await r.json();

  result.model = session.model;
  result.defaultView = session.defaultView;
  result.views = session.views.map((v) => v.id);
  result.capabilities = session.capabilities.map(
    (c) => `${c.key}=${c.status.state}`);

  const collections = session.graphCollections;
  const label = collections[0].label;

  /* Every graph the session offers, entered in turn through the SAME element.
   * Entering them one at a time rather than trusting the first is what makes
   * this a check of the whole document: the source graph, the initial Native
   * graph and the canonical one are three different projections, and only one
   * of them is the default view. */
  for (const g of collections[0].graphs) {
    const el = document.createElement('model-explorer-visualizer');
    el.graphCollections = collections;
    el.config = { defaultGraphId: g.id };
    const processed = awaitProcessed(el, 60_000);
    MOUNT.replaceChildren(el);
    const outcome = await processed;
    result.graphs.push({ id: g.id, nodes: g.nodes.length, ...outcome });

    /* Node data is graph-addressed in the session and graph-addressed here, so
     * a set whose `graph` field named a graph the collection does not hold
     * would surface as a throw rather than as a silently empty overlay. */
    for (const d of session.nodeDataSets) {
      if (d.graph !== g.id) continue;
      const results = {};
      for (const entry of d.results) {
        results[entry.nodeId] = { value: entry.value.value };
      }
      try {
        /* NOT keyed by graph id. The element wraps this in `{[graph.id]: n}`
         * itself against the graph in the pane, so a caller that keys it too
         * lands one level too deep and the run reads `.results` off a graph
         * id. 0A recorded that call as accepted because nothing threw at the
         * time; here it does, which is the difference between a spike that
         * observes an API and a gate that uses it. */
        el.addNodeDataProviderData(d.name, {
          results,
          gradient: [
            { stop: 0, bgColor: '#e8f5e9' },
            { stop: 1, bgColor: '#ffcdd2' },
          ],
        });
        result.nodeData = { name: d.name, graph: g.id, accepted: true,
                            count: Object.keys(results).length };
      } catch (e) {
        result.nodeData = { name: d.name, graph: g.id, error: String(e) };
      }
    }
  }

  /* The DETAIL path, which the 0A spike proved possible with a hand-authored
   * fixture and this drives with a real one: a graph nobody has seen arrives
   * late, is linked from a node already on screen, and is entered.
   *
   * This is the browser half of the on-demand design, and it is the only place
   * the merge is checked against the element rather than against our own
   * validator. */
  const dr = await fetch('detail-delta.json');
  if (dr.ok) {
    const delta = await dr.json();
    const merged = structuredClone(collections);
    const coll = merged.find((c) => c.label === delta.collection);
    coll.graphs.push(delta.graph);
    /* The parent link and the graph arrive TOGETHER -- a link to a graph that
     * is not installed is a dangling reference, and a graph nobody links to is
     * unreachable. */
    const parentId = delta.view.id.split('/').slice(1, -2).join('/');
    const parent = coll.graphs.find((g) => g.id === parentId);
    const node = parent && parent.nodes.find(
      (n) => (n.subgraphIds || []).length === 0 && n.id === 'v' + delta.view.id.split('/').at(-1).slice(1));
    if (node) node.subgraphIds = [delta.graph.id];

    const el = document.createElement('model-explorer-visualizer');
    el.graphCollections = merged;
    el.config = { defaultGraphId: parentId };
    const processed = awaitProcessed(el, 60_000);
    MOUNT.replaceChildren(el);
    const outcome = await processed;
    let selectedInParent = false;
    try {
      el.selectNode(node ? node.id : 'v0', parentId, delta.collection);
      selectedInParent = true;
    } catch (e) {
      result.errors.push(String(e));
    }
    /* `selectNode` selects a node WITHIN a graph; it does not itself activate
     * that node's linked subgraph, so the call above never entered
     * `delta.graph` -- it only found the link on screen in `parentId`. A real
     * click on a node with `subgraphIds` activates the subgraph through the
     * same primitive `selectNode` uses (`selectGraphInPane`), just given the
     * subgraph's own id instead of the parent's, which is what this second
     * call does. The observable proof is `modelGraphProcessed` firing FOR
     * `delta.graph.id` specifically -- not merely firing again, which it does
     * on unrelated things too (a node reveal from the first call can queue
     * one of its own), and would still read as "entered" for a second call
     * that pointlessly reselected the graph already active. */
    const enteredProcessed = awaitProcessed(el, 60_000, delta.graph.id);
    let entered = false;
    try {
      el.selectNode(delta.graph.nodes[0].id, delta.graph.id, delta.collection);
      entered = true;
    } catch (e) {
      result.errors.push(String(e));
    }
    const enteredOutcome = await enteredProcessed;
    result.detail = {
      graph: delta.graph.id,
      nodes: delta.graph.nodes.length,
      parent: parentId,
      linkedFrom: node ? node.id : null,
      selectedInParent,
      entered,
      enteredFired: enteredOutcome.fired,
      ...outcome,
    };
  }

  /* The `tasksData` evidence.
   *
   * `Me_fusion`'s overlay rides on the kernel graph's own `tasksData` rather
   * than on a comparison (`me_export.ml:380-392`), and the npm build this
   * repository used to vendor -- `ai-edge-model-explorer-visualizer@0.1.2` --
   * did not read that field ANYWHERE: zero occurrences in its `src` and its
   * `dist` alike, so the overlay was emitted and silently dropped. The bundle
   * is now built from the submodule that also pins our OCaml schema, which is
   * the whole point of building it.
   *
   * Two halves, because either alone is weak. The count over the loaded bundle
   * is what distinguishes the two runtimes at all -- with a control, so a zero
   * cannot be a fetch or minifier artefact. The graph list is what makes it a
   * statement about OUR document: `SplitPane.ngOnInit` now calls
   * `addEdgeOverlayData` for this payload, and a structurally wrong one throws
   * there, which under 0.1.2 was unreachable. */
  const bundle = await (await fetch('vendor/main_browser.js')).text();
  const occurrences = (needle) => bundle.split(needle).length - 1;
  result.bundle = {
    tasksData: occurrences('tasksData'),
    control: occurrences('edgeOverlaysDataListLeftPane'),
  };
  result.tasksDataGraphs = collections[0].graphs
    .filter((g) => g.tasksData != null)
    .map((g) => g.id);

  /* What the renderer does with `groupNodeAttributes` is its business; what
   * this records is that a document carrying them still processes, since they
   * are the one part of our output keyed by namespace rather than by id. */
  const canonical = collections[0].graphs.find(
    (g) => g.groupNodeAttributes !== undefined && g.groupNodeAttributes !== null);
  result.groupAttributes = canonical
    ? { graph: canonical.id, groups: Object.keys(canonical.groupNodeAttributes).length }
    : null;
  result.collectionLabel = label;
  result.done = true;
}

run().catch((e) => {
  result.error = String(e && e.stack ? e.stack : e);
  result.done = true;
});
