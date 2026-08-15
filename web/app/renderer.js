export class Renderer {
  // Serializes installs: a candidate must never be torn out of the mount while it is
  // still processing, or Angular throws NG0953 ("emit for destroyed OutputRef") from
  // its still-running background work. Each install waits for the previous one to
  // settle (success or timeout) before touching the DOM.
  #queue = Promise.resolve();
  constructor({ mount, timeout = 90_000 }) { this.mount = mount; this.timeout = timeout; this.current = null; }
  async install(renderText) {
    const state = JSON.parse(renderText);
    const view = state.views?.find((v) => v.id === state.defaultView);
    const graphId = view?.graph || state.graphCollections?.[0]?.graphs?.[0]?.id;
    if (!graphId) throw new Error('session has no renderable graph');
    const candidate = document.createElement('model-explorer-visualizer');
    candidate.graphCollections = state.graphCollections;
    // `defaultGraphId` is not a real VisualizerConfig field (it appears nowhere in the
    // library's types or compiled bundle) -- kept here in case a future version adds it,
    // but nothing depends on it doing anything. The library auto-selects some graph on
    // its own; selectNode() below is what actually navigates to the one we want.
    candidate.config = { defaultGraphId: graphId };
    const modelName = state.model?.name ?? '(unknown)';
    const nodeCounts = Object.fromEntries(
      (state.graphCollections ?? []).flatMap((c) => c.graphs ?? []).map((g) => [g.id, g.nodes?.length ?? 0]),
    );
    const collectionLabel = state.graphCollections?.[0]?.label;
    const targetFirstNodeId = state.graphCollections?.[0]?.graphs?.find((g) => g.id === graphId)?.nodes?.[0]?.id;
    console.debug(`[renderer] model=${modelName} queued for defaultGraphId=${graphId} (${nodeCounts[graphId]} nodes); graphs=${JSON.stringify(nodeCounts)}`);
    const turn = this.#queue.catch(() => {}).then(async () => {
      const previous = this.current;
      this.mount.replaceChildren(candidate);
      // The timeout/heartbeat clock starts HERE, once the candidate is actually live in
      // the DOM and has a real chance to process -- not at install() call time, or time
      // spent queued behind a previous, still-settling install would eat its own budget.
      const started = performance.now();
      const elapsed = () => Math.round(performance.now() - started);
      const seen = new Set();
      console.debug(`[renderer] model=${modelName} appended to mount, waiting for graph ${graphId}`);
      const completed = new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          console.warn(`[renderer] model=${modelName} timed out after ${elapsed()}ms waiting for graph ${graphId} (${nodeCounts[graphId]} nodes); received=${JSON.stringify([...seen])} of ${JSON.stringify(Object.keys(nodeCounts))}`);
          clearInterval(heartbeat);
          const error = new Error('renderer timed out');
          error.stillProcessing = true;
          reject(error);
        }, this.timeout);
        const heartbeat = setInterval(() => {
          console.debug(`[renderer] model=${modelName} still waiting at ${elapsed()}ms; received so far=${JSON.stringify([...seen])}`);
        }, 5000);
        let selected = false;
        candidate.addEventListener('modelGraphProcessed', (event) => {
          const seenId = event.detail?.modelGraph?.id;
          seen.add(seenId);
          console.debug(`[renderer] model=${modelName} modelGraphProcessed id=${seenId} (${nodeCounts[seenId]} nodes) wanted=${graphId} at ${elapsed()}ms`);
          if (seenId === graphId) { clearTimeout(timer); clearInterval(heartbeat); resolve(); return; }
          // The library settled on some OTHER graph as its own default; navigate to the
          // one we actually asked for, the same way a manual click in the UI does (and the
          // only mechanism confirmed to work -- see web/src/session.js's selectNode use).
          if (!selected && targetFirstNodeId) {
            selected = true;
            console.debug(`[renderer] model=${modelName} selecting target graph ${graphId} via node ${targetFirstNodeId} at ${elapsed()}ms`);
            try {
              candidate.selectNode(targetFirstNodeId, graphId, collectionLabel);
            } catch (error) {
              clearTimeout(timer);
              clearInterval(heartbeat);
              reject(error);
            }
          }
        });
      });
      try {
        await completed;
        for (const set of state.nodeDataSets || []) {
          if (set.graph !== graphId) continue;
          const results = Object.fromEntries((set.results || []).map(([nodeId, value]) => [nodeId, { value: value.value }]));
          candidate.addNodeDataProviderData(set.name, { results, gradient: [{ stop: 0, bgColor: '#e8f5e9' }, { stop: 1, bgColor: '#ffcdd2' }] });
        }
        this.current = candidate;
      } catch (error) {
        // A candidate that timed out is presumed still processing in the background:
        // tearing it out of the DOM here would destroy a live component mid-flight,
        // which is exactly what throws Angular's NG0953. Leave it mounted (broken,
        // but not destroyed) rather than restoring `previous` over it.
        if (!error?.stillProcessing) this.mount.replaceChildren(...(previous ? [previous] : []));
        else this.current = null;
        throw error;
      }
    });
    this.#queue = turn.catch(() => {});
    await turn;
  }
}
