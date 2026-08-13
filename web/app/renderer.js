export class Renderer {
  constructor({ mount, timeout = 30_000 }) { this.mount = mount; this.timeout = timeout; this.current = null; }
  async install(renderText) {
    const state = JSON.parse(renderText);
    const view = state.views?.find((v) => v.id === state.defaultView);
    const graphId = view?.graph || state.graphCollections?.[0]?.graphs?.[0]?.id;
    if (!graphId) throw new Error('session has no renderable graph');
    const candidate = document.createElement('model-explorer-visualizer');
    candidate.graphCollections = state.graphCollections;
    candidate.config = { defaultGraphId: graphId };
    const completed = new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('renderer timed out')), this.timeout);
      candidate.addEventListener('modelGraphProcessed', (event) => {
        if (event.detail?.modelGraph?.id === graphId) { clearTimeout(timer); resolve(); }
      });
    });
    const previous = this.current;
    this.mount.replaceChildren(candidate);
    try {
      await completed;
      for (const set of state.nodeDataSets || []) {
        if (set.graph !== graphId) continue;
        const results = Object.fromEntries((set.results || []).map(([nodeId, value]) => [nodeId, { value: value.value }]));
        candidate.addNodeDataProviderData(set.name, { results, gradient: [{ stop: 0, bgColor: '#e8f5e9' }, { stop: 1, bgColor: '#ffcdd2' }] });
      }
      this.current = candidate;
    } catch (error) {
      this.mount.replaceChildren(...(previous ? [previous] : []));
      throw error;
    }
  }
}
