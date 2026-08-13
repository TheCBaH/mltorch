import { Coordinator } from './coordinator.js';
import { Renderer } from './renderer.js';
import { SourceStore, validateCatalog } from './source_store.js';

const $ = (id) => document.getElementById(id);
const status = $('status'), error = $('error'), select = $('catalogue'), file = $('local-file');
const reload = $('reload'), clearLocal = $('clear-local');
const bridge = globalThis.mltorch;

function showError(message) { error.textContent = message; error.hidden = false; }
function clearError() { error.hidden = true; error.textContent = ''; }

async function main() {
  if (!bridge?.hard || !bridge?.request || !bridge?.session) throw new Error('MLTorch bridge did not load');
  const store = new SourceStore({ hard: bridge.hard });
  const renderer = new Renderer({ mount: $('visualizer') });
  const coordinator = new Coordinator({
    hard: bridge.hard, bridge, render: renderer,
    workerFactory: () => new Worker(new URL('./worker.js', document.baseURI)),
    onStatus: (text) => { status.textContent = text; }, onError: showError,
  });
  const response = await fetch(new URL('./catalog.json', document.baseURI));
  if (!response.ok) throw new Error(`catalogue unavailable (${response.status})`);
  const catalog = validateCatalog(await response.json());
  for (const model of catalog.models) { const option = new Option(model.displayName, model.id); select.add(option); }
  const choose = async (id) => {
    clearError(); coordinator.cancel();
    const entry = catalog.models.find((model) => model.id === id);
    if (!entry) return showError('unknown catalogue model');
    const source = await store.fetchCatalog({ ...entry, sourceRef: catalog.sourceRef });
    await coordinator.load({ ...source, readBytes: () => store.freshBytes() });
  };
  select.addEventListener('change', () => { history.pushState({}, '', `?model=${encodeURIComponent(select.value)}`); choose(select.value).catch((e) => showError(e.message)); });
  file.addEventListener('change', () => { if (!file.files[0]) return; clearError(); coordinator.cancel(); const source = store.setLocal(file.files[0]); history.pushState({}, '', location.pathname); coordinator.load({ ...source, readBytes: () => store.freshBytes() }).catch((e) => showError(e.message)); });
  reload.addEventListener('click', () => {
    const source = store.source;
    if (!source) return showError('choose a model first');
    clearError(); coordinator.cancel();
    coordinator.load({ ...source, readBytes: () => store.freshBytes() }).catch((e) => showError(e.message));
  });
  clearLocal.addEventListener('click', () => {
    if (store.source?.kind !== 'local') return;
    coordinator.cancel(); store.clear(); file.value = ''; clearError(); status.textContent = 'Local file cleared';
  });
  addEventListener('popstate', () => {
    const id = new URL(location.href).searchParams.get('model') || catalog.defaultModel;
    if (!catalog.models.some((model) => model.id === id)) return showError('unknown catalogue model');
    select.value = id; choose(id).catch((e) => showError(e.message));
  });
  const initial = new URL(location.href).searchParams.get('model') || catalog.defaultModel;
  if (!catalog.models.some((model) => model.id === initial)) return showError('unknown catalogue model');
  select.value = initial; await choose(initial);
}

main().catch((e) => showError(e.message || String(e)));
