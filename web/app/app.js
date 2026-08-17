import { Coordinator } from './coordinator.js';
import { Renderer } from './renderer.js';
import { SourceStore, validateCatalog } from './source_store.js';
import * as P from './presentation.js';
import * as panels from './panels.js';

const $ = (id) => document.getElementById(id);
const status = $('status'), error = $('error'), notice = $('notice');
const select = $('catalogue'), file = $('local-file'), viewSelect = $('view');
const reload = $('reload'), clearLocal = $('clear-local');
const bridge = globalThis.mltorch;

function showError(message) { error.textContent = message; error.hidden = false; }
function clearError() { error.hidden = true; error.textContent = ''; }
function showNotice(message) { notice.textContent = message; notice.hidden = false; }
function clearNotice() { notice.hidden = true; notice.textContent = ''; }

async function main() {
  if (!bridge?.hard || !bridge?.request || !bridge?.session) throw new Error('MLTorch bridge did not load');
  const store = new SourceStore({ hard: bridge.hard });
  // The quarantine ceiling is the OCaml one: `Me_limits.response_live_bytes`
  // budgets the browser's retained elements against exactly this number.
  const renderer = new Renderer({ mount: $('visualizer'), hardMaxQuarantined: bridge.hard.maxQuarantinedElements });

  /* Browser-owned state. `index` is derived from the coordinator's retained
   * session and never from anything else; `controls` is what the user has
   * chosen, which is only the same as the effective options between requests. */
  let index = null;
  let controls = { optional: [...P.OPTIONAL_STAGES], effort: null };
  let loadedModel = null;   // catalogue id behind the retained session; null for a local file
  let pending = null;       // { model, view, push } of the load in flight

  const renderControls = () => {
    panels.renderStageControls($('stage-controls'), $('backbone-note'), controls, (optional) => {
      controls = { ...controls, optional };
      reexport();
    });
    panels.renderEffortControls($('effort-controls'), controls, (effort) => {
      controls = { ...controls, effort };
      reexport();
    });
    panels.renderFoldControl($('fold'), $('fold-note'), index);
    panels.renderEffectiveOptions($('effective-options'), coordinator.effectiveOptions);
  };

  const renderSession = (viewId) => {
    if (!index) return;
    panels.renderViewSelector(viewSelect, index, viewId);
    panels.renderUnavailable($('unavailable-list'), $('unavailable').firstElementChild, index);
    panels.renderDiagnostics($('diagnostics-list'), $('diagnostics').firstElementChild, index);
    panels.renderValidation($('validation-body'), $('validation').firstElementChild, index);
    panels.renderLegend($('legend'), index);
  };

  const coordinator = new Coordinator({
    hard: bridge.hard, bridge, render: renderer,
    workerFactory: () => new Worker(new URL('./worker.js', document.baseURI)),
    onStatus: (text) => { status.textContent = text; }, onError: showError,
    /* The one completion point. Everything that describes a session -- the
     * index, the controls, the URL -- is built here and nowhere else, because
     * `load()` returns as soon as the worker has been posted to and cannot say
     * whether the model was ever shown. */
    onSession: ({ sessionText, effectiveOptions, viewId }) => {
      index = P.buildIndex(sessionText);
      loadedModel = pending?.model ?? null;
      // The controls follow the NORMALISED options, so what is ticked is what
      // was actually requested rather than what was typed.
      controls = P.controlsFromOptions(effectiveOptions);
      renderControls();
      renderSession(viewId);
      const stale = P.staleNotice(pending?.view, viewId);
      if (stale) showNotice(stale); else clearNotice();
      // A user-originated selection -- a model or an option -- earns a history
      // entry; a reload, a normalisation and a history restoration do not.
      // Either way the URL written is the ECHOED options and the view actually
      // shown, never what a control spelled or a stale URL asked for.
      writeUrl(pending?.push ? 'push' : 'replace', viewId);
      pending = null;
    },
  });

  const writeUrl = (mode, viewId) => {
    // A local file cannot be reproduced from a URL, so its URL claims nothing.
    const url = loadedModel
      ? `${location.pathname}${P.encodeUrl({ model: loadedModel, options: coordinator.effectiveOptions, view: viewId })}`
      : location.pathname;
    if (mode === 'push') history.pushState({}, '', url);
    else history.replaceState({}, '', url);
  };

  const startLoad = async (source, { model = null, options, view = null, push = false }) => {
    clearError(); clearNotice();
    coordinator.cancel();
    pending = { model, view, push };
    await coordinator.load({ ...source, readBytes: () => store.freshBytes() },
      { options, prefer: P.preferredViews(view) });
  };

  /* Every construction-control change starts a new export, keeping the view
   * currently on screen as a preference. */
  const reexport = () => {
    const source = store.source;
    if (!source) return showError('choose a model first');
    startLoad(source, {
      model: loadedModel, options: P.optionsFromControls(controls),
      view: coordinator.view, push: true,
    }).catch((e) => showError(e.message));
  };

  const loadCatalog = async (id, options, view, push) => {
    const entry = catalog.models.find((model) => model.id === id);
    if (!entry) return showError('unknown catalogue model');
    clearError(); clearNotice();
    coordinator.cancel();
    const source = await store.fetchCatalog({ ...entry, sourceRef: catalog.sourceRef });
    await startLoad(source, { model: id, options, view, push });
  };

  /* Presentation only: no worker message, no bridge call, no rewritten session
   * document -- the renderer replays the text the coordinator already holds. */
  const changeView = async (viewId, mode) => {
    clearError(); clearNotice();
    coordinator.cancel();
    await coordinator.present({ view: viewId });
    renderSession(coordinator.view);
    writeUrl(mode, coordinator.view);
  };

  const response = await fetch(new URL('./catalog.json', document.baseURI));
  if (!response.ok) throw new Error(`catalogue unavailable (${response.status})`);
  const catalog = validateCatalog(await response.json());
  for (const model of catalog.models) { const option = new Option(model.displayName, model.id); select.add(option); }
  renderControls();

  select.addEventListener('change', () => {
    loadCatalog(select.value, P.optionsFromControls(controls), null, true)
      .catch((e) => showError(e.message));
  });
  file.addEventListener('change', () => {
    if (!file.files[0]) return;
    clearError(); clearNotice(); coordinator.cancel();
    const source = store.setLocal(file.files[0]);
    loadedModel = null;
    // No history entry, and no query: a local file cannot be reproduced from a
    // URL, so the one written must not look as though it could be.
    startLoad(source, { model: null, options: P.optionsFromControls(controls), view: null })
      .catch((e) => showError(e.message));
  });
  reload.addEventListener('click', () => {
    const source = store.source;
    if (!source) return showError('choose a model first');
    // The controls are browser state now, so a reload rebuilds the
    // configuration on screen rather than the bridge's defaults.
    startLoad(source, { model: loadedModel, options: P.optionsFromControls(controls), view: coordinator.view })
      .catch((e) => showError(e.message));
  });
  clearLocal.addEventListener('click', () => {
    if (store.source?.kind !== 'local') return;
    coordinator.cancel(); store.clear(); file.value = ''; clearError(); status.textContent = 'Local file cleared';
  });
  viewSelect.addEventListener('change', () => {
    changeView(viewSelect.value, 'push').catch((e) => {
      showError(e.message);
      // The selector must not claim a view the page failed to open.
      renderSession(coordinator.view);
    });
  });

  /* Startup and history share one rule: compare the request the URL describes
   * with the request behind the retained session. A difference is a load; an
   * equal request with a different view is a presentation change; anything else
   * is nothing at all. `popstate` never pushes -- it is already navigating. */
  const applyUrl = async (mode) => {
    const decoded = P.decodeUrl(location.search);
    const id = decoded.model ?? catalog.defaultModel;
    if (!catalog.models.some((model) => model.id === id)) return showError('unknown catalogue model');
    select.value = id;
    const options = P.optionsFromUrl(decoded);
    const retained = coordinator.session && loadedModel
      ? P.requestKey(loadedModel, coordinator.effectiveOptions)
      : null;
    if (P.requestKey(id, options) !== retained) {
      controls = P.controlsFromOptions(options);
      renderControls();
      // Never a push: startup has no entry to add to, and a `popstate` is
      // already navigating.
      return loadCatalog(id, options, decoded.view, false);
    }
    if (decoded.view && decoded.view !== coordinator.view && index?.viewById.has(decoded.view)) {
      return changeView(decoded.view, mode);
    }
  };

  addEventListener('popstate', () => { applyUrl('replace').catch((e) => showError(e.message)); });
  await applyUrl('replace');
}

main().catch((e) => showError(e.message || String(e)));
