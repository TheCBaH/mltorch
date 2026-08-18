import { Coordinator } from './coordinator.js';
import { Renderer } from './renderer.js';
import { SourceStore, validateCatalog } from './source_store.js';
import * as P from './presentation.js';
import * as panels from './panels.js';

const $ = (id) => document.getElementById(id);
const status = $('status'), error = $('error'), notice = $('notice');
const select = $('catalogue'), file = $('local-file'), viewSelect = $('view');
const compareSelect = $('comparison');
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
  let pending = null;       // { model, presentation, push } of the load in flight
  /* The single view to return to when a comparison is closed. A comparison is
   * not a view, so while one is on screen the view selector has nothing true to
   * show; remembering the last single view is what makes "Single view" a
   * well-defined destination rather than a guess. */
  let lastView = null;

  const asView = (presentation) =>
    (presentation?.kind === 'single' ? presentation.view : null);
  const asComparison = (presentation) =>
    (presentation?.kind === 'comparison' ? presentation.comparison : null);
  /* A load resolves a VIEW: `prefer` is an ordered list of view ids, and a
   * comparison names two graphs rather than one. So a URL that asks for a
   * comparison still loads a single view first, and the comparison is opened on
   * top of the document as a presentation change. */
  const preferFor = (presentation) => P.preferredViews(asView(presentation));

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

  /* Everything that describes WHAT IS ON SCREEN, from the presentation actually
   * opened. The pane headings and the comparison summary are rendered from the
   * same value as the selectors, so the chrome cannot describe one presentation
   * while another is showing. */
  const renderSession = (presentation) => {
    if (!index) return;
    const comparisonId = asComparison(presentation);
    const comparison = comparisonId ? index.comparisonById.get(comparisonId) ?? null : null;
    panels.renderViewSelector(viewSelect, index, asView(presentation) ?? lastView);
    panels.renderComparisonSelector(compareSelect, index, comparisonId);
    panels.renderPaneLabels($('pane-labels'), index, comparison);
    panels.renderComparisonSummary($('presentation-note'), comparison);
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
      const shown = coordinator.presentation;
      if (viewId) lastView = viewId;
      renderSession(shown);
      const wanted = pending?.presentation ?? null;
      const push = pending?.push ?? false;
      pending = null;
      // Resolved against the document that actually arrived: a comparison this
      // export does not declare is stale in exactly the way a missing view is.
      const resolved = P.resolvePresentation(index, wanted);
      const stale = wanted?.kind === 'comparison'
        ? P.staleComparisonNotice(wanted.comparison, resolved)
        : P.staleNotice(asView(wanted), viewId);
      if (stale) showNotice(stale); else clearNotice();
      // A user-originated selection -- a model or an option -- earns a history
      // entry; a reload, a normalisation and a history restoration do not.
      // Either way the URL written is the ECHOED options and the presentation
      // actually shown, never what a control spelled or a stale URL asked for.
      writeUrl(push ? 'push' : 'replace', shown);
      // A comparison cannot be reached by a load's view preference, so it is
      // opened on top of the document just installed. It replaces the URL it
      // just wrote rather than pushing: one navigation, one entry.
      if (resolved?.kind === 'comparison') {
        changePresentation(resolved, 'replace').catch((e) => showError(e.message));
      }
    },
  });

  const writeUrl = (mode, presentation) => {
    // A local file cannot be reproduced from a URL, so its URL claims nothing.
    const url = loadedModel
      ? `${location.pathname}${P.encodeUrl({ model: loadedModel, options: coordinator.effectiveOptions, presentation })}`
      : location.pathname;
    if (mode === 'push') history.pushState({}, '', url);
    else history.replaceState({}, '', url);
  };

  const startLoad = async (source, { model = null, options, presentation = null, push = false }) => {
    clearError(); clearNotice();
    coordinator.cancel();
    pending = { model, presentation, push };
    await coordinator.load({ ...source, readBytes: () => store.freshBytes() },
      { options, prefer: preferFor(presentation) });
  };

  /* Every construction-control change starts a new export, keeping whatever is
   * currently on screen -- view or comparison -- as a preference. */
  const reexport = () => {
    const source = store.source;
    if (!source) return showError('choose a model first');
    startLoad(source, {
      model: loadedModel, options: P.optionsFromControls(controls),
      presentation: coordinator.presentation, push: true,
    }).catch((e) => showError(e.message));
  };

  const loadCatalog = async (id, options, presentation, push) => {
    const entry = catalog.models.find((model) => model.id === id);
    if (!entry) return showError('unknown catalogue model');
    clearError(); clearNotice();
    coordinator.cancel();
    const source = await store.fetchCatalog({ ...entry, sourceRef: catalog.sourceRef });
    await startLoad(source, { model: id, options, presentation, push });
  };

  /* Presentation only: no worker message, no bridge call, no rewritten session
   * document -- the renderer replays the text the coordinator already holds.
   *
   * The descriptor is translated into the renderer's selection here and nowhere
   * else, so `presentation.js`'s closed sum is the only vocabulary the rest of
   * the page speaks. */
  const changePresentation = async (presentation, mode) => {
    clearError(); clearNotice();
    const before = coordinator.presentation;
    coordinator.cancel();
    await coordinator.present(presentation.kind === 'comparison'
      ? { comparison: presentation.comparison }
      : { view: presentation.view });
    const shown = coordinator.presentation;
    if (shown?.kind === 'single') lastView = shown.view;
    // Always: a refused change must not leave a control claiming a destination
    // the page never reached.
    renderSession(shown);
    // But only when something actually moved. A cancelled or refused change
    // leaves the previous presentation on screen, and a history entry for it
    // would record a navigation that never happened.
    if (!P.samePresentation(before, shown)) writeUrl(mode, shown);
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
    startLoad(source, { model: null, options: P.optionsFromControls(controls), presentation: null })
      .catch((e) => showError(e.message));
  });
  reload.addEventListener('click', () => {
    const source = store.source;
    if (!source) return showError('choose a model first');
    // The controls are browser state now, so a reload rebuilds the
    // configuration on screen rather than the bridge's defaults.
    startLoad(source, {
      model: loadedModel, options: P.optionsFromControls(controls),
      presentation: coordinator.presentation,
    }).catch((e) => showError(e.message));
  });
  clearLocal.addEventListener('click', () => {
    if (store.source?.kind !== 'local') return;
    coordinator.cancel(); store.clear(); file.value = ''; clearError(); status.textContent = 'Local file cleared';
  });
  /* Both selectors fail the same way: on error the chrome is re-rendered from
   * what the coordinator is still showing, so a control can never claim a
   * presentation the page failed to open. */
  const changeFromControl = (presentation) => {
    changePresentation(presentation, 'push').catch((e) => {
      showError(e.message);
      renderSession(coordinator.presentation);
    });
  };
  viewSelect.addEventListener('change', () => {
    changeFromControl(P.singlePresentation(viewSelect.value));
  });
  compareSelect.addEventListener('change', () => {
    const id = compareSelect.value;
    // "Single view" returns to the view the selector is showing, which is the
    // last single view opened -- not a fresh default, and never a guess.
    changeFromControl(id === panels.NO_COMPARISON
      ? P.singlePresentation(viewSelect.value)
      : P.comparisonPresentation(id));
  });

  /* Startup and history share one rule: compare the request the URL describes
   * with the request behind the retained session. A difference is a load; an
   * equal request naming a different, RESOLVING presentation is a presentation
   * change; anything else is nothing at all. `popstate` never pushes -- it is
   * already navigating. */
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
      return loadCatalog(id, options, decoded.presentation, false);
    }
    const wanted = P.resolvePresentation(index, decoded.presentation);
    if (wanted) {
      if (P.samePresentation(wanted, coordinator.presentation)) return;
      return changePresentation(wanted, mode);
    }
    // The URL named something this session does not declare. Nothing is opened
    // and nothing is torn down; the URL is corrected to what is on screen so a
    // second `popstate` cannot keep re-asking for it.
    if (decoded.presentation) {
      const notice = decoded.presentation.kind === 'comparison'
        ? P.staleComparisonNotice(decoded.presentation.comparison, null)
        : P.staleNotice(decoded.presentation.view, coordinator.view);
      if (notice) showNotice(notice);
      writeUrl('replace', coordinator.presentation);
    }
  };

  addEventListener('popstate', () => { applyUrl('replace').catch((e) => showError(e.message)); });
  await applyUrl('replace');
}

main().catch((e) => showError(e.message || String(e)));
