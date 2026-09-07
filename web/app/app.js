import { Coordinator } from './coordinator.js';
import { Renderer } from './renderer.js';
import { SourceStore, validateCatalog } from './source_store.js';
import * as P from './presentation.js';
import * as panels from './panels.js';

const $ = (id) => document.getElementById(id);
const status = $('status'), error = $('error'), notice = $('notice');
const select = $('catalogue'), file = $('local-file'), viewSelect = $('view');
const compareSelect = $('comparison');
const constantsSelect = $('constants');
const reload = $('reload'), clearLocal = $('clear-local');
const flowButton = $('flow-open');
const symbolicAction = $('symbolic-action'), symbolicSelection = $('symbolic-selection');
const symbolicOpen = $('symbolic-open');
const bridge = globalThis.mltorch;

function showError(message) { error.textContent = message; error.hidden = false; }
function clearError() { error.hidden = true; error.textContent = ''; }
function showNotice(message) { notice.textContent = message; notice.hidden = false; }
function clearNotice() { notice.hidden = true; notice.textContent = ''; }

async function main() {
  if (!bridge?.hard || !bridge?.request?.buildDetail || !bridge?.session?.applyDetail) throw new Error('MLTorch bridge did not load');
  const store = new SourceStore({ hard: bridge.hard });
  // The quarantine ceiling is the OCaml one: `Me_limits.response_live_bytes`
  // budgets the browser's retained elements against exactly this number.
  /* `onFlowSelect` reports a SELECTION, never a navigation: under select-then-
   * act the action row is what navigates, and only when pressed. `null` means
   * the selection was cleared. */
  const renderer = new Renderer({
    mount: $('visualizer'),
    hardMaxQuarantined: bridge.hard.maxQuarantinedElements,
    onFlowSelect: (selection) => {
      flowSelection = selection;
      const canonical = index?.viewById.get('v/canonical');
      operatorSelection = coordinator?.presentation?.kind === 'single'
        && coordinator.presentation.view === 'v/canonical'
        && selection?.graphId === canonical?.graph
        && selection?.collectionLabel === canonical?.collection
        && /^n[0-9]+$/.test(selection.nodeId)
        ? selection : null;
      renderFlowChrome(); renderSymbolicAction();
    },
  });

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
  /* The flow node the user last selected, or null. Browser-only state: it is not
   * part of any presentation, so it never reaches the URL. */
  let flowSelection = null;
  let operatorSelection = null;

  const asView = (presentation) =>
    (presentation?.kind === 'single' ? presentation.view : null);
  const asComparison = (presentation) =>
    (presentation?.kind === 'comparison' ? presentation.comparison : null);
  const asFlow = (presentation) =>
    (presentation?.kind === 'flow' ? presentation.view : null);
  /* A load resolves a VIEW: `prefer` is an ordered list of view ids, and a
   * comparison names two graphs rather than one. So a URL that asks for a
   * comparison still loads a single view first, and the comparison is opened on
   * top of the document as a presentation change. */
  const preferFor = (presentation) => P.preferredViews(asView(presentation));

  /* The catalogue id whose capability signal should govern the stage controls
   * and stay selected in the dropdown right now: the model behind the
   * retained session, or -- before any session exists -- whatever the
   * selector already shows. */
  const currentCatalogueId = () => loadedModel ?? select.value ?? null;

  const renderControls = (selectedId = currentCatalogueId()) => {
    const support = catalog.models.find((m) => m.id === selectedId)?.support ?? null;
    panels.renderStageControls($('stage-controls'), $('backbone-note'), controls, support, (optional) => {
      controls = { ...controls, optional };
      panels.renderCatalogueOptions(select, catalog, optional, currentCatalogueId());
      reexport();
    });
    panels.renderEffortControls($('effort-controls'), controls, (effort) => {
      controls = { ...controls, effort };
      reexport();
    });
    panels.renderFoldControl($('fold'), $('fold-note'), index);
    panels.renderEffectiveOptions($('effective-options'), coordinator.effectiveOptions);
    panels.renderCatalogueOptions(select, catalog, controls.optional, selectedId);
    renderConstantsControl();
  };

  /* Reflects the coordinator's OWN mode, never a browser-local copy of it --
   * `coordinator.constants` already defaults to `P.DEFAULT_CONSTANTS` before
   * any load, so there is nothing this control needs to track on its own. */
  const renderConstantsControl = () => { constantsSelect.value = coordinator.constants; };

  /* Everything that describes WHAT IS ON SCREEN, from the presentation actually
   * opened. The pane headings and the comparison summary are rendered from the
   * same value as the selectors, so the chrome cannot describe one presentation
   * while another is showing. */
  /* The flow half, re-rendered on its own whenever a selection changes -- a
   * selection is not a presentation change, so nothing else moves. */
  const renderFlowChrome = () => {
    if (!index) return;
    const active = asFlow(coordinator.presentation) !== null;
    panels.renderFlowControl(flowButton, $('flow-note'), index, active);
    panels.renderFlowLegend($('flow-legend'), index, active);
    panels.renderFlowSelection($('flow-selection'), index, active ? flowSelection : null,
      (presentation) => {
        changeFromControl(presentation);
      });
  };

  const renderSymbolicAction = () => {
    symbolicAction.hidden = !operatorSelection;
    if (operatorSelection) {
      symbolicSelection.textContent = `Selected canonical operator ${operatorSelection.nodeId}`;
    }
  };

  const renderSession = (presentation) => {
    if (!index) return;
    const comparisonId = asComparison(presentation);
    const comparison = comparisonId ? index.comparisonById.get(comparisonId) ?? null : null;
    panels.renderViewSelector(viewSelect, index, asView(presentation) ?? lastView);
    panels.renderComparisonSelector(compareSelect, index, comparisonId);
    panels.renderPaneLabels($('pane-labels'), index, comparison);
    panels.renderComparisonSummary($('presentation-note'), comparison);
    renderFlowChrome();
    renderSymbolicAction();
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
        : wanted?.kind === 'flow'
          ? P.staleFlowNotice(wanted.view, resolved)
          : P.staleNotice(asView(wanted), viewId);
      if (stale) showNotice(stale); else clearNotice();
      // A user-originated selection -- a model or an option -- earns a history
      // entry; a reload, a normalisation and a history restoration do not.
      // Either way the URL written is the ECHOED options and the presentation
      // actually shown, never what a control spelled or a stale URL asked for.
      writeUrl(push ? 'push' : 'replace', shown);
      // Neither a comparison NOR a flow can be reached by a load's view
       // preference -- `prefer` is a list of stage view ids, and these name two
       // graphs and a flow graph respectively. Both are opened on top of the
       // document just installed, replacing the URL just written rather than
       // pushing: one navigation, one entry.
      if (resolved && resolved.kind !== 'single') {
        changePresentation(resolved, 'replace').catch((e) => showError(e.message));
      }
    },
    onDetail: ({ sessionText, viewId }) => {
      index = P.buildIndex(sessionText);
      operatorSelection = null;
      if (viewId) lastView = viewId;
      renderSession(coordinator.presentation);
      writeUrl('push', coordinator.presentation);
    },
  });

  const writeUrl = (mode, presentation) => {
    // A local file cannot be reproduced from a URL, so its URL claims nothing.
    const url = loadedModel
      ? `${location.pathname}${P.encodeUrl({
        model: loadedModel, options: coordinator.effectiveOptions, presentation, constants: coordinator.constants,
      })}`
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
    /* The one place a descriptor becomes a renderer selection. A flow must NOT
     * fall through to `{view}`: the stage-only route would refuse it, and the
     * page would report a failure for a destination it declares. */
    await coordinator.present(presentation.kind === 'comparison'
      ? { comparison: presentation.comparison }
      : presentation.kind === 'flow'
        ? { flow: presentation.view }
        : { view: presentation.view });
    const shown = coordinator.presentation;
    if (shown?.kind === 'single') lastView = shown.view;
    // A selection belongs to the flow that was on screen when it was made.
    if (shown?.kind !== 'flow') flowSelection = null;
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
  renderControls(catalog.defaultModel);

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
  flowButton.addEventListener('click', () => {
    // A toggle: into the flow, or back to the single view the selector shows.
    changeFromControl(asFlow(coordinator.presentation)
      ? P.singlePresentation(viewSelect.value)
      : P.flowPresentation(index.flowView.id));
  });
  compareSelect.addEventListener('change', () => {
    const id = compareSelect.value;
    // "Single view" returns to the view the selector is showing, which is the
    // last single view opened -- not a fresh default, and never a guess.
    changeFromControl(id === panels.NO_COMPARISON
      ? P.singlePresentation(viewSelect.value)
      : P.comparisonPresentation(id));
  });
  symbolicOpen.addEventListener('click', () => {
    const selected = operatorSelection;
    if (!selected) return;
    const node = Number(selected.nodeId.slice(1));
    if (!Number.isSafeInteger(node)) return showError('selected operator id is invalid');
    clearError(); clearNotice();
    const installed = index.detailByParent.get(`${selected.graphId}\u0000${selected.nodeId}`);
    if (installed) {
      changePresentation(P.singlePresentation(installed.id), 'push').catch((e) => showError(e.message));
      return;
    }
    coordinator.openDetail({ parentGraph: selected.graphId, node })
      .catch((e) => showError(e.message));
  });
  /* No worker, no bridge `prepare`/`commit` -- only `groupConstants` -- and no
   * change to which graph or view is on screen. The same failure handling as
   * `changeFromControl`: on error the control reverts to what is actually
   * showing rather than keeping the selection the page failed to reach. */
  constantsSelect.addEventListener('change', () => {
    clearError(); clearNotice();
    coordinator.cancel();
    coordinator.setConstants(constantsSelect.value).then(() => {
      writeUrl('push', coordinator.presentation);
    }, (e) => showError(e.message)).finally(renderConstantsControl);
  });

  /* Startup and history share one rule: compare the request the URL describes
   * with the request behind the retained session. A difference is a load; an
   * equal request naming a different, RESOLVING presentation is a presentation
   * change; anything else is nothing at all. `popstate` never pushes -- it is
   * already navigating. */
  const applyUrl = async (mode) => {
    const decoded = P.decodeUrl(location.search);
    const desiredConstants = P.constantsFromUrl(decoded);
    const id = decoded.model ?? catalog.defaultModel;
    if (!catalog.models.some((model) => model.id === id)) return showError('unknown catalogue model');
    const options = P.optionsFromUrl(decoded);
    const retained = coordinator.session && loadedModel
      ? P.requestKey(loadedModel, coordinator.effectiveOptions)
      : null;
    if (P.requestKey(id, options) !== retained) {
      // No re-present: the plain setter, since the load about to start (never
      // the session on screen right now, which `cancel()` is about to tear
      // down anyway) is what will pick this mode up.
      coordinator.constants = desiredConstants;
      controls = P.controlsFromOptions(options);
      // Renders the selector with `id` kept and reselected, ahead of the load
      // this triggers below -- the model about to load must already be the one
      // on screen, not whatever the previous filtering left selected.
      renderControls(id);
      // Never a push: startup has no entry to add to, and a `popstate` is
      // already navigating.
      return loadCatalog(id, options, decoded.presentation, false);
    }
    // Same document already on screen: a changed `constants` is a
    // presentation change of its own, independent of which view/comparison
    // the rest of this function goes on to resolve.
    if (desiredConstants !== coordinator.constants) {
      await coordinator.setConstants(desiredConstants);
      renderConstantsControl();
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
        : decoded.presentation.kind === 'flow'
          ? P.staleFlowNotice(decoded.presentation.view, null)
          : P.staleNotice(decoded.presentation.view, coordinator.view);
      if (notice) showNotice(notice);
      writeUrl('replace', coordinator.presentation);
    }
  };

  addEventListener('popstate', () => { applyUrl('replace').catch((e) => showError(e.message)); });
  await applyUrl('replace');
}

main().catch((e) => showError(e.message || String(e)));
