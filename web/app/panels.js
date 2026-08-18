/* The chrome around the mount: controls, disclosures, evidence, legend.
 *
 * DOM only. Every value that came from a model, a capability or a diagnostic is
 * written with `textContent`, never `innerHTML` -- these strings are bounded by
 * `Me_limits` but they are not trusted markup.
 *
 * This module owns no lifecycle. It never touches `#visualizer`, never calls
 * the renderer, and never starts a request; it renders what it is given and
 * reports control changes to its caller.
 */

import {
  BACKBONE_STAGES, OPTIONAL_STAGES, EFFORTS, RANK_BUCKETS,
  capabilityWording,
} from './presentation.js';

const STAGE_NAMES = {
  source: 'Exported Program',
  initial_native: 'Initial Native',
  canonical: 'Canonical Native',
  native4d: 'Native4D',
  stage_program: 'Stage Program',
  kernel: 'Kernel',
  fusion: 'Kernel fusion',
};

const EFFORT_NAMES = { quick: 'Quick', standard: 'Standard', thorough: 'Thorough' };

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

function checkbox(id, labelText, checked, disabled, onChange) {
  const label = el('label', null, 'option');
  const input = document.createElement('input');
  input.type = 'checkbox';
  input.id = id;
  input.checked = checked;
  input.disabled = !!disabled;
  input.addEventListener('change', () => onChange(input.checked));
  label.append(input, el('span', labelText));
  return label;
}

/* --------------------------------------------------------------- controls */

/** Optional stages only. The three backbones are stated, never offered. */
export function renderStageControls(container, backboneNote, controls, onChange) {
  clear(container);
  backboneNote.textContent =
    `Always built: ${BACKBONE_STAGES.map((s) => STAGE_NAMES[s]).join(', ')}. `
    + 'Every comparison and every later stage rests on these, so they are not optional.';
  const selected = new Set(controls.optional);
  /* Read back from the boxes rather than from `selected`, which is only a
   * snapshot: a second click before the re-render that follows the first would
   * otherwise compute its change against a stale set and drop one. */
  const current = () => OPTIONAL_STAGES.filter((s) => container.querySelector(`#stage-${s}`)?.checked);
  for (const stage of OPTIONAL_STAGES) {
    container.append(checkbox(`stage-${stage}`, STAGE_NAMES[stage], selected.has(stage), false,
      () => onChange(current())));
  }
}

/** Off, Quick, Standard, Thorough — mapped exactly to `verifySymbolic`. */
export function renderEffortControls(container, controls, onChange) {
  clear(container);
  const add = (value, text) => {
    const label = el('label', null, 'option');
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'effort';
    input.value = value ?? '';
    input.checked = controls.effort === value;
    input.addEventListener('change', () => { if (input.checked) onChange(value); });
    label.append(input, el('span', text));
    container.append(label);
  };
  add(null, 'Off');
  for (const effort of EFFORTS) add(effort, EFFORT_NAMES[effort]);
}

/* Folding is inert for every source this build accepts, and the capability the
 * exporter returned is what says so -- the control reflects it rather than
 * asserting anything of its own. */
export function renderFoldControl(input, note, index) {
  const capability = index?.capabilityByKey.get('feature:fold');
  const wording = capability ? capabilityWording(capability) : null;
  input.checked = false;
  input.disabled = true;
  note.textContent = wording && wording.state !== 'available'
    ? `Unavailable: ${wording.detail}. A PT2 archive can supply constant payloads; a bare model.json cannot.`
    : 'A PT2 archive can supply constant payloads; a bare model.json cannot.';
}

/* The NORMALISED options, echoed by the bridge. Shown because request-stage
 * ordering and duplicates are not user choices: what was asked for is what
 * OCaml made of the request, not how a control spelled it. */
export function renderEffectiveOptions(node, options) {
  if (!options) { node.textContent = ''; return; }
  const stages = (options.stages ?? []).map((s) => STAGE_NAMES[s] ?? s).join(', ');
  const verification = options.verifySymbolic
    ? `symbolic verification ${options.verifySymbolic}`
    : 'symbolic verification off';
  node.textContent = `Requested: ${stages} · folding ${options.fold ? 'on' : 'off'} · ${verification}`;
}

/* ---------------------------------------------------------- presentation */

/** Exactly the session's stage views, in the order it declared them. */
export function renderViewSelector(select, index, currentViewId) {
  clear(select);
  for (const view of index.views) {
    const option = new Option(view.label, view.id);
    option.selected = view.id === currentViewId;
    select.add(option);
  }
  select.disabled = index.views.length <= 1;
}

/* Exactly the session's declared comparisons, in the order it declared them,
 * with `Comparison.label` verbatim -- `Session.comparisons` is the sole
 * authority for what is selectable, and no capability is consulted.
 *
 * The leading empty option is the only entry not from the session, and it has
 * to exist: without it the control could enter compare mode but never leave,
 * since choosing the view selector is a different control. Its value is the
 * empty string, which is not a legal comparison id, so it cannot collide.
 */
export const NO_COMPARISON = '';

export function renderComparisonSelector(select, index, currentId) {
  clear(select);
  select.add(new Option('Single view', NO_COMPARISON));
  for (const comparison of index.comparisons) {
    const option = new Option(comparison.label, comparison.id);
    option.selected = comparison.id === currentId;
    select.add(option);
  }
  if (!currentId) select.value = NO_COMPARISON;
  select.disabled = index.comparisons.length === 0;
}

/* What the two panes are, named from the session's own stage views.
 *
 * Rendered only for a comparison that is ON SCREEN. A heading written while a
 * replacement was still being prepared would label the graph still showing with
 * the name of one it may never be replaced by.
 */
export function renderPaneLabels(node, index, comparison) {
  clear(node);
  if (!comparison) { node.hidden = true; return; }
  node.hidden = false;
  const heading = (side) => index.viewByGraph.get(side.graph)?.label ?? side.graph;
  node.append(el('span', `Left: ${heading(comparison.left)}`, 'pane-label'));
  node.append(el('span', `Right: ${heading(comparison.right)}`, 'pane-label'));
}

/* What the comparison on screen actually claims, said from the session and
 * nothing else. The count is exact and is never described as complete: the
 * exporter emits the correspondence components it computed, and a node in
 * neither pane's component is simply unclaimed.
 */
export function renderComparisonSummary(node, comparison) {
  clear(node);
  if (!comparison) { node.hidden = true; return; }
  node.hidden = false;
  const sync = comparison.sync ?? {};
  const count = Array.isArray(sync.entries) ? sync.entries.length : 0;
  const say = (line) => node.append(el('p', line, 'note'));

  if (count > 0) {
    say(`Navigation uses the ${count} exported correspondence components. `
      + 'Nodes outside them are not claimed to correspond.');
  } else {
    // The two regimes an empty list can mean are distinguished on the wire, not
    // guessed here -- see `matchNodeIdFallback` in `Me_session.Sync_navigation`.
    say('No explicit changed-node mapping was exported.');
    say(sync.matchNodeIdFallback === true
      ? 'This comparison declares that nodes sharing an identifier correspond, '
        + 'so navigation pairs them and only the nodes whose identifiers changed are marked.'
      : 'Nothing pairs the two panes, so selecting a node on one side moves nothing on the other.');
  }

  if (sync.showDiffHighlights === true) {
    say('Nodes with no counterpart are outlined: removed nodes on the left, '
      + 'added nodes on the right.');
  }

  const overlays = (list, side) => {
    const n = Array.isArray(list) ? list.length : 0;
    if (n > 0) say(`The ${side} pane carries ${n} supplied edge overlays.`);
  };
  overlays(comparison.overlaysLeft, 'left');
  overlays(comparison.overlaysRight, 'right');
}

/* Every capability that is not a navigable view, with its own state kept
 * distinct: "not requested" is a choice, "unavailable" is the exporter's typed
 * reason, and Loop IR and Code generation stay visibly unimplemented rather
 * than silently disappearing. */
export function renderUnavailable(container, summary, index) {
  clear(container);
  const rows = index.capabilities
    .map(capabilityWording)
    .filter((w) => w.state !== 'available');
  summary.textContent = `Unavailable outputs (${rows.length})`;
  if (rows.length === 0) {
    container.append(el('p', 'Every declared output is available.', 'note'));
    return;
  }
  const list = el('dl', null, 'rows');
  for (const row of rows) {
    list.append(el('dt', `${row.name} — ${row.title}`, `state-${row.state}`));
    list.append(el('dd', row.detail));
  }
  container.append(list);
}

export function renderDiagnostics(container, summary, index) {
  clear(container);
  summary.textContent = `Diagnostics (${index.diagnostics.length})`;
  if (index.diagnostics.length === 0) {
    container.append(el('p', 'No diagnostics were reported.', 'note'));
    return;
  }
  const list = el('dl', null, 'rows');
  for (const d of index.diagnostics) {
    const where = d.graph ? ` · ${d.graph}` : '';
    list.append(el('dt', `${d.code}${where}`));
    list.append(el('dd', d.truncated ? `${d.message} (truncated)` : d.message));
  }
  container.append(list);
}

/* -------------------------------------------------------------- evidence */

/* Counts are `int64_as_string` on the wire and are rendered VERBATIM: parsing
 * one through `Number` would lose exactness past 2^53, and nothing here needs
 * their numeric value. */
function countList(bindings) {
  const list = el('dl', null, 'rows');
  for (const binding of bindings) {
    list.append(el('dt', binding.label));
    list.append(el('dd', binding.count));
  }
  return list;
}

/* Two keys, said as two things. `feature:verification` is a COMPOSED
 * whole-pipeline report counted in verifier clusters; `feature:pass_audits`
 * counts audit REPORTS, and an omitted audit report does not invalidate an
 * available composed summary. */
export function renderValidation(container, summary, index) {
  clear(container);
  const verification = index.capabilityByKey.get('feature:verification');
  const wording = verification ? capabilityWording(verification) : null;

  if (!index.verification) {
    summary.textContent = 'Validation evidence — none';
    const why = wording && wording.state !== 'available'
      ? `${wording.title}: ${wording.detail}`
      : 'Symbolic verification was not requested for this export.';
    container.append(el('p', why, 'note'));
    container.append(el('p',
      'No verification provider is installed on the graph. Nothing here claims that this transformation was checked.',
      'note'));
    return;
  }

  summary.textContent = 'Validation evidence';
  container.append(el('h3', 'Symbolic transformation verification'));
  container.append(el('p',
    'A composed, whole-pipeline report, counted in verifier clusters. Each label '
    + 'carries its strength and coverage; neither is shortened.', 'note'));
  container.append(countList(index.verification));

  container.append(el('h3', 'Per-pass audits'));
  if (!index.passAudits) {
    container.append(el('p', 'No per-pass audit reports were retained.', 'note'));
    return;
  }
  const audits = index.passAudits;
  container.append(el('p',
    `Retained ${audits.retainedReports} audit reports; omitted ${audits.omittedReports}. `
    + 'These count reports, not clusters — an omitted report does not invalidate the composed summary above.',
    'note'));
  if ((audits.omittedCounts ?? []).length > 0) {
    container.append(el('p', 'Clusters in the omitted reports:', 'note'));
    container.append(countList(audits.omittedCounts));
  }
}

/* The named rank buckets, repeated beside the mount so a colour on a node has a
 * meaning on screen rather than in a commit message. Shown only when a
 * verification provider was actually installed. */
export function renderLegend(container, index) {
  clear(container);
  if (!index.verification) { container.hidden = true; return; }
  container.hidden = false;
  container.append(el('h3', 'Verification outcome'));
  const list = el('dl', null, 'legend-rows');
  for (const bucket of RANK_BUCKETS) {
    const term = el('dt', bucket.title);
    term.style.backgroundColor = bucket.bg;
    list.append(term, el('dd', bucket.meaning));
  }
  container.append(list);
  container.append(el('p',
    'A node shows the weakest outcome over its outputs, with the exporter’s own label.', 'note'));
}
