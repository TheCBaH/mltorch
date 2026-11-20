/* The lowering pane: what a backend family makes of one canonical node.
 *
 * DOM only, like `panels.js`: it renders the model it is given and reports
 * choices to its caller; it starts no request and owns no state. Every string
 * from an artifact is written with `textContent`.
 *
 * The family and variant names are NOT written here. The tabs and the variant
 * bar are built from `vocabulary` (the bridge's, which reads `Lowering_key`),
 * so a family added in OCaml appears here with no edit. What this file does own
 * is the wording: a refusal or note arrives as `{ kind, data }`, a tag with
 * arguments, and becomes text here and only here.
 */

const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
};

export const familyOf = (spelling) => spelling.split(':', 1)[0];

/* `data` is the producer's own rendering of the payload for every kind that
 * carries more than an id; the sentence around it is the page's. */
const REFUSALS = {
  granted: ([node]) => `Node ${node} is granted: no Loop body realizes it, so no family has code to show.`,
  loop_c: ([why]) => `The C emitter refused this kernel: ${why}`,
  loop_lowering: ([why]) => `Re-lowering the kernel under this pass selection failed: ${why}`,
  loop_wasm: ([why]) => `The WebAssembly emitter refused this kernel: ${why}`,
  ssa_backends: ([why]) => `The SSA backends refused this kernel: ${why}`,
  ssa_lowering: ([why]) => `This kernel has no SSA program: ${why}`,
  unknown_node: ([node]) => `The model has no node ${node}.`,
  unsupported_family: ([family]) => `The ${family} family has no artifact for a node yet.`,
  unsupported_policy: ([why]) => `This numerical policy is not available for the family: ${why}`,
};

const NOTES = {
  fp32_refusal: ([why]) => `Left in binary64: ${why}`,
  fused_multiply_adds: () => 'The plan holds fused multiply-adds.',
  helpers: ([text]) => text,
  local_doubles: ([text]) => text,
  loop_target: ([text]) => text,
  numerics: ([text]) => text,
  numerics_not_applicable: ([text]) => `Not applied (the family has no numerical policy): ${text}`,
  row_blocking: () => 'The plan blocked row loops around a vector loop.',
  ssa_fp32_refusal: ([why]) => `Left in binary64: ${why}`,
  ssa_target: ([text]) => text,
  vectorization: ([text]) => text,
  working_precision: ([text]) => text,
};

const tagText = (table, tag) => {
  const data = Array.isArray(tag?.data) ? tag.data.map(String) : [];
  const words = table[tag?.kind];
  // A kind this page has no sentence for is still shown, never dropped.
  return words ? words(data) : [String(tag?.kind), ...data].join(': ');
};
export const refusalText = (tag) => tagText(REFUSALS, tag);
export const noteText = (tag) => tagText(NOTES, tag);
export const isUnknownNode = (artifact) =>
  artifact?.body?.kind === 'refused' && artifact.body.refusal?.kind === 'unknown_node';

function renderBody(column, status) {
  if (status.state === 'loading') { column.append(el('p', 'Lowering…', 'note')); return; }
  if (status.state === 'failed') { column.append(el('p', 'Lowering failed; the reason is reported above.', 'code-refusal')); return; }
  const { body, notes } = status.artifact;
  if (body.kind === 'refused') {
    column.append(el('p', refusalText(body.refusal), 'code-refusal'));
  } else {
    const pre = el('pre', null, 'code-text');
    pre.append(el('code', body.text));
    column.append(pre);
    if (body.truncated) {
      column.append(el('p',
        `This artifact is longer than the size limit; the text above is cut after ${new TextEncoder().encode(body.text).length} bytes.`,
        'code-truncated'));
    }
  }
  if (notes.length > 0) {
    const list = el('ul', null, 'code-notes');
    for (const note of notes) list.append(el('li', noteText(note)));
    column.append(list);
  }
}

function renderColumn(model, side, handlers) {
  const status = model[side];
  const column = el('section', null, 'code-column');
  column.dataset.side = side;
  const current = familyOf(status.spelling);
  const tabs = el('div', null, 'code-tabs');
  tabs.setAttribute('role', 'tablist');
  for (const family of model.vocabulary) {
    const tab = el('button', family.name, 'code-tab');
    tab.type = 'button';
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-selected', String(family.name === current));
    tab.addEventListener('click', () => handlers.onSelect(side, family.name));
    tabs.append(tab);
  }
  column.append(tabs);

  const family = model.vocabulary.find((f) => f.name === current);
  const choices = family ? [...family.variants] : [];
  // A spelling that is valid but not offered (a numerics policy typed into the
  // URL) must still be what the bar shows, not silently another variant.
  if (!choices.includes(status.spelling)) choices.push(status.spelling);
  const bar = el('select', null, 'code-variant');
  bar.setAttribute('aria-label', `${side} variant`);
  for (const choice of choices) {
    const option = el('option', choice);
    option.value = choice;
    bar.append(option);
  }
  bar.value = status.spelling;
  bar.addEventListener('change', () => handlers.onSelect(side, bar.value));
  column.append(bar);
  renderBody(column, status);
  return column;
}

/* model: { vocabulary, node, left, right|null, diff|null } where a side is
 * { spelling, state: 'loading'|'ready'|'failed', artifact? }. */
export function renderCodePane(root, model, handlers) {
  root.replaceChildren();
  if (!model) { root.hidden = true; return; }
  root.hidden = false;
  root.classList.toggle('code-pane--compare', model.right !== null);

  const head = el('div', null, 'code-head');
  head.append(el('h3', `Lowering of ${model.node}`));
  const compare = el('button', model.right ? 'Close comparison' : 'Compare', 'code-compare');
  compare.type = 'button';
  compare.addEventListener('click', () => handlers.onCompare(model.right === null));
  const close = el('button', 'Close', 'code-close');
  close.type = 'button';
  close.addEventListener('click', () => handlers.onClose());
  head.append(compare, close);
  root.append(head);

  const columns = el('div', null, 'code-columns');
  columns.append(renderColumn(model, 'left', handlers));
  if (model.right) columns.append(renderColumn(model, 'right', handlers));
  root.append(columns);

  if (model.right) {
    const diff = el('section', null, 'code-diff');
    diff.append(el('h4', `Diff: ${model.left.spelling} → ${model.right.spelling}`));
    if (model.diff === null) {
      diff.append(el('p', 'A comparison needs the text of both artifacts.', 'note'));
    } else if (model.diff === '') {
      diff.append(el('p', 'The two artifacts are identical.', 'note'));
    } else {
      const pre = el('pre', null, 'code-diff-text');
      for (const line of model.diff.split('\n')) {
        const row = el('span', line + '\n', line.startsWith('+') ? 'diff-add' : line.startsWith('-') ? 'diff-del' : line.startsWith('@') ? 'diff-hunk' : null);
        pre.append(row);
      }
      diff.append(pre);
    }
    root.append(diff);
  }
}
