/* The Stage 0A driver.
 *
 * Every function here answers one question the coordinator design currently
 * ASSUMES, and records the answer on `window.__spike` for the Playwright spec
 * to read. It is deliberately not a library: nothing in web/src is meant to
 * survive into the application unchanged, because what survives is the answers.
 */

const MOUNT = document.getElementById('mount');
const answers = {};
window.__spike = { answers, log: [] };

function note(k, v) {
  answers[k] = v;
  window.__spike.log.push(`${k}=${JSON.stringify(v)}`);
}

async function fetchJson(p) {
  const r = await fetch(p);
  if (!r.ok) throw new Error(`${p}: ${r.status}`);
  return r.json();
}

/* Q1: does `modelGraphProcessed` fire reliably, and can it be given a timeout?
 *
 * The whole install protocol is built on awaiting it. If it can be missed, the
 * transaction wedges; if it cannot be bounded, a hostile or broken payload
 * hangs the page. Both halves are the question. */
function awaitProcessed(el, ms) {
  return new Promise((resolve) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (!settled) { settled = true; resolve({ fired: false, timedOut: true }); }
    }, ms);
    el.addEventListener('modelGraphProcessed', () => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        resolve({ fired: true, timedOut: false });
      }
    }, { once: true });
  });
}

function makeElement(collections, config, uiState) {
  const el = document.createElement('model-explorer-visualizer');
  el.graphCollections = collections;
  if (config) el.config = config;
  if (uiState) el.initialUiState = uiState;
  return el;
}

async function run() {
  /* Q0: does the global survive the visualizer bundle loading over it? The
   * bundle reads `window.modelExplorer.assetFilesBaseUrl` at element
   * construction, so a bundle that replaced the object would silently fall
   * back to the defaults -- which is indistinguishable from the override
   * being ignored, and only one of those has a fix. */
  note('globalConfig', {
    setBeforeBundle: window.__spikeConfigBefore,
    setAfterBundle: window.__spikeConfigAfter,
    now: JSON.stringify(window.modelExplorer ?? null),
  });

  const collections = await fetchJson('collection.json');
  const detail = await fetchJson('detail.json');

  /* Q2: can an element be built OFF-DOM and swapped in?
   *
   * §6.1.1 requires it: it is what puts every failable step BEFORE the commit
   * and leaves only replaceChild after it. If the element only works once
   * connected, the fallback is a snapshot/restore of every installed property,
   * and that decision has to be made here rather than assumed. */
  const off = makeElement(collections, { defaultGraphId: 'g/flow' });
  const builtOffDom = !off.isConnected;
  const processed = awaitProcessed(off, 15000);
  MOUNT.appendChild(off);           // attach AFTER properties were set
  const r1 = await processed;
  note('offDomBuildThenAttach', { builtOffDom, ...r1 });

  /* Q3: config REPLACEMENT versus element recreation.
   *
   * If assigning `config` on a live element takes effect, the transaction can
   * mutate in place and never needs two elements alive at once -- which is the
   * largest term in the response allocation peak. */
  const before = off.config;
  off.config = { ...off.config, nodeLabelsToHide: ['bias'] };
  note('configAssignable', {
    accepted: off.config !== before,
    readBack: JSON.stringify(off.config),
  });

  /* Q4: installing a NEW graph and a parent subgraphIds link AFTER
   * modelGraphProcessed, then entering it and returning.
   *
   * This is exactly the expression-detail path: the delta arrives late, adds a
   * graph nobody has seen, and links it from a node already on screen. */
  const withDetail = structuredClone(collections);
  withDetail[0].graphs.push(structuredClone(detail[0].graphs[0]));
  const parent = withDetail[0].graphs.find((g) => g.id === 'g/native/000');
  const target = parent.nodes.find((n) => n.id === 'n3');
  target.subgraphIds = ['expr/g/native/000/n3/t0'];

  const staged = makeElement(withDetail, { defaultGraphId: 'g/native/000' });
  const stagedProcessed = awaitProcessed(staged, 15000);
  /* The captured native replaceChild -- the coordinator uses a captured
   * reference so a page that overwrites Node.prototype.replaceChild cannot
   * intercept the one operation the transaction depends on. */
  const replaceChild = Node.prototype.replaceChild;
  replaceChild.call(MOUNT, staged, off);
  const r4 = await stagedProcessed;
  note('lateGraphInstall', { ...r4, mountChildren: MOUNT.children.length });

  /* Q5: subgraphIds navigation, and returning. */
  let navigated = false;
  try {
    staged.selectNode('n3', 'g/native/000', 'mltorch:spike');
    navigated = true;
  } catch (e) {
    note('selectNodeThrew', String(e));
  }
  note('subgraphNavigation', { navigated });

  /* Q6: node data through BOTH provider entry points. */
  const nodeData = {
    results: { n1: { value: 3 }, n2: { value: 7 } },
    gradient: [
      { stop: 0, bgColor: '#fff' },
      { stop: 1, bgColor: '#f00' },
    ],
  };
  const providers = {};
  try {
    staged.addNodeDataProviderData('cost', { 'g/native/000': nodeData });
    providers.byGraphId = true;
  } catch (e) { providers.byGraphId = String(e); }
  try {
    staged.addNodeDataProviderDataWithGraphIndex('cost2', {
      'mltorch:spike': { 0: nodeData },
    });
    providers.byGraphIndex = true;
  } catch (e) { providers.byGraphIndex = String(e); }
  note('nodeDataProviders', providers);

  /* Q7: uiStateChanged -> initialUiState round trip. This is how "back to
   * flow" restores the pane the user left. */
  const uiState = await new Promise((resolve) => {
    const t = setTimeout(() => resolve(null), 5000);
    staged.addEventListener('uiStateChanged', (ev) => {
      clearTimeout(t);
      resolve(ev.detail ?? null);
    }, { once: true });
    staged.selectNode('n1', 'g/native/000', 'mltorch:spike');
  });
  note('uiStateChanged', { fired: uiState !== null });

  if (uiState) {
    const restored = makeElement(collections, { defaultGraphId: 'g/flow' }, uiState);
    const rp = awaitProcessed(restored, 15000);
    replaceChild.call(MOUNT, restored, staged);
    const r7 = await rp;
    note('initialUiStateAccepted', r7);
  }

  window.__spike.done = true;
}

run().catch((e) => {
  window.__spike.error = String(e && e.stack ? e.stack : e);
  window.__spike.done = true;
});
