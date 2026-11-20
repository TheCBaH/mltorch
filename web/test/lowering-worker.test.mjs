/* The lowering worker, driven the way the page drives it: a Build_lowering
 * request built by the real bridge, posted to the real built worker, over the
 * committed mobilenetv2_050 model.json. Node 0's JavaScript is compared with
 * what the native `native_graph lower` CLI prints, so the browser worker and
 * the command line cannot drift apart.
 *
 * Gated on `make webapp.build` (it reads the dune output) and, for the CLI
 * comparison, on `dune build bin/native_graph.exe`; `webapp.bridge-runtest`
 * does both. */

import assert from 'node:assert/strict';
import test from 'node:test';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(`${root}/`);
const model = resolve(root, 'modules/devcontainer.pytorch-image-models/models/mobilenetv2_050/models/model.json');
const cli = resolve(root, '_build/default/bin/native_graph.exe');

const mltorch = (() => {
  const loaded = require('./_build/default/js/webapp/webapp_bridge.bc.js');
  return loaded?.mltorch ?? globalThis.mltorch;
})();

/* The worker talks through the global `postMessage` and `onmessage`. */
const replies = [];
globalThis.postMessage = (message) => replies.push(message);
require('./_build/default/js/webapp/lowering_worker.bc.js');

const bytes = readFileSync(model);
let counter = 0;
const ask = async (node, variant, dialect = 'native') => {
  const id = `0f8fad5b-d9cb-469f-a165-70867728950e-${counter++}`;
  const built = mltorch.request.buildLowering({
    id, options: {}, key: { dialect, node, variant },
    source: { name: 'model.json', bytes: String(bytes.length), kind: 'local' },
  });
  assert.equal(built.ok, true, built.error);
  const seen = replies.length;
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length);
  const started = performance.now();
  await globalThis.onmessage({ data: { request: built.request, bytes: copy } });
  const elapsed = performance.now() - started;
  const final = replies.slice(seen).map((r) => ({ meta: JSON.parse(r.meta), payload: r.payload }))
    .find((r) => r.meta.kind !== 'progress');
  assert.ok(final, 'the worker answered');
  assert.equal(final.meta.id, id);
  return { ...final, elapsed };
};
/* A whole-graph summary: one metric of every node, as the page asks for an overlay. */
const askSummary = async (metric, dialect = 'native') => {
  const id = `0f8fad5b-d9cb-469f-a165-70867728950e-${counter++}`;
  const built = mltorch.request.buildLoweringSummary({
    id, options: {}, summary: { dialect, metric },
    source: { name: 'model.json', bytes: String(bytes.length), kind: 'local' },
  });
  assert.equal(built.ok, true, built.error);
  const seen = replies.length;
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length);
  const started = performance.now();
  await globalThis.onmessage({ data: { request: built.request, bytes: copy } });
  const elapsed = performance.now() - started;
  const final = replies.slice(seen).map((r) => ({ meta: JSON.parse(r.meta), payload: r.payload }))
    .find((r) => r.meta.kind !== 'progress');
  assert.ok(final, 'the worker answered');
  assert.equal(final.meta.id, id);
  return { ...final, elapsed };
};
const artifactOf = (final) => JSON.parse(new TextDecoder().decode(final.payload));

test('node 0 of mobilenetv2_050 lowers to the JavaScript the CLI prints', async () => {
  const first = await ask(0, 'js');
  assert.equal(first.meta.kind, 'artifact', JSON.stringify(first.meta));
  const artifact = artifactOf(first);
  assert.equal(artifact.body.kind, 'text');
  assert.equal(artifact.body.language, 'js');
  assert.equal(artifact.body.truncated, false);
  const printed = execFileSync(cli, ['lower', '--model', model, '--node', '0', '--artifact', 'js'], { encoding: 'utf8' });
  assert.ok(printed.includes(artifact.body.text), 'the CLI prints the worker\'s text');
  console.log(`first request ${first.elapsed.toFixed(0)} ms`);
});

test('a second node of the same model is answered from the memoized bundle', async () => {
  const again = await ask(0, 'loop');
  assert.equal(again.meta.kind, 'artifact');
  const a = artifactOf(again);
  assert.equal(a.body.language, 'loop');
  const other = await ask(1, 'js');
  assert.equal(other.meta.kind, 'artifact');
  console.log(`memoized requests ${again.elapsed.toFixed(0)} ms, ${other.elapsed.toFixed(0)} ms`);
  assert.ok(again.elapsed < 1000 && other.elapsed < 1000, 'no canonical rebuild');
});

test('a node the model does not have is a typed refusal, not a failure', async () => {
  const none = await ask(999999, 'js');
  assert.equal(none.meta.kind, 'artifact');
  const a = artifactOf(none);
  assert.equal(a.body.kind, 'refused');
  assert.equal(a.body.refusal.kind, 'unknown_node');
});

test('Native4D nodes lower, and a family without a gate is a typed refusal', async () => {
  const c = await ask(415, 'c', 'native4d');
  assert.equal(c.meta.kind, 'artifact', JSON.stringify(c.meta));
  assert.equal(c.meta.lowering.dialect, 'native4d');
  const a = artifactOf(c);
  assert.equal(a.key.dialect, 'native4d');
  assert.equal(a.body.kind, 'text');
  const printed = execFileSync(cli, ['lower', '--model', model, '--dialect', 'native4d', '--node', '415', '--artifact', 'c'], { encoding: 'utf8' });
  assert.ok(printed.includes(a.body.text), 'the CLI prints the worker\'s native4d text');
  const coral = artifactOf(await ask(415, 'coral', 'native4d'));
  assert.equal(coral.body.kind, 'refused');
  assert.equal(coral.body.refusal.kind, 'unsupported_dialect');
  assert.deepEqual(coral.body.refusal.data, ['native4d', 'coral']);
  console.log(`native4d first request ${c.elapsed.toFixed(0)} ms`);
});

test('a summary request answers one metric of every node, timed', async () => {
  const timings = {};
  for (const metric of ['vector_loops', 'scalar_loops', 'coral_cycles']) {
    const r = await askSummary(metric);
    assert.equal(r.meta.kind, 'summary', JSON.stringify(r.meta));
    assert.deepEqual(r.meta.summary, { dialect: 'native', metric });
    const doc = JSON.parse(new TextDecoder().decode(r.payload));
    assert.equal(doc.metric, metric);
    const ids = Object.keys(doc.values);
    assert.ok(ids.length > 50 && ids.every((id) => /^n\d+$/.test(id)), 'canonical node ids');
    assert.ok(Object.values(doc.values).every((v) => v === null || typeof v === 'number'));
    assert.ok(Object.values(doc.values).some((v) => typeof v === 'number' && v > 0));
    timings[metric] = `${(r.elapsed / 1000).toFixed(1)}s`;
  }
  console.log(`summary timings (memoized bundle) ${JSON.stringify(timings)}`);
});

test('a Native4D summary has no Coral estimate', async () => {
  const r = await askSummary('coral_cycles', 'native4d');
  assert.equal(r.meta.kind, 'summary', JSON.stringify(r.meta));
  const doc = JSON.parse(new TextDecoder().decode(r.payload));
  assert.ok(Object.keys(doc.values).length > 50);
  assert.ok(Object.values(doc.values).every((v) => v === null));
  const loops = await askSummary('scalar_loops', 'native4d');
  const l = JSON.parse(new TextDecoder().decode(loops.payload));
  assert.ok(Object.values(l.values).some((v) => typeof v === 'number'));
  console.log(`native4d summary ${(loops.elapsed / 1000).toFixed(1)}s`);
});

test('the vocabulary names every metric, dialect and the families each dialect reaches', () => {
  assert.deepEqual([...mltorch.lowering.metrics()], ['coral_cycles', 'scalar_loops', 'vector_loops']);
  assert.deepEqual([...mltorch.lowering.dialects()], ['native', 'native4d']);
  const vocabulary = mltorch.lowering.vocabulary();
  const reached = (dialect) => vocabulary.filter((f) => f.dialects.includes(dialect)).map((f) => f.name);
  assert.deepEqual(reached('native4d'), ['c', 'js', 'loop', 'ssa', 'wat']);
  assert.equal(reached('native').length, vocabulary.length);
});
