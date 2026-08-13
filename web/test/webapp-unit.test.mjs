import assert from 'node:assert/strict';
import test from 'node:test';
import { validateCatalog } from '../app/source_store.js';
import { Coordinator } from '../app/coordinator.js';

const digest = 'a'.repeat(64);
const catalogue = { schemaVersion: 1, sourceRef: 'a'.repeat(40), defaultModel: 'a', models: [
  { id: 'a', displayName: 'A', bytes: '1', sha256: digest, url: 'models/a.json' },
] };

test('catalogue validation requires sorted, digested entries', () => {
  assert.equal(validateCatalog(catalogue).models[0].bytes, 1n);
  assert.throws(() => validateCatalog({ ...catalogue, models: [{ ...catalogue.models[0], sha256: 'A'.repeat(64) }] }));
  assert.throws(() => validateCatalog({ ...catalogue, defaultModel: 'missing' }));
});

test('coordinator transfers fresh request and source bytes exactly once', async () => {
  const worker = { postMessage(...args) { this.args = args; }, terminate() {} };
  const bridge = { request: { buildSession: () => ({ ok: true, request: new ArrayBuffer(2) }) }, session: {} };
  const coordinator = new Coordinator({ workerFactory: () => worker, bridge, render: {}, hard: {}, onStatus() {}, onError() {} });
  const bytes = new ArrayBuffer(3);
  await coordinator.load({ name: 'fixture.json', bytes: 3n, kind: 'local', readBytes: async () => bytes });
  assert.deepEqual(worker.args[0], { request: worker.args[0].request, bytes: worker.args[0].bytes });
  assert.equal(worker.args[1][0], worker.args[0].request);
  assert.equal(worker.args[1][1], bytes);
});
