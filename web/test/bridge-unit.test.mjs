/* The jsoo bridge, exercised as the browser actually calls it.
 *
 * `test/model_explorer/me_request_test.ml` links `model_explorer_export`, not
 * this executable, so it can check `Options.create` and the `Me_request` codec
 * but never the `Js.Unsafe` decoder in front of them, its `options` echo, or a
 * malformed raw JavaScript value. Those are exactly what a browser can send, so
 * they are checked here against the real built artifact rather than a
 * reimplementation of it.
 *
 * Gated on `make webapp.build`: it reads the dune output, which the plain
 * `webapp.runtest` target deliberately does not produce.
 */

import assert from 'node:assert/strict';
import test from 'node:test';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const bundle = './_build/default/js/webapp/webapp_bridge.bc.js';
const loaded = createRequire(`${root}/`)(bundle);
/* `Js.export` targets `module.exports` when there is one and the global
 * otherwise; which it picks is jsoo's business, not this suite's. */
const mltorch = loaded?.mltorch ?? globalThis.mltorch;

const ID = '0f8fad5b-d9cb-469f-a165-70867728950e-0';
const LOCAL = { name: 'model.json', bytes: '1024', kind: 'local' };
const ALL = ['source', 'initial_native', 'canonical', 'native4d', 'stage_program', 'kernel', 'fusion'];

const build = (options, source = LOCAL, id = ID) =>
  mltorch.request.buildSession({ id, source, options, limits: {} });

/* The decoded request is what the worker will see; the echo is what the page
 * is told it asked for. A test that checked only one of them could not catch
 * the two disagreeing, which is the whole point of echoing. */
const decoded = (result) => JSON.parse(new TextDecoder().decode(result.request)).options;

function accepted(options) {
  const result = build(options);
  assert.equal(result.ok, true, result.error);
  return result;
}

function rejected(options, expected) {
  const result = build(options);
  assert.equal(result.ok, false, 'expected a rejection');
  assert.equal(result.error, expected);
}

test('the bridge is exported and carries its hard limits', () => {
  assert.deepEqual(Object.keys(mltorch).sort(), ['hard', 'request', 'session']);
  assert.ok(Number.isInteger(mltorch.hard.maxQuarantinedElements));
});

/* ------------------------------------------------------------ the defaults */

test('an absent options object reproduces the pre-options request exactly', () => {
  for (const options of [undefined, {}]) {
    const result = accepted(options);
    assert.deepEqual(decoded(result).stages, ALL);
    assert.equal(decoded(result).fold, false);
    assert.equal(decoded(result).verifySymbolic, undefined);
    assert.deepEqual(result.options, { stages: ALL, fold: false, verifySymbolic: null });
  }
});

/* ------------------------------------------------------------ valid values */

test('every effort crosses intact, and null is off', () => {
  for (const effort of ['quick', 'standard', 'thorough']) {
    const result = accepted({ verifySymbolic: effort });
    assert.equal(decoded(result).verifySymbolic, effort);
    assert.equal(result.options.verifySymbolic, effort);
  }
  const off = accepted({ verifySymbolic: null });
  assert.equal(decoded(off).verifySymbolic, undefined);
  assert.equal(off.options.verifySymbolic, null);
});

/* Normalisation lives in `Options.create`, which filters `all_stages` by
 * membership. The page cannot derive that, which is why the echo exists. */
test('stages are deduplicated and reordered, and the echo says so', () => {
  const result = accepted({ stages: ['kernel', 'source', 'kernel', 'canonical'] });
  assert.deepEqual(decoded(result).stages, ['source', 'canonical', 'kernel']);
  assert.deepEqual(result.options.stages, ['source', 'canonical', 'kernel']);
});

/* No source this UI accepts can turn folding on, but the wire is closed
 * regardless -- so the value is checked here rather than left untested until
 * PT2 sources land. */
test('fold crosses the wire even though no browser source can enable it', () => {
  const result = accepted({ fold: true });
  assert.equal(decoded(result).fold, true);
  assert.equal(result.options.fold, true);
});

test('a catalogue origin carries its provenance', () => {
  const result = build({}, {
    name: 'resnet18', bytes: '10', kind: 'catalog',
    catalog: { url: 'models/resnet18.json', ref: 'abc', verifiedSha256: 'a'.repeat(64) },
  });
  assert.equal(result.ok, true, result.error);
});

/* --------------------------------------------------------- the closed edge */

test('an unknown option value is rejected by name, never coerced', () => {
  rejected({ stages: ['nope'] }, 'unknown output stage "nope"');
  rejected({ verifySymbolic: 'deep' }, 'unknown verification effort "deep"');
});

test('a malformed option type is rejected by name, never coerced', () => {
  rejected(null, 'options must be an object');
  rejected(5, 'options must be an object');
  rejected({ stages: 'source' }, 'stages must be an array');
  rejected({ stages: [1] }, 'every stages entry must be a string');
  rejected({ fold: 'yes' }, 'fold must be a boolean');
  rejected({ verifySymbolic: 7 }, 'verifySymbolic must be a string');
});

/* `Options.create` refuses an empty list itself; the bridge must let that
 * reach the caller rather than substituting a default for it. */
test('an empty stage list is refused rather than defaulted', () => {
  rejected({ stages: [] }, 'a request must ask for at least one stage');
});

test('the source is read as strictly as the options', () => {
  const bad = (source, expected) => {
    const result = build({}, source);
    assert.equal(result.ok, false, 'expected a rejection');
    assert.equal(result.error, expected);
  };
  /* Every non-"catalog" kind used to be silently read as local, so a typo
   * produced a request with no provenance instead of a rejection. */
  bad({ ...LOCAL, kind: 'remote' }, 'unknown source kind "remote"');
  bad({ name: 'm', bytes: '10' }, 'source.kind must be a string');
  bad({ name: 'm', bytes: '10', kind: 'catalog' }, 'source.catalog must be an object');
  bad({ ...LOCAL, bytes: 'abc' }, 'source.bytes must be a decimal string');
  bad({ ...LOCAL, bytes: 1024 }, 'source.bytes must be a string');
});

/* The reason survives the boundary. Before this the bridge unwrapped every
 * failure with a null printer and reported `Printexc.to_string`, so a rejected
 * request said nothing about what was wrong with it. */
test('a request-domain failure reports its own reason, not an opaque exception', () => {
  const result = build({}, LOCAL, 'not-a-uuid');
  assert.equal(result.ok, false);
  assert.equal(result.error, 'malformed request id');

  const digest = build({}, {
    name: 'm', bytes: '10', kind: 'catalog',
    catalog: { url: 'u', ref: 'r', verifiedSha256: 'zz' },
  });
  assert.equal(digest.ok, false);
  assert.equal(digest.error, 'source digest is not 64 lowercase hex bytes');
});
