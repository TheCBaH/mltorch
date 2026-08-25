const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const SHA256 = /^[0-9a-f]{64}$/;

function fail(message) { throw new Error(message); }

function decimal(value, field) {
  if (typeof value !== 'string' || !/^(0|[1-9][0-9]*)$/.test(value)) fail(`invalid ${field}`);
  return BigInt(value);
}

/* A model's capability signal from the payload-free support sweep (see
 * `presentation.js`'s `stageSupport`), or `null` when the model was not in
 * that sweep. `native4dConverts`/`kernelConverts` are `true`/`false`/`null` --
 * `null` only when Native import itself already failed, per
 * `.ai/pt2_model_support.md`'s `resolve_branch` -- and the blockers are prose,
 * so only their presence and type are checked here, never their content. */
function validateSupport(raw) {
  if (raw === null || raw === undefined) return null;
  const boolOrNull = (v) => v === null || typeof v === 'boolean';
  const stringOrNull = (v) => v === null || typeof v === 'string';
  if (!raw || typeof raw.nativeBuilds !== 'boolean' ||
      !boolOrNull(raw.native4dConverts) || !stringOrNull(raw.native4dBlocker) ||
      !boolOrNull(raw.kernelConverts) || !stringOrNull(raw.kernelBlocker)) fail('invalid catalogue model support');
  return Object.freeze({
    nativeBuilds: raw.nativeBuilds,
    native4dConverts: raw.native4dConverts, native4dBlocker: raw.native4dBlocker,
    kernelConverts: raw.kernelConverts, kernelBlocker: raw.kernelBlocker,
  });
}

export function validateCatalog(raw) {
  if (!raw || raw.schemaVersion !== 1 || typeof raw.sourceRef !== 'string' ||
      typeof raw.defaultModel !== 'string' || !Array.isArray(raw.models)) fail('invalid catalogue');
  let previous = '';
  const models = raw.models.map((model) => {
    if (!model || !ID.test(model.id) || typeof model.displayName !== 'string' ||
        typeof model.url !== 'string' || !/^[^/:?#][^?#]*$/.test(model.url) ||
        !SHA256.test(model.sha256)) fail('invalid catalogue model');
    const bytes = decimal(model.bytes, 'model bytes');
    const support = validateSupport(model.support ?? null);
    if (model.id <= previous) fail('catalogue models must be unique and sorted');
    previous = model.id;
    return Object.freeze({ ...model, bytes, support });
  });
  if (!models.some((m) => m.id === raw.defaultModel)) fail('unknown default model');
  return Object.freeze({ sourceRef: raw.sourceRef, defaultModel: raw.defaultModel, models });
}

async function sha256Hex(buffer, crypto) {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', buffer));
  return Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
}

async function readBounded(response, maximum, signal) {
  const length = response.headers.get('content-length');
  if (length !== null && (!/^(0|[1-9][0-9]*)$/.test(length) || BigInt(length) > maximum)) {
    fail('catalogue response is too large');
  }
  if (!response.body) {
    if (length === null) fail('streaming response unavailable without a length');
    return response.arrayBuffer();
  }
  const reader = response.body.getReader();
  const chunks = [];
  let count = 0n;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    count += BigInt(value.byteLength);
    if (count > maximum) { await reader.cancel(); signal?.throwIfAborted?.(); fail('catalogue response is too large'); }
    chunks.push(value);
  }
  const bytes = new Uint8Array(Number(count));
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return bytes.buffer;
}

export class SourceStore {
  #source = null;
  constructor({ fetch: fetchImpl = fetch.bind(globalThis), crypto: cryptoImpl = crypto, hard }) {
    this.fetch = fetchImpl;
    this.crypto = cryptoImpl;
    this.hard = hard;
  }
  get source() { return this.#source; }
  clear() { this.#source = null; }
  setLocal(file) {
    if (!(file instanceof Blob)) fail('choose a local file');
    if (BigInt(file.size) > BigInt(this.hard.maxJsonBytes)) fail('local file is too large');
    const name = String(file.name || 'model.json').slice(0, 128);
    this.#source = { kind: 'local', name, blob: file, bytes: BigInt(file.size) };
    return this.#source;
  }
  async fetchCatalog(entry, { signal } = {}) {
    if (entry.bytes > BigInt(this.hard.maxJsonBytes)) fail('catalogue model is too large');
    const response = await this.fetch(new URL(entry.url, document.baseURI), { signal, credentials: 'same-origin' });
    if (!response.ok) fail(`model fetch failed (${response.status})`);
    const buffer = await readBounded(response, entry.bytes, signal);
    if (BigInt(buffer.byteLength) !== entry.bytes) fail('catalogue model length mismatch');
    if ((await sha256Hex(buffer, this.crypto)) !== entry.sha256) fail('catalogue model digest mismatch');
    this.#source = {
      kind: 'catalog', name: entry.displayName, blob: new Blob([buffer], { type: 'application/json' }), bytes: entry.bytes,
      catalog: { url: entry.url, ref: entry.sourceRef, verifiedSha256: entry.sha256 },
    };
    return this.#source;
  }
  async freshBytes() {
    if (!this.#source) fail('no source selected');
    const bytes = await this.#source.blob.arrayBuffer();
    if (BigInt(bytes.byteLength) !== this.#source.bytes) fail('source changed while reading');
    return bytes;
  }
}
