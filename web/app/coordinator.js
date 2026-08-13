export class Coordinator {
  #pending = null;
  #epoch = 0;
  constructor({ workerFactory, bridge, render, onStatus = () => {}, onError = () => {}, hard }) {
    Object.assign(this, { workerFactory, bridge, render, onStatus, onError, hard });
  }
  cancel() {
    this.#epoch += 1;
    if (this.#pending) this.#pending.worker.terminate();
    this.#pending = null;
  }
  async load(source) {
    if (this.#pending) throw new Error('a request is already in flight');
    const epoch = ++this.#epoch;
    const id = `${crypto.randomUUID()}-0`;
    const requestSource = {
      name: source.name,
      bytes: source.bytes.toString(),
      kind: source.kind,
      ...(source.catalog ? { catalog: source.catalog } : {}),
    };
    const built = this.bridge.request.buildSession({ id, source: requestSource, options: {}, limits: {} });
    if (!built.ok) throw new Error(built.error || 'request construction failed');
    const worker = this.workerFactory();
    const pending = this.#pending = { epoch, id, worker, terminal: false };
    worker.onmessage = (event) => this.#message(pending, event);
    worker.onerror = () => this.#fail(pending, 'worker failed');
    const bytes = await source.readBytes();
    if (this.#pending !== pending) return;
    worker.postMessage({ request: built.request, bytes }, [built.request, bytes]);
    this.onStatus('Loading model…');
  }
  async #message(pending, event) {
    if (this.#pending !== pending || pending.epoch !== this.#epoch || pending.terminal) return;
    const frame = event.data;
    if (!frame || typeof frame.meta !== 'string' || new TextEncoder().encode(frame.meta).byteLength > this.hard.maxResponseMetaBytes) return this.#fail(pending, 'malformed worker metadata');
    let meta;
    try { meta = JSON.parse(frame.meta); } catch { return this.#fail(pending, 'malformed worker metadata'); }
    const terminal = new Set(['session', 'failed', 'protocol_failure', 'delta']);
    if (!['progress', ...terminal].includes(meta.kind) || (meta.id && meta.id !== pending.id)) return this.#fail(pending, 'unexpected worker response');
    if (meta.kind === 'progress') { this.onStatus(`${meta.phase}: ${meta.done}${meta.total ? `/${meta.total}` : ''}`); return; }
    pending.terminal = true;
    if (meta.kind !== 'session') return this.#fail(pending, meta.kind === 'delta' ? 'unexpected detail response' : (meta.error?.message || 'model export failed'));
    if (!(frame.payload instanceof ArrayBuffer) || frame.payload.byteLength !== meta.bytes || frame.payload.byteLength > this.hard.maxResponseDocumentBytes) return this.#fail(pending, 'invalid session payload');
    let documentText;
    try { documentText = new TextDecoder('utf-8', { fatal: true }).decode(frame.payload); } catch { return this.#fail(pending, 'session is not UTF-8'); }
    const prepared = this.bridge.session.prepare(pending.id, frame.meta, documentText);
    if (!prepared.ok) return this.#fail(pending, prepared.error || 'invalid session');
    try { await this.render.install(prepared.render); this.bridge.session.commit(prepared.token); this.#pending = null; this.onStatus('Model loaded'); }
    catch (error) { this.bridge.session.abort(prepared.token); this.#fail(pending, error.message || 'renderer failed'); }
  }
  #fail(pending, message) {
    if (this.#pending !== pending) return;
    pending.worker.terminate(); this.#pending = null; this.onError(String(message));
  }
}
