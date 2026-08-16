const text = (error) => (error && error.message) || String(error);
const reloadText = (reason) => `${reason}. Reload the page to continue.`;
const POISONED = reloadText('the application is in an unknown state');

/* A finalize failure always means the bridge accepted a session the page is not
 * showing. None of these texts claims the commit was rolled back, because it
 * was not. */
const FINALIZE_MESSAGES = {
  known_old: 'the model was installed but the view could not be switched to it',
  cancelled: 'the model was installed but its prepared view had already been discarded',
  inconsistent: 'the model was installed but the view is in an unknown state',
  misuse: 'the model was installed but the renderer rejected the view it had prepared',
};
const finalizeMessage = (result) => {
  const base = FINALIZE_MESSAGES[result?.state] ?? 'the model was installed but could not be shown';
  return result?.error ? `${base} (${result.error})` : base;
};
const detail = (marked) => (marked?.ok === false && marked?.error ? ` (${marked.error})` : '');

export class Coordinator {
  #pending = null;
  #epoch = 0;
  #poisoned = false;

  constructor({ workerFactory, bridge, render, onStatus = () => {}, onError = () => {}, hard }) {
    Object.assign(this, { workerFactory, bridge, render, onStatus, onError, hard });
  }

  /* Authority. Everything user-visible and everything that touches the bridge is
   * gated on this, `#poisoned` included: that is what makes "no already-running
   * continuation can reach commit after a fatal failure" true by construction. */
  #isCurrent(pending) {
    return this.#pending === pending
      && pending.epoch === this.#epoch
      && !pending.cancelled
      && !this.#poisoned;
  }

  cancel() {
    const pending = this.#pending;
    this.#epoch += 1;
    this.#pending = null;
    try {
      if (pending) { pending.cancelled = true; this.#terminal(pending, {}); }
      this.#cancelRenderer();
    } finally {
      if (pending) this.#settleToken(pending, 'abort');
    }
  }

  async load(source) {
    if (this.#poisoned) throw new Error(POISONED);
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
    const pending = this.#pending = {
      epoch, id, worker, terminal: false, cancelled: false, retired: false,
      token: null, tokenSettled: false,
    };
    worker.onmessage = (event) => this.#message(pending, event);
    worker.onerror = () => this.#terminal(pending, { message: 'worker failed' });
    const bytes = await source.readBytes();
    if (this.#pending !== pending) return;
    worker.postMessage({ request: built.request, bytes }, [built.request, bytes]);
    this.onStatus('Loading model…');
  }

  /* ------------------------------------------------------------ settlement */

  /* Settles ONE record: detach, report, retire. `#fatal` is the separate path
   * for declaring the whole application unusable. */
  #terminal(pending, { message } = {}) {
    const owned = this.#pending === pending;            // captured BEFORE detaching
    if (owned) this.#pending = null;
    // A terminal error is a true statement about this record, so it is emitted
    // before retirement: delivery must not depend on whether a callback
    // re-entered during teardown. Success is the opposite case and waits for
    // `undisturbed` below.
    if (owned && message) this.onError(String(message));
    const epoch = this.#epoch;
    const retired = this.#retireWorker(pending);        // fallible, and RE-ENTRANT
    const undisturbed = owned && this.#epoch === epoch;
    return { owned, retired, undisturbed };
  }

  #retireWorker(pending) {
    if (!pending || pending.retired) return false;
    pending.retired = true;
    try { pending.worker.terminate(); return true; }
    catch (error) { console.warn('[coordinator] worker termination failed', error); return false; }
  }

  /* Exactly-once bridge settlement. `tokenSettled` is set BEFORE the call, so a
   * throw cannot leave the token eligible for a second attempt; `threw` and a
   * returned `{ok:false}` stay distinguishable. */
  #settleToken(pending, how) {
    if (!pending || !pending.token || pending.tokenSettled) {
      return { attempted: false, ok: true, threw: false, error: null };
    }
    pending.tokenSettled = true;
    try {
      const result = how === 'commit'
        ? this.bridge.session.commit(pending.token)
        : this.bridge.session.abort(pending.token);
      return { attempted: true, ok: !!result?.ok, threw: false, error: result?.error ?? null };
    } catch (error) {
      return { attempted: true, ok: false, threw: true, error: text(error) };
    }
  }

  #cancelRenderer() {
    try { this.render.cancel?.(); }
    catch (error) { console.warn('[coordinator] renderer cancellation failed', error); }
  }

  /* The ONE fatal path. A failure can prove the application unusable while
   * `#pending` holds some other request, so this never settles "its" record --
   * it settles whatever is current, and detaches that victim BEFORE the first
   * call that could run somebody else's JavaScript. */
  #fatal(reason, { initiating = null } = {}) {
    const first = !this.#poisoned;
    this.#poisoned = true;                              // refuses every later load()
    this.#epoch += 1;                                   // invalidates every outstanding record
    const victim = this.#pending;                       // may differ from `initiating`
    this.#pending = null;
    // From here nothing re-entrant can move the victim out from under us: a
    // re-entrant cancel() finds #pending === null, and a re-entrant #fatal sees
    // first === false.
    if (initiating && initiating !== victim) this.#retireWorker(initiating);
    if (victim) {
      this.#settleToken(victim, 'abort');               // no-op if it already committed
      this.#retireWorker(victim);
    }
    this.#cancelRenderer();
    if (first) this.onError(reloadText(reason));        // exactly one, ever
  }

  /* Every `abortReady` result is consumed as a value; none is ignored. */
  #consumeAbort(pending, result, { silent }) {
    if (result?.ok) return { fatal: false, note: '' };  // includes already:'cancelled'
    if (result?.state === 'cleanup_failed') {
      return { fatal: false, note: silent ? '' : ' (the abandoned view could not be removed)' };
    }
    const reason = result?.state === 'misuse'
      ? 'the renderer rejected a view it had prepared'
      : (result?.error || 'the renderer cannot continue');
    this.#fatal(reason, { initiating: pending });
    return { fatal: true, note: '' };
  }

  /* ------------------------------------------------------------- protocol */

  async #message(pending, event) {
    try {
      await this.#handle(pending, event);
    } catch (error) {
      // Nothing in `#handle` is expected to throw -- every fallible call is
      // contained -- so reaching here is a defect, and an unknown state.
      this.#fatal(`the page failed unexpectedly: ${text(error)}`, { initiating: pending });
    }
  }

  async #handle(pending, event) {
    if (!this.#isCurrent(pending) || pending.terminal) return;
    const frame = event.data;
    if (!frame || typeof frame.meta !== 'string' || new TextEncoder().encode(frame.meta).byteLength > this.hard.maxResponseMetaBytes) return this.#terminal(pending, { message: 'malformed worker metadata' });
    let meta;
    try { meta = JSON.parse(frame.meta); } catch { return this.#terminal(pending, { message: 'malformed worker metadata' }); }
    const terminal = new Set(['session', 'failed', 'protocol_failure', 'delta']);
    if (!['progress', ...terminal].includes(meta.kind) || (meta.id && meta.id !== pending.id)) return this.#terminal(pending, { message: 'unexpected worker response' });
    if (meta.kind === 'progress') { this.onStatus(`${meta.phase}: ${meta.done}${meta.total ? `/${meta.total}` : ''}`); return; }
    pending.terminal = true;
    if (meta.kind !== 'session') return this.#terminal(pending, { message: meta.kind === 'delta' ? 'unexpected detail response' : (meta.error?.message || 'model export failed') });
    if (!(frame.payload instanceof ArrayBuffer) || frame.payload.byteLength !== meta.bytes || frame.payload.byteLength > this.hard.maxResponseDocumentBytes) return this.#terminal(pending, { message: 'invalid session payload' });
    let documentText;
    try { documentText = new TextDecoder('utf-8', { fatal: true }).decode(frame.payload); } catch { return this.#terminal(pending, { message: 'session is not UTF-8' }); }
    const prepared = this.bridge.session.prepare(pending.id, frame.meta, documentText);
    if (!prepared.ok) return this.#terminal(pending, { message: prepared.error || 'invalid session' });

    pending.token = prepared.token;                     // stored BEFORE any await
    let handle;
    try {
      handle = await this.render.install(prepared.render);
    } catch (error) {
      this.#settleToken(pending, 'abort');
      // Classify by kind BEFORE applying the stale rule: staleness decides
      // whether to speak about one request, but a terminal renderer is a fact
      // about the whole application.
      if (error?.kind === 'inconsistent') {
        return this.#fatal(error.message || 'the renderer cannot continue', { initiating: pending });
      }
      // A valid `cancelled` always comes with lost authority, because
      // Coordinator.cancel() detaches #pending and bumps the epoch before it
      // cancels the renderer. Suppression is on that proof, not on the kind:
      // a `cancelled` for a still-current record is a protocol mismatch, and
      // returning silently would leave the record in #pending forever.
      if (!this.#isCurrent(pending)) return;
      return this.#terminal(pending, {
        message: error?.kind === 'cancelled'
          ? 'the renderer cancelled a request that was still current'
          : (error?.message || 'renderer failed'),
      });
    }
    if (!this.#isCurrent(pending)) {                    // stale: silent, but still consumed
      this.#consumeAbort(pending, this.render.abortReady(handle), { silent: true });
      return this.#settleToken(pending, 'abort');
    }

    const committed = this.#settleToken(pending, 'commit');
    if (committed.threw) {
      // UNKNOWN bridge state: `webapp_bridge.ml` clears `staged` before it
      // builds its response, so a throw can genuinely land after the state
      // transition. Nothing is removed and no rollback is claimed.
      const marked = this.render.markInconsistent('session state unknown after commit');
      return this.#fatal(`the session may or may not have been installed${detail(marked)}`,
        { initiating: pending });
    }
    if (!committed.ok) {                                // KNOWN rejection: rollback is real
      const aborted = this.#consumeAbort(pending, this.render.abortReady(handle), { silent: false });
      if (aborted.fatal) return;                        // #fatal already settled and reported
      return this.#terminal(pending, {
        message: `${committed.error || 'session commit failed'}${aborted.note}`,
      });
    }
    // Commit CONFIRMED. Losing authority now is not an ordinary stale result:
    // the bridge already holds the new session, so anything but a clean
    // finalize means bridge and presentation disagree.
    if (!this.#isCurrent(pending)) {
      return this.#fatal('the session was installed after the request was superseded',
        { initiating: pending });
    }
    const finalized = this.render.finalize(handle);
    if (!finalized.ok) return this.#fatal(finalizeMessage(finalized), { initiating: pending });
    const settled = this.#terminal(pending, {});        // detach + terminate, no error
    if (settled.undisturbed) this.onStatus('Model loaded');
  }
}
