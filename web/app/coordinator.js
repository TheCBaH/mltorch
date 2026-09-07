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

/* A presentation change installs no model and commits nothing, so none of the
 * texts above is true of it. These say only what happened: the view did not
 * change, and the one on screen is still the right view of the same document. */
const PRESENT_MESSAGES = {
  known_old: 'the view could not be switched; the current one is still shown',
  cancelled: 'the prepared view was discarded before it could be shown',
  misuse: 'the renderer rejected the view it had prepared',
};
const presentMessage = (result) => {
  const base = PRESENT_MESSAGES[result?.state] ?? 'the view could not be changed';
  return result?.error ? `${base} (${result.error})` : base;
};
const detail = (marked) => (marked?.ok === false && marked?.error ? ` (${marked.error})` : '');

export class Coordinator {
  #pending = null;
  #epoch = 0;
  #poisoned = false;
  /* What the bridge and the renderer produced, retained so a presentation
   * change can replay it. Nothing derived lives here: the page builds its own
   * index from `#session`, and the coordinator never reads one. A failed load
   * leaves the previous triple in place, so the visible view, the controls and
   * the URL keep describing one document. */
  #session = null;
  #source = null;
  #options = null;
  /* The closed presentation descriptor the renderer actually opened -- a single
   * view or a comparison. `view` is DERIVED from it rather than stored beside
   * it: two fields could disagree, and "which view is showing" has no answer
   * while a comparison is on screen. */
  #presentation = null;
  /* A display preference, not a request: it changes no options, no worker
   * message and no retained `#session` -- only which text `render.install`
   * is given. Defaults to the browser's own default so a coordinator built
   * before any URL is read already matches `presentation.js`'s. */
  #constantsMode = 'grouped';

  constructor({ workerFactory, bridge, render, onStatus = () => {}, onError = () => {}, onSession = () => {}, onDetail = () => {}, hard }) {
    Object.assign(this, { workerFactory, bridge, render, onStatus, onError, onSession, onDetail, hard });
  }

  get session() { return this.#session; }
  get effectiveOptions() { return this.#options; }
  get presentation() { return this.#presentation; }
  get constants() { return this.#constantsMode; }
  /* Records the mode with no re-present, for a caller about to `load()` a
   * different document anyway -- `setConstants` would otherwise re-present
   * the session this is REPLACING, wastefully and just before `cancel()`
   * tears it down. */
  set constants(mode) { this.#constantsMode = mode; }
  /** The single view on screen, or `null` -- including while a comparison is. */
  get view() {
    return this.#presentation?.kind === 'single' ? this.#presentation.view : null;
  }

  /* The text actually handed to the renderer: `#session` unchanged, or the
   * bridge's `groupConstants` transform of it. Called at every `render.install`
   * site rather than cached on `#session`, so the mode can change between two
   * presentations of the SAME retained document without a second bridge call
   * lingering on the wrong one. A rejection becomes a plain `Error` -- neither
   * `inconsistent` nor `cancelled` -- so it takes the ordinary "this attempt
   * failed, the previous view stands" path both call sites already have. */
  #installText(sessionText) {
    if (this.#constantsMode !== 'grouped') return sessionText;
    const result = this.bridge.session.groupConstants(sessionText);
    if (!result.ok) throw new Error(result.error || 'constant grouping failed');
    return result.render;
  }

  /* The renderer selection that reopens whatever is on screen right now, for
   * `setConstants` to replay through the ordinary `present()` path -- the
   * inverse of the descriptor `render.install` hands back as
   * `handle.presentation`. `undefined` (load with no selection) when nothing
   * is on screen yet, since `present()` itself already refuses that case. */
  #currentSelection() {
    const p = this.#presentation;
    if (!p) return undefined;
    if (p.kind === 'comparison') return { comparison: p.comparison };
    if (p.kind === 'flow') return { flow: p.view };
    return { view: p.view };
  }

  /* Changes the constants display mode and, if a session is already on
   * screen, re-presents it -- same graph, same view, through the existing
   * candidate lifecycle, no worker request. Before the first load this only
   * records the mode for the load to pick up. */
  async setConstants(mode) {
    this.#constantsMode = mode;
    if (!this.#session) return;
    return this.present(this.#currentSelection());
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

  /* `options` is the page's construction options, forwarded verbatim to the
   * bridge, which normalises them and echoes the result back; `prefer` is the
   * ordered view preference the renderer resolves against the arriving
   * document, so the first and only install already shows the right graph.
   *
   * Both default, but ONLY so the fixtures that assert the bridge's own
   * defaults keep asserting exactly that. Every app-originated load passes
   * options -- omitting them would silently rebuild with defaults and discard
   * whatever the user has selected. */
  async load(source, { options = {}, prefer = [] } = {}) {
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
    const built = this.bridge.request.buildSession({ id, source: requestSource, options, limits: {} });
    if (!built.ok) throw new Error(built.error || 'request construction failed');
    const worker = this.workerFactory();
    const pending = this.#pending = {
      epoch, id, worker, terminal: false, cancelled: false, retired: false,
      token: null, tokenSettled: false, prefer, options: built.options ?? null,
      kind: 'session', source,
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
    // A presentation record has no worker at all; retiring it is a no-op rather
    // than a caught throw, so `#terminal`'s `retired` stays meaningful.
    if (!pending.worker) return true;
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
    if (pending.kind === 'detail') return this.#handleDetail(pending, frame, meta);
    if (meta.kind !== 'session') return this.#terminal(pending, { message: meta.kind === 'delta' ? 'unexpected detail response' : (meta.error?.message || 'model export failed') });
    if (!(frame.payload instanceof ArrayBuffer) || frame.payload.byteLength !== meta.bytes || frame.payload.byteLength > this.hard.maxResponseDocumentBytes) return this.#terminal(pending, { message: 'invalid session payload' });
    let documentText;
    try { documentText = new TextDecoder('utf-8', { fatal: true }).decode(frame.payload); } catch { return this.#terminal(pending, { message: 'session is not UTF-8' }); }
    const prepared = this.bridge.session.prepare(pending.id, frame.meta, documentText);
    if (!prepared.ok) return this.#terminal(pending, { message: prepared.error || 'invalid session' });

    pending.token = prepared.token;                     // stored BEFORE any await
    let handle;
    try {
      handle = await this.render.install(this.#installText(prepared.render), { prefer: pending.prefer });
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
    // Retention lands BEFORE `#terminal`: it is pure state that cannot throw,
    // and it became true the instant the commit was confirmed. Only the
    // outward-facing callbacks wait on `undisturbed` below.
    this.#session = prepared.render;
    this.#source = pending.source;
    this.#options = pending.options;
    this.#presentation = handle.presentation ?? null;
    const settled = this.#terminal(pending, {});        // detach + terminate, no error
    // `undisturbed` is what makes a callback safe to hand control to: the
    // record is out of `#pending` and the epoch is unchanged, so a synchronous
    // `load()`, `present()` or `cancel()` from inside `onSession` meets the
    // same guards a fresh caller would, instead of the in-flight one.
    if (!settled.undisturbed) return;
    this.onSession({
      sessionText: this.#session,
      effectiveOptions: this.#options,
      viewId: this.view,
    });
    this.onStatus('Model loaded');
  }

  /* A detail is a second worker transaction over the retained source.  The
   * bridge validates and merges the delta before a candidate is made, so a
   * failed request leaves the parent document and its selected operator intact. */
  async openDetail(key) {
    if (this.#poisoned) throw new Error(POISONED);
    if (this.#pending) throw new Error('a request is already in flight');
    if (!this.#session || !this.#source || !this.#options) throw new Error('no model is loaded');
    const epoch = ++this.#epoch;
    const id = `${crypto.randomUUID()}-0`;
    const source = this.#source;
    const requestSource = {
      name: source.name, bytes: source.bytes.toString(), kind: source.kind,
      ...(source.catalog ? { catalog: source.catalog } : {}),
    };
    const built = this.bridge.request.buildDetail({ id, source: requestSource, options: this.#options, key });
    if (!built.ok) throw new Error(built.error || 'detail request construction failed');
    const worker = this.workerFactory();
    const detailId = `expr/${key.parentGraph}/n${key.node}`;
    const pending = this.#pending = {
      epoch, id, worker, terminal: false, cancelled: false, retired: false,
      token: null, tokenSettled: false, kind: 'detail', key, detailId,
    };
    worker.onmessage = (event) => this.#message(pending, event);
    worker.onerror = () => this.#terminal(pending, { message: 'worker failed' });
    const bytes = await source.readBytes();
    if (this.#pending !== pending) return;
    worker.postMessage({ request: built.request, bytes }, [built.request, bytes]);
    this.onStatus('Building symbolic computation…');
  }

  async #handleDetail(pending, frame, meta) {
    if (meta.kind !== 'delta') {
      return this.#terminal(pending, { message: meta.error?.message || 'symbolic computation failed' });
    }
    if (!(frame.payload instanceof ArrayBuffer) || frame.payload.byteLength !== meta.bytes
        || frame.payload.byteLength > this.hard.maxResponseDocumentBytes) {
      return this.#terminal(pending, { message: 'invalid symbolic detail payload' });
    }
    let documentText;
    try { documentText = new TextDecoder('utf-8', { fatal: true }).decode(frame.payload); }
    catch { return this.#terminal(pending, { message: 'symbolic detail is not UTF-8' }); }
    const merged = this.bridge.session.applyDetail(
      pending.id, pending.key, this.#session, frame.meta, documentText,
    );
    if (!merged.ok) return this.#terminal(pending, { message: merged.error || 'invalid symbolic detail' });
    let handle;
    try {
      handle = await this.render.install(this.#installText(merged.render), { view: pending.detailId });
    } catch (error) {
      if (!this.#isCurrent(pending)) return;
      return this.#terminal(pending, { message: error?.message || 'renderer failed' });
    }
    if (!this.#isCurrent(pending)) {
      this.#consumeAbort(pending, this.render.abortReady(handle), { silent: true });
      return;
    }
    const finalized = this.render.finalize(handle);
    if (!finalized.ok) return this.#terminal(pending, { message: presentMessage(finalized) });
    this.#session = merged.render;
    this.#presentation = handle.presentation ?? null;
    const settled = this.#terminal(pending, {});
    if (!settled.undisturbed) return;
    this.onDetail({ sessionText: this.#session, effectiveOptions: this.#options, viewId: this.view });
    this.onStatus('Symbolic computation opened');
  }

  /* ---------------------------------------------------------- presentation */

  /* Show a different view of the session already retained.
   *
   * No worker, and no bridge call: `webapp_bridge.ml`'s `staged` slot was
   * consumed by the commit that installed this document, so there is nothing to
   * prepare or commit again -- and nothing needs to be. The retained text is
   * the same bridge-validated document; only the graph on screen changes.
   *
   * The pinned visualizer cannot be re-pointed in place (assigning `config` on
   * a live element does not take), so this still goes through the full
   * candidate transaction, which is also what keeps the current view visible
   * until the replacement is ready. */
  async present(selection) {
    if (this.#poisoned) throw new Error(POISONED);
    if (this.#pending) throw new Error('a request is already in flight');
    if (!this.#session) throw new Error('no model is loaded');
    const epoch = ++this.#epoch;
    const pending = this.#pending = {
      epoch, id: null, worker: null, terminal: false, cancelled: false,
      retired: false, token: null, tokenSettled: false,
    };
    let handle;
    try {
      handle = await this.render.install(this.#installText(this.#session), selection);
    } catch (error) {
      // Same classification as a load: a terminal renderer is a fact about the
      // whole application, staleness is about one request.
      if (error?.kind === 'inconsistent') {
        return this.#fatal(error.message || 'the renderer cannot continue', { initiating: pending });
      }
      if (!this.#isCurrent(pending)) return;
      return this.#terminal(pending, {
        message: error?.kind === 'cancelled'
          ? 'the renderer cancelled a request that was still current'
          : (error?.message || 'renderer failed'),
      });
    }
    if (!this.#isCurrent(pending)) {                    // stale: silent, but still consumed
      this.#consumeAbort(pending, this.render.abortReady(handle), { silent: true });
      return;
    }
    const finalized = this.render.finalize(handle);
    // NOT the load matrix. A load reaches `finalize` with the bridge session
    // already committed, so anything but a clean finalize means bridge and
    // presentation disagree and only a reload can settle it. Nothing was
    // committed here, and a `known_old` leaves the previous view connected,
    // visible, and still correct FOR THE SAME DOCUMENT -- so the only fatal
    // outcome is a renderer that has declared itself terminal. Treating the
    // rest as fatal would turn a failed view switch into a needless reload.
    if (!finalized.ok) {
      if (finalized.state === 'inconsistent') {
        return this.#fatal(finalized.error || 'the renderer cannot continue', { initiating: pending });
      }
      return this.#terminal(pending, { message: presentMessage(finalized) });
    }
    this.#presentation = handle.presentation ?? null;
    const settled = this.#terminal(pending, {});
    if (settled.undisturbed) this.onStatus('View changed');
  }
}
