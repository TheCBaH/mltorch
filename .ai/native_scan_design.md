# Bounded ordered scan — foundation contracts

## Status and scope

Status: **contracts recorded; the `Expr` primitive and Region-side
construction/checking/rendering are landed, execution is not.** This record
fixes the representation, error, meter, budget, and rendering contracts for
the bounded-scan primitive, per the LSTM foundation's staged plan, and is
updated in place as each piece lands rather than only once at the end: `Expr`
(`Scan`/`Scan_limits`/`Scan_meter`/`Scan_admission`, both value projections,
inline evaluation) landed first; `Region_local`/`Region_program` construction,
checking and rendering (this section and "RHS rendering" below) landed next.
Running a scan-backed Region program — `Region_execution`/`Region_eval`
actually executing a trace, with a shared metered budget — has not, and fails
with an explicit, typed, temporary error (see "Region propagation") rather
than an inexhaustive match or silent wrong answer. Read it together with
`native_compute_design.md` (Region computation) and
`native_kernel_dsl_design.md` (Kernel IR and `Hard` ceilings), which it
extends rather than restates.

Everything here targets an inference-only, ordered, single-step-lookback
recurrence over two indices (`row`, `lane`), sufficient for LSTM's per-batch,
per-timestep gate recurrence and general enough to host other recurrences
later. It excludes: general fixed points, multi-step lookback, unrolled
compile-time-only recurrence, and training-time recurrence (backprop through
the scan is out of scope).

File citations below are current as of this record; the working plan that
motivated this primitive is not itself a tracked document and is not named
here — see the migration inventory in this repository's execution history for
that provenance instead.

## Representation

`Scan` joins the existing single recursive group in
`lib/expr_internal/expr_repr.ml` (currently a 3-way group over `value` /
`bool_expr` / `reduction`, `expr_repr.ml:1-31`) rather than opening a second
one — `Scan.t` embeds `Value.t` (`init`/`update`) and `Value.t` gains a case
embedding `Scan.t`, so the cycle must live where the file's own header already
says the one intentionally recursive boundary lives.

```ocaml
and scan = {
  width : int;  steps : int;
  lane : Reduce_var.t;   (* bound in init AND update *)
  step : Reduce_var.t;   (* bound in update only *)
  prev : Local_var.t;    (* previous-row reader, bound in update only *)
  init : value;  update : value;
}
(* value gains: *)
| Local_scan_at of Local_var.t * Role.Position.t Index.t * Role.Position.t Index.t
| Scan_at       of scan       * Role.Position.t Index.t * Role.Position.t Index.t
```

`trace[0,l] = init[lane:=l]`; `trace[s+1,l] = update[step:=s, lane:=l,
prev:=trace[s,·]]`. Every lane at step `s+1` reads the same completed row `s`
— lanes are not visible to each other mid-step. `prev` is read through
`Local_at (prev, i)`, an ordinary local read; row and lane are always two
separate index arguments and are never packed into one flattened index, so no
`divmod` is needed to recover them (`divmod` is a Stage 2 concern, over the
LSTM gate math itself — see below).

`Local_scan_at` names a materialized trace the same way `Local`/`Local_at`
name a scalar/vector local, giving Region's dependency-order check
(`region_program.ml:241-250`), shape-agreement check
(`region_program.ml:196-215`), and slot lookup something to key on before any
scan actually executes. `specialize_pixel` is what turns a cached
`Local_scan_at` read into an inline, re-executing `Scan_at` — this rewrite is
exactly why specialization must re-measure (below): a Region program that
only ever reads a trace cheaply can specialize into one that recomputes it.

`lib/expr_internal/value.ml`'s structural `tag` function
(`value.ml:136-147`, ending `Local_at _ -> 10`) gets `Local_scan_at = 11` and
`Scan_at = 12`, appended, never renumbered. Both projections need their own
case in `compare` (`value.ml:166-207`) and `hash` (`value.ml:211-279`, the
recursive `go` at `235-279`), using structural equality/hashing of `scan`
(including `lane`/`step`/`prev` identities, `width`, `steps`).

## Construction API and error type

`Value.t` is `private` (`expr_api.ml:240`), so a different library
(`Region_program`) cannot build `Local_scan_at` by hand — both projections
need public constructors:

```ocaml
val Value.scan_at       : Scan.t      -> row:position Index.t -> lane:position Index.t -> Value.t
val Value.local_scan_at : Local_var.t -> row:position Index.t -> lane:position Index.t -> Value.t
```

`scan.ml` and `scan_limits.ml` are new **internal** modules in
`lib/expr_internal/` (not `lib/expr/`, which holds only the `expr_api.ml`
signature and the `expr.ml`/`expr.mli` façade that aliases the internal
modules one-for-one). `expr.ml` gains `module Scan = Expr_internal.Scan` and
`module Scan_limits = Expr_internal.Scan_limits` alongside the existing
aliases; `expr_api.ml`'s `module type S` gains matching signature entries.

```ocaml
val Builder.scan :
  limits:Scan_limits.t -> width:int -> steps:int ->
  init:(lane:position Index.t -> Value.t Builder.t) ->
  update:(step:position Index.t -> lane:position Index.t ->
          previous_at:(position Index.t -> Value.t) -> Value.t Builder.t) ->
  (Scan.t, Scan.error) Err.t Builder.t

module Scan : sig
  type t = private {
    width : int;  steps : int;
    lane : Reduce_var.t;  step : Reduce_var.t;  prev : Local_var.t;
    init : Value.t;  update : Value.t;
  }
  type error =
    | Bad_steps  of int
    | Bad_width  of int
    | Prev_in_init
    | State_over_limit   of { limit : int }     (* early stop: the LIMIT *)
    | Step_in_init
    | Unbounded_reduction_context
    | Updates_over_limit of { limit : int64 }
  val pp_error : Format.formatter -> error -> unit
end
```

`Scan.t` is exposed as a private record, not opaque, for the same reason
`Reduction.t` is (`expr_api.ml:210-310`): Native needs both binders and both
children directly — `Region_program.check` and its folds, the printer, and
`Region_execution.evaluate_locals`, which runs the trace by reading `width`,
`steps`, `init`, `update` rather than by going through a projection.

The `check.ml:14-16` convention — "the payload is the LIMIT, not the
measure" — applies only to the two early-stopping budget cases
(`State_over_limit`, `Updates_over_limit`); the other five errors are
structural (a malformed descriptor) and carry the offending value or nothing.

**Freshening obligation, stated on `Builder.scan` itself:** a caller composing
independently built fragments (e.g. a Region update callback that constructs
a sub-expression via a separate builder call before combining it) must
freshen that fragment — through the existing `Expr.Rewrite.freshen` — before
combination, using the *shared* builder state (`run_from`, not `Builder.run`,
which restarts numbering at ordinal 0 per `builder.ml:1-7`'s explicit
"two computations run from `initial` deliberately reuse ordinals" rule).
Freshening the already-combined tree cannot recover which binder a captured
free reference meant to name, so this is a caller obligation, not something
`Builder.scan` can repair after the fact. `Expr.Builder.fresh_local` already
exists in the `Builder` signature (immediately after `fresh_reduce`), so
minting `prev` reuses that primitive rather than needing a new one.

## Scope rules and two binder namespaces

| Binder | Bound in | Rejected at construction in |
|---|---|---|
| `lane` | `init`, `update` | — |
| `step` | `update` | `init` |
| `prev` | `update` | `init` |

The constructor rejects only the **newly minted** `step`/`prev` identities
appearing free in `init`; it must not require either body to be closed over
only the scan's own binders, because a Region scan's `update` legitimately
reads earlier Region locals. Correspondingly, `Region_program.check` supplies
per-child masking during the descriptor traversal rather than dumping the
scan's binders into one global `allowed_free` set, which would lose the
per-child distinction between `init` and `update`.

`prev` makes `Local_var.t` a binder for the first time. `Fold.binders :
Value.t -> Reduce_var.t list` (`expr_api.ml:432`, impl
`fold.ml:384-405`) keeps reporting only reducers — `lane` **twice** (two
sibling scopes, `init` and `update`) and `step` once — and gains a sibling
`Fold.local_binders` for `prev`. `Duplicate_binder` in
`lib/expr_internal/check.ml:8` splits into a reducer case and a local case;
`duplicate_binder`/`fragment`/`value` (`check.ml:29-103`) gain the two `init`
rejection rules above. `Rewrite.freshen`'s free-identity avoidance
(`rewrite.ml:175-179`, currently reducer-only — confirmed there is no local
counterpart today) gains a matching `Local_var.Set` avoidance pass, since a
freshened scan must also avoid colliding with any already-free local,
including one captured from an enclosing Region scope.

Per-child scope masking must land in three specific `fold.ml` functions, not
the range a prior draft of this record cited: `locals` (`fold.ml:312-317`),
`scalar_locals` (`fold.ml:323-327`), `vector_locals` (`fold.ml:329-333`), plus
a new scan-local set alongside them. `free_reducers` (`fold.ml:347-377`)
needs masking for `lane`/`step`; `binders` (`fold.ml:384-405`) does not, since
it intentionally reports binders unscoped today and stays that way — only
`Fold.local_binders` is new, not a change to `binders`'s existing contract.

`rewrite.ml`'s `rebuild` (`rewrite.ml:96-153`) gains an `~on_reduce` case for
scan reducers plus a parallel local-binding channel; `freshen` mints all
three identities (`lane`, `step`, `prev`); `substitute_locals` skips `prev`
(it is bound, not free) and gains the `Scan` case in
`Expr.Rewrite.local_binding` (`expr_api.ml:501-503`, since `specialize_pixel`
rewrites `Local_scan_at` into `Scan_at` through that same channel).

Scan order is semantically significant throughout: no rewrite may reorder,
reassociate, or tree-reduce a scan's updates, matching the prerequisite
finding that LSTM's recurrence cannot be expressed as `Semantics.sum` /
`max_reduce`'s reduction combine.

## Region propagation

The Region builder threads no error channel today —
`type 'a t = Expr.Builder.state -> Region_local.t list -> 'a * Expr.Builder.state`
(`region_program.ml:388-390`) — because `scalar`/`vector` always invoke their
continuation and only `finish` returns `Err.t`. A failing `Expr.Builder.scan`
has nowhere to report to under that shape, so `Region_program.Builder.scan`
takes a continuation and reports failure by never invoking it:

```ocaml
val Region_program.Builder.scan :
  limits:Expr.Scan_limits.t -> width:int -> steps:int ->
  init:… -> update:… ->
  ((row:index -> lane:index -> Expr.Value.t) -> (program, error) Err.t t) ->
  (program, error) Err.t t
```

On failure, it short-circuits with `Err.fail` and the continuation never
runs. `scalar`/`vector`/`run` keep their existing polymorphic-in-`'a`
signatures unchanged — every chain still ends at `finish`. `scan` takes
**callbacks** rather than a pre-built `Scan.t`, because the Region builder
mints identities through `run_from` from the shared state, while
`Expr.Builder.run` would restart at ordinal 0 — reusing ordinals across two
independently-run computations is deliberate elsewhere in the builder and
must not leak into scan construction.

**Landed.** `Region_local.Rhs.t` gains `Scan of Expr.Scan.t`, alongside the
existing `Scalar`/`Vector` cases; `Region_local.Shape.t` gains a matching
`Scan of { width : int; steps : int }`, and `Rhs.slot_count` reserves the
whole materialized trace, `(steps + 1) * width` slots — safe as plain `int`
arithmetic (not `Int64`-checked) because `Expr.Scan_limits`'s own hard
ceilings (`hard_max_state`/`hard_max_updates`, both `2^20`) already bound
`width` and `steps * width` at construction, well inside the 32-bit range
this repository's own convention requires checking. `Region_program.Builder.scan`
matches the signature above exactly, converting `Expr.Builder.scan`'s failure
via `Err.map_error (fun e -> `Scan e)` (never a raw `Err.fail` re-wrap, which
would double-wrap the detection trace) and, on success, minting the trace's
own `Local_var.t` and handing the continuation
`Expr.Value.local_scan_at id`. `Region_program.error` gains `` `Scan of
Expr.Scan.error ``.

`Region_local.Rhs.value` (the "one `Expr.Value.t` a local carries" convenience
`Region_program.check`'s folds already use) handles `Scan` by wrapping it as
`Expr.Value.scan_at s ~row:Expr.Index.zero ~lane:Expr.Index.zero` — a
CLOSED placeholder, never evaluated, that lets every existing
`Expr.Fold`/`Expr.Check.fragment` call reuse its already-correct per-child
`lane`/`step`/`prev` masking unchanged (`Fold.free_reducers`/`locals`/
`scalar_locals`/`vector_locals` all recurse into `init`/`update` through
their own `Scan_at` case). This realizes "the descriptor traversal supplies
per-child masking" with no new Region-side scope-checking code: a scan
local's `allowed_free` is `Reduce_var.Set.empty`, same as a scalar's, because
`free_reducers` on the wrapper already excludes `lane` (from `init`) and
`lane`/`step` (from `update`) before `Region_program.check` ever sees the
result. `Shape_mismatch.t` gains a `read : Scalar_read | Vector_read |
Scan_read` field (three declared shapes need to know which READ triggered
the mismatch, not just the binary scalar-vs-vector inference the old
two-shape message inferred); `shape_error` checks all three
`Expr.Fold.{scalar,vector,scan}_locals` sets against all three
"declared-as-something-else" predicates. Dependency order (forward/unknown
local detection) needed no new code: `first_scope_error` already calls
`Fold.locals` on the same wrapper, which already unions scalar/vector/scan
references with `prev` correctly excluded.

**Not landed here: execution.** `Region_execution.evaluate_locals` and
`Region_eval.evaluate_locals` still dispatch only on `Scalar`/`Vector`;
reaching a `Scan` local now fails with a new, explicit
`` `Scan_execution_not_implemented of Expr.Local_var.t `` case in
`Region_eval.error` (mirrored into `Kernel_eval.error`) rather than an
inexhaustive-match compile error. This is a temporary boundary, not a
permanent limitation — it exists only because charging a shared meter across
a trace's lanes and steps, and the streaming multi-slot materializer that
must hold it, are a later step's deliverable (see "Runtime metering" below
and the execution-sequencing note in the project's execution ledger), and it
is expected to disappear once that lands, at which point this case becomes
unreachable and should be removed. `specialize_pixel`'s per-local rewrite
does handle `Scan`: since `rewrite` runs `freshen`/`substitute_locals` on
exactly the `Rhs.value` wrapper above, and both preserve a `Scan_at` node's
outer shape, unwrapping the rewritten wrapper recovers the freshened
`Scan.t` that `Expr.Rewrite.Scan` needs (an `invalid_arg` on the
structurally-impossible non-`Scan_at` result documents the invariant rather
than silently mismatching).

## Runtime metering

Static construction and admission checks cannot bound every context a
`Scan.t` can later appear in: a checked scan can be composed under another
reduction, inserted by a raw rewrite, or passed as a raw `Value.t` straight
to the evaluator. Runtime metering is therefore required even after a scan
passes construction.

```ocaml
module Scan_limits : sig
  type t
  module Field : sig type t = Max_state | Max_updates end
  module Invalid : sig type t = { field : Field.t; value : int64 } end
  type error = Invalid of Invalid.t
  val create : max_state:int -> max_updates:int64 -> (t, error) Err.t
  val max_state : t -> int
  val max_updates : t -> int64
  val default : t
  val pp_error : Format.formatter -> error -> unit
end

module Scan_meter : sig
  type t
  type error =
    | State_over_limit  of { limit : int }
    | Updates_exhausted of { limit : int64 }
  val create : limits:Scan_limits.t -> t
  val charge_update : t -> (unit, error) Err.t
  val remaining : t -> int64
  val pp_error : Format.formatter -> error -> unit
end
(* State reservation is internal to Expr — not in this signature. *)
```

`create` rejects negative fields as `Invalid`, using the same **exclusive
`v >= hard`** rule `Kernel.Limits.create` already applies. Zero is valid and
meaningful for both fields: `max_state = 0` forbids every scan (each reserves
`2 * width >= 2`); `max_updates = 0` admits only descriptors that can never
update (`steps = 0`, whose trace is just the initial row).

**Charge operation and boundary rule, stated once, used everywhere:** exactly
`limit` calls to `charge_update` succeed, and the next call fails **before**
its update body is evaluated. The standalone inline `Scan_at` evaluator and
both Region executors (`Region_execution`, `Region_eval`) call this same
function, so their off-by-one behavior at the boundary cannot drift apart.

**State reservation.** On entry to inline `Scan_at` evaluation, reserve
`2 * width` against the meter's currently live state; release back to the
prior level on every exit path — success, a typed error, and an `Err.Escape`
unwind alike. This enforces the *nesting peak*, not a running total, so it
also catches a row-zero projection (no updates, but state is still reserved)
and a scan that appears only inside another scan's `init`. Reservation and
release stay internal to `Scan_meter`'s implementation — no public
"release" operation exists — because a caller could otherwise under-count
state with an unmatched, duplicate, or wrong-width release. `Region`'s trace
locals write directly into their preflighted slot range instead of going
through inline reservation at all: they only ever need the public
`charge_update` operation, since their storage was already accounted for by
`max_local_slots` at preflight.

**Four reset units — one fresh meter each:**

| Entry point | Fresh meter scope | Calls sharing it |
|---|---|---|
| Standalone `Expr.Eval.value` | Whole raw value | All nested evaluations |
| Region materialization | One Region key | All scalar/vector/trace locals, nested scans and every emitter for that key |
| Region `value_at` | One invocation | All locals and the selected emitter |
| Pixel execution | One output coordinate | The complete pixel expression |

Two separate `Expr.Eval.value` calls that happen to fall within the same
Region key must never create two meters; conversely two `value_at` calls for
different coordinates sharing one key must remain independent, each with its
own full allowance, even though `Kernel_eval.value_at` reaches this path
independently of fusion admission.

**Propagation.** The meter is threaded, not defaulted, at every production
call site: `region_execution.ml`, `region_eval.ml`, `schedule.ml`, and both
`kernel_eval.ml` Pixel arms (the on-demand `eval_value` path's `` `Pixel ``
arm and the whole-tensor `materialize` path's `` `Pixel `` arm are two
distinct sites, both needing "create the meter inside each Pixel callback").
Encountering an inline `Scan_at` with no `?scan_meter` supplied fails with
`` `Scan_meter_required `` before reserving state or evaluating either body —
there is no silent default. Cached `Local_scan_at` reads take no update
charge and go through the projection-reader contract below instead.
`Schedule.ground` changes to return `Err.t`, crossing `Tensor.materialize`'s
float-returning callback with `Err.Escape` in place of today's
`Err.or_raise` (`stage_program.ml`'s `ground`, `stage_program.ml:48-83`, the
`Region_execution.lower` call at `stage_program.ml:71` specifically).

`Expr.Eval.error` gains `` `Scan_meter of Scan_meter.error `` and
`` `Scan_meter_required ``, with `Eval.scan_meter_error : Scan_meter.error ->
Eval.error` as the one conversion, used via `Err.map_error` in both Expr and
Native's direct trace loop.

## Projection evaluation and its error contract

A projection runs exactly `row` time steps: two buffers hold the completed
and next row, bounds-checking row and lane first, then running `row * width`
lane updates, then reading the selected lane. Row 0 returns the initialized
state; row `steps` returns the final state. A nested scan charges the same
meter its enclosing evaluation is using — there is no separate nested budget.

```ocaml
module Scan_projection : sig
  type t = { local : Local_var.t option; row : int; lane : int }
end
module Scan_bounds : sig
  type t = { projection : Scan_projection.t; extent : int }
end
type scan_error =
  | Lane_out_of_range of Scan_bounds.t
  | Row_out_of_range of Scan_bounds.t
  | Unknown_local of Local_var.t
type scan_reader = Local_var.t -> row:int -> lane:int -> (float, scan_error) Err.t
val pp_scan_error : Format.formatter -> scan_error -> unit
(* Extend the existing closed polymorphic [error] row with: *)
(* | `Scan_projection of scan_error *)
val scan_error : scan_error -> error
val value :
  ?local:(Local_var.t -> float option) ->
  ?local_at:(Local_var.t -> int -> float option) ->
  ?scan:scan_reader ->
  ?scan_meter:Scan_meter.t ->
  ?reducer:(Reduce_var.t * int) ->
  ?on_reduction:(unit -> unit) ->
  Env.t -> output:int Coord.t -> Value.t -> (float, error) Err.t
```

`extent` is the exclusive upper bound of what was indexed (`steps + 1` rows,
or `width` lanes) — observed shape data, never a budget limit, so it never
appears in a `Scan_meter.error`. A reader resolves the local id in its trace
table first (a scalar/vector id is not a trace and reports
`Unknown_local`), then checks row, then lane, before indexing — **row wins**
on a simultaneous row/lane failure, tested explicitly. Inline `Scan_at`
applies the identical bounds rule with `local = None`; a cached
`Local_scan_at` read reports `Some id`. Row extent and flattened storage
offset use checked arithmetic throughout — per this repository's 32-bit
`int` rule for `js_of_ocaml`-reachable libraries (`lib/expr` is one), a
`(steps + 1) * width` product must be bounds-checked before narrowing to
`int`, not just re-checked after folding. `Region_execution` and
`Region_eval` implement `scan_reader` with this exact type and widen the
resulting `Expr.Eval.error` unchanged — they do not define a second,
competing projection-error type. Static Region shape checks continue to
reject a scalar/vector local used as a trace before any execution begins.

## Static measures

These bound a compact AST before execution — an admission and
cost-reporting tool, not a runtime guarantee. `max_size` bounds syntax
nodes, not iteration counts, so an update body can still hide an arbitrarily
long `Reduce`; that gap is pre-existing (not introduced by scans) and stays
explicitly out of scope here. A nested scan's updates multiply by an
enclosing reduction's extent only when that extent is a statically constant
`hi - lo`; a scan beneath a statically unbounded reduction is rejected
outright, since LSTM's own scans sit at Region-local top level and cost this
target nothing.

```
U(d)    = width * init_updates(d) + steps * width * (1 + update_updates(d))
per_key = Σ scalar-RHS U(d) + Σ vector-RHS extent*U(d) + Σ Rhs.Scan U(d)
        + outputs_per_key * Σ emitter U(d)
S(d)    = 2*width + max(nested standalone state in init, in update)   (* executable Scan_at *)
T(d)    =           max(nested standalone state in init, in update)   (* trace RHS *)
scan_peak = 0                                                          if no scan is present
          = total_slots + max(trace-RHS T(d), scalar/vector/emitter S(d))   otherwise
```

A scan-free program has `scan_peak = 0` even when it owns scalar/vector
slots, so disabling scans (`max_scan_state = 0`) must never reject an
existing non-scan Region program. When a scan is present, `total_slots` —
the sum of `(steps + 1) * width` over every trace local — is the resident
baseline: a trace RHS writes row `s + 1` directly into its slot range while
reading row `s`, allocating no extra old/next buffer, unlike inline
`Scan_at`, which does. Updates sum over *occurrences*: two inline
projections of the same descriptor are charged twice; a cached
`Local_scan_at` read costs no update. All products/sums use checked or
saturating `Int64` arithmetic.

`Field`, one distinct typed error per measure:

| Field | Bounds | Errored as |
|---|---|---|
| `max_local_slots` | Region trace storage: total slot count, `(steps+1)*width` per scan local | `Local_words_over_limit` (renamed; a total, not a Scan-constructor error) |
| `max_scan_state` | Peak *live* scan state | `Scan.error` / `Scan_meter.error` |
| `max_scan_updates_per_key` | Recurrence iterations for one Region key | Region-level (see below) |
| `max_scan_updates_total` | Summed `keys * per_key` across a Kernel's logical values | `Kernel`-level only |

An oversized trace is reported by `checked_slot_total`
(`region_program.ml:227-239`, whose existing comment argues the opposite of
what is wanted here — *"`max_size` doubles as the ceiling ... rather than a
new, separately-tuned knob"* — and must be rewritten), not by `Scan.error`.
Per-key aggregation is a Region-level failure, so it is not representable by
any single case of `Scan.error` either. `Region_context.program`
(`region_context.ml:67`) and `Region_computation`'s four op-specific call
sites (`region_computation.ml:54-58` RMSNorm, `:76-80` LayerNorm, `:127-132`
SDPA, `:139-143` Softmax) each currently erase the entire
`Region_program.error` into a payload-free `Invalid_program`
(`Err.map_error (fun _ -> Invalid_program) result`). All five sites need to
carry the real error forward — the fix is **wrap the whole
`Region_program.error`, not select one case out of it** — which means giving
`Invalid_program` a payload (or an equivalent replacement constructor) in
both `Region_context.error` and `Region_computation.error`, not a
single-site patch.

### Three censuses

Numbers below are arithmetic over the checked-in corpus's LSTM shapes
(`test/data/pt2_json_model_support.jsonl:82`, `sequencer2d_s`, 911 nodes,
`native_builds:false`, blocked at 36 occurrences of
`torch.ops.aten.lstm.input`), inspected directly rather than cited from an
untracked planning file. The corpus carries two distinct
`(batch, seq_len, hidden)` shapes, all single-layer (`Q=1`),
bidirectional (`R=2`), with biases, batch-first layout, `dropout=0`,
`train=false`: 28 occurrences at `(B,L,K)=(16,16,96)` and 8 at
`(B,L,K)=(32,32,48)`. Using `width = 2K` (the scan's concatenated
cell/hidden state, per the arithmetic contract):

**1. Max per-key update count for one admitted Region program.**
`per_key = R * L * width`. Both corpus shapes give the same value, since
`L * K` is 1536 either way: `2 * 16 * 192 = 2 * 32 * 96 = 6144`
(`≈ 6.1k`). This is the number `max_scan_updates_per_key`'s default must
clear.

**2. Summed `keys * per_key` over live logical values, worst case.** Keys
are batch entries (`B`); the initial one-Region-program-per-output-ordinal
landing (before step 19's sharing work) executes the full recurrence
independently per live output, up to 3 (`output`, `h_n`, `c_n`). Per
occurrence: `B * per_key * live_outputs`. Worst case (`live_outputs = 3`)
summed over the corpus:

```
28 * (16 * 6144 * 3) = 28 * 294,912 =  8,257,536
 8 * (32 * 6144 * 3) =  8 * 589,824 =  4,718,592
                                     -----------
                                      12,976,128   (≈ 13.0M)
```

This is the number `max_scan_updates_total`'s default must clear; it is a
Kernel-only bound (see below), and it is a worst case pending step 16's
actual liveness measurement — some outputs may be dead and elided before
this cost is paid at all.

**3. The Direct path's total is not independently bounded here.**
`Eval_direct` materializes every node's output with no cross-output sharing,
so its total scales with the downstream fan-out of live LSTM outputs across
the whole graph, not just with the op's own shape — a quantity this record
cannot compute without walking the finished graph. This is exactly why
`max_scan_updates_total` is scoped to `Kernel.create`'s aggregation only,
stated as a narrowing rather than implied as a whole-graph guarantee: there
is no single choke point before `Eval_direct.run` or `Stage_program.ground`
begin materializing. Direct and Stage-ground executions remain covered by
the per-key bound and the runtime meter alone. Re-verify all three censuses
against the finished LSTM programs in step 16, once real liveness and
fan-out are measurable.

### Chosen defaults and hard ceilings

Unlike `Hard.depth` / `Hard.eval_depth` / `Hard.eval_recursion`
(`kernel.ml:76-110`), which are genuine stack-overflow frontiers discovered
empirically under node, slot/state counts are **memory- and array-length
bound, not stack-bound** — there is no small-number wall to discover by the
same method. A standalone probe (`Array.make`/fill/read-back over
`8192`, `65536`, `196,608` (`= 6144 * 32`), `1,000,000`, and `50,000,000`
float elements) completed without error on both backends: natively
(`Sys.int_size = 63`, `Sys.max_array_length = 18,014,398,509,481,983`) and
under `js_of_ocaml`/node (`Sys.int_size = 32`,
`Sys.max_array_length = 536,870,911`). Even the `js_of_ocaml` array-length
ceiling sits roughly 40,000× above the largest corpus-derived need (13.0M
updates), so these defaults are policy choices made with deliberate,
stated headroom over the censuses above, not measured frontiers:

| Field | Census | Default | Headroom |
|---|---:|---:|---|
| `max_local_slots` | `total_slots = R*(steps+1)*width`: `2*17*192=6528` (dominant shape) | `8192` | `+25%` over 6528 |
| `max_scan_state` | `scan_peak ≈ total_slots` (LSTM emits no nested inline `Scan_at`, so `T(d)=S(d)=0`) | `8192` | matches `max_local_slots`; no inline scan is expected to need more |
| `max_scan_updates_per_key` | `6144` (census 1) | `8192` | `+33%` |
| `max_scan_updates_total` | `12,976,128` (census 2) | `16,000,000L` | `+23%`, deliberately modest since this budget is a real cost control, not a safety margin |

Hard ceilings (the `Kernel.Limits.Hard` counterparts, following the existing
`depth`/`size`/`values` pattern of round, generous absolute maxima) are
chosen from the same probe evidence rather than discovered:
`Hard.max_local_slots = Hard.max_scan_state = 1_048_576` (2^20; 8 MiB of
`float`s per key, comfortably inside what the probe exercised),
`Hard.max_scan_updates_per_key = 1_048_576`,
`Hard.max_scan_updates_total = 100_000_000L`. None of these claims a
performance guarantee at the ceiling — only that allocation and indexing at
that scale does not crash either backend; wall-clock cost at the ceiling is
a caller's problem, exactly as it already is for `Hard.size`.

## Validated execution artifact

Today, `Region_execution.lower : Region_program.t -> t` and `lower_region :
Region_program.t -> lowered` take no limits at all, and `materialize`/
`value_at` accept any `~output_shape` — `region_execution.mli`'s comment
(line 34) already *claims* callers pass "an already-validated lowered Region
program," but nothing enforces that claim. This record makes it true:

```ocaml
val Region_program.preflight :
  max_local_slots:int -> max_scan_state:int -> max_scan_updates:int64 ->
  output_shape:Vec6.shape -> t -> (unit, error) Err.t

val Region_execution.lower :
  limits:… -> output_shape:Vec6.shape -> Region_program.t -> (t, Region_program.error) Err.t
val Region_execution.lower_region :
  limits:… -> output_shape:Vec6.shape -> Region_program.t -> (lowered, Region_program.error) Err.t
val Region_execution.materialize : ?counters:counters -> lowered -> env:… -> (…, error) Err.t
val Region_execution.value_at    : lowered -> env:… -> output:Vec6.coord -> (…, error) Err.t
```

`lowered` stores the validated shape and limits and is what a fresh per-key
meter is built from; `materialize`/`value_at` stop taking a shape parameter
at all, since it can no longer disagree with what was validated. Both
`lower` and `lower_region` become result-valued, migrating their callers:
`kernel_eval.ml:170` (`lower_region`); `eval_direct.ml:137`,
`native4d/eval_direct4.ml:142` (drifted one line from an earlier count of
this migration list — both currently read `Region_execution.Pixel_loop _ ->
assert false`, i.e. today neither Direct path expects a bare Pixel program to
reach it at all), `stage_program.ml:71` (inside `ground`, whose full body is
`stage_program.ml:48-83`), plus the two test call sites
(`test/native/region_program_test.ml:219`,
`test/native/region_compute_test.ml:699`). `lower_region` is revalidated a
second time after `Kernel_eval.converted` rewrites the emitter via
`Result_conversion.apply` — validating only the unconverted form does not
cover a rewritten executable expression.

`Stage_program.ground` gains `?limits`, preflights every stage before
materializing the first one, and returns `Err.t` — using `Err.Escape` to
cross `Schedule.ground`'s float-returning `Tensor.materialize` callback,
replacing today's `Err.or_raise`. Callers to migrate:
`lib/native_op_walk/native_verify.ml` (propagates the error onward) and the
unwrap-with-`Err.or_raise ~pp_error` test sites:
`test/native/graph_symbolic_{activation,pointwise,shape,norm,pad_slice,pool,
conv,combine}_test.ml`, `test/native/{stage_program,kernel_eval}_test.ml`,
`test/native4d/compute_test.ml`. Re-enumerate these when implementing —
counts here are not a completion check, only a starting inventory.

`max_scan_updates_total` is enforced only at `Kernel.create`, which sums
across a Kernel's logical values; there is no whole-graph choke point
upstream of `Eval_direct.run` or `Stage_program.ground`, so this bound is
named as Kernel-scoped rather than implied as universal (matching census 3
above).

## Specialization and rewrite re-measurement

Region validation admits `Local_scan_at` as a constant-time cached read, but
`specialize_pixel` replaces it with an inline `Scan_at` — so a chain where a
later scan cheaply reads an earlier trace becomes, after specialization, a
descriptor that **re-executes** the earlier scan inside its own update. Cost
can multiply past the admitted limits while the specialized AST stays
compact, and today's contract only carries `max_size`/`max_depth`
(`region_program.mli:39-40`), neither of which is sensitive to this. So
`specialize_pixel` takes scan limits and re-measures the fully rewritten
value with a typed error, threaded through `Stage.pixel_body` and
`Ground_eval`; likewise `substitute_loads` and `substitute_locals` take
`limits` and re-measure their result. `freshen`, `alpha_normalize`,
`substitute_output`, `substitute_reducer`, and `map_sources` stay cheap and
unmeasured — none of them can turn a cached read into a re-executing one.

## Grounding meter and verdict mapping

`Ground_eval` (`lib/native/transform/ground_eval.ml`/`.mli`) is unmetered
today in exactly the way construction cost matters: `at`
(`ground_eval.mli:113`, whose own doc comment states both `at`'s traversal
and `expand` are total) and the internal `ground`/`body_at`
(`ground_eval.ml:245`, `:417-428`) build without any budget. `expand`
(`ground_eval.ml:477-525`) already threads a `~budget:int` counter, but only
charges `n + Ground_expr.size body` **after** `body_at` has already fully
constructed `body` (`:490-492`) — so the unmetered blowup this record targets
happens inside the very call that is supposed to be budgeted.

Two exact accounts, both **reset per proof attempt**, both shared by both
roots within an attempt:

| Account | Unit and charge | Reset |
|---|---|---|
| `max_nodes : int` | Current logical nodes in `lhs + rhs`; a new root costs its size, replacing a one-node cell with size `n` costs `n - 1` | One proof attempt |
| `max_ground_nodes : int64` | Cumulative logical node occurrences constructed or copied, including discarded intermediates | One proof attempt |

Ground the left root then the right, and expand them in that same explicit
sequence — never rely on tuple-argument evaluation order for charging or
error precedence. The Structural attempt and the subsequent constant-bound
attempt each get fresh accounts; a discarded attempt must never consume the
authoritative attempt's allowance, and a new coordinate/member comparison
starts fresh as well.

For construction fuel: a newly created node costs one plus any logical
subtrees copied from cached state; attaching a freshly constructed child (or
moving an accumulator into its sole successor) transfers already-charged
occurrences without re-charging them; reusing a cached previous lane or cell
body costs its cached logical size **at every embedding**, even when the
implementation reuses a pointer; discarded scan rows and max-pool
accumulators consume fuel with no refund. The pair account (`max_nodes`)
charges only retained results and replacement deltas, never unchanged
ancestors again, and refuses a selection that would exceed its remaining
allowance *before* embedding it — a cached logical size makes this check
possible without walking the oversized value. All size arithmetic is
checked, matching this repository's general 32-bit-`int` rule for
`js_of_ocaml`-reachable code; a checked logical size is stored beside every
grounded intermediate rather than recomputed with `Ground_expr.size`, since
`Ground_expr` has ordinary tree semantics and pointer sharing does not bound
a recursive consumer walking it. A sharing-aware ground IR remains a
separate, later design question, not addressed here.

```ocaml
module Budget : sig type t = { max_ground_nodes : int64; max_nodes : int } end
module Meter  : sig type t val create : Budget.t -> t end
module Term   : sig type t val expression : t -> Ground_expr.t val size : t -> int64 end
val default_budget : Budget.t
(* Extend [Ground_eval.error] with exactly: *)
(* | `Ground_nodes_over_limit of int64 | `Pair_nodes_over_limit of int *)
val at     : meter:Meter.t -> Env.t -> Tensor_id.t -> Vec6.coord -> (Term.t, error) Err.t
val expand : meter:Meter.t -> boundary:(Ground_expr.Origin.t -> Cluster_var.t option) ->
             Env.t -> Term.t -> (Term.t, error) Err.t
```

`at` registers a root in its meter; `expand` replaces that registered root
and returns its successor — the old term stops being active. Both live in
`Ground_eval`, which must not depend on `Map_verify`, so registration and the
pair delta cannot be forgotten by a caller. A term belongs to its creating
meter; passing a foreign or already-replaced term is `invalid_arg`, checked
by an internal identity/generation token — a failed construction ends that
attempt outright, with no way to probe a partial frontier. Nonpositive
budget fields behave as zero allowance (the first positive charge fails);
pair admission is checked before construction fuel when both would reject
the same insertion, giving deterministic precedence.

**Profile migration.** `Map_verify.Budget.t`
(`map_verify_types.ml:43-84`/`map_verify.mli:60-84`) gains
`max_ground_nodes : int64` alongside its existing `max_coords`/`max_nodes`/
`max_rounds`/`sample`. Initial policy values are ten times each existing
pair cap — starting allowances, not measured LSTM capacity claims:

| Profile | Existing `max_nodes` | New `max_ground_nodes` |
|---|---:|---:|
| `Budget.default` | 200,000 | `2_000_000L` |
| `Budget.cumulative` | 1,000,000 | `10_000_000L` |
| `Budget.release` | 50,000 | `500_000L` |
| `Effort.Quick` | 20,000 | `200_000L` |
| `Effort.Standard` | uses `Budget.release` | uses `Budget.release` |
| `Effort.Thorough` | 200,000 | `2_000_000L` |

`Ground_eval.default_budget = { max_ground_nodes = 2_000_000L; max_nodes =
200_000 }`; `Map_verify.Budget.default`'s new field is derived from it.
Migrate the full external literal in
`test/native/verify_boundary_test.ml:359-362` (currently `{ max_coords =
4096; max_nodes = 200_000; max_rounds = 0; sample = None }`, no
`max_ground_nodes`) to include `2_000_000L`. Update `Budget.pp` to
`{coords<=%d ground_nodes<=%Ld nodes<=%d rounds<=%d sample=%a}` and review
affected goldens.

**Verdict mapping.** `Unproved.t` (`map_verify_types.ml:194-267`, 10
constructors today; `Verdict.labels` at `:303-313` totals
`6 + 10 + 1 = 17`, pinned by
`test/native/outcome_label_test.ml:73`'s `verdicts=17 labels=17`) gains
`Max_ground_nodes of int64`:

| Grounding error | Verdict |
|---|---|
| `` `Ground_nodes_over_limit limit `` | `Unproved (Max_ground_nodes limit)` |
| `` `Pair_nodes_over_limit limit `` | `Unproved (Max_nodes limit)` |
| Other existing errors | `Unproved (Eval error)` |

`Unproved.pp` prints `over max_ground_nodes (%Ld)`; `reason` returns `over
max_ground_nodes`; `reasons` gains that label; the pinned test count becomes
18. **Placement note:** `Unproved.t`'s existing constructor order is already
not alphabetical (`Eval, Exhausted, Max_nodes, Max_rounds, Max_clusters, ...`
— `Max_clusters` should sort before `Max_nodes`/`Max_rounds` under this
repository's alphabetical-ordering convention but does not, predating this
work). `Max_ground_nodes` is inserted at its alphabetically correct position
relative to its immediate neighbors going forward (immediately before
`Max_nodes`) without reordering the pre-existing violation, which is out of
this change's scope.

**Regressions required** (isolate each budget by giving the other
sufficient allowance): pair exhaustion (oversized root reduction, root scan,
scan behind a later expanded cell, dense repeated-lane recurrence, two roots
each using half-to-three-quarters of the cap, a later expansion crossing it)
expecting `Unproved (Max_nodes limit)`; pair accounting (an exactly-at-cap
pair surviving several replacement rounds, a sparse scan projection and
`Max_pool.Value` fitting despite discarded intermediates); construction
exhaustion (discarded scan rows/max-pool accumulators exhausting fuel while
the pair fits, repeated cached embeddings charged at every embedding)
expecting `Unproved (Max_ground_nodes limit)`; and reset/boundary cases
(fresh full budgets per attempt, exactly-limit vs. next-charge distinguishing
the two verdicts).

## Fusion admission

`Kernel_elab.admit` (`kernel_elab.ml:71-76`) rejects fusion only when
`Kernel.pixel_expression` is `None` — a singleton `Scan_at` program is still
a pixel expression by that test, so it is not rejected today. `admit` gains
an explicit scan/recurrent-effect summary, tested at both the planner and
direct entry points. `Fusion_plan`'s `pointwise` test
(`fusion_plan.ml:145-148`) already covers the *planner* path only, because it
happens to consult `Region_program.Fold.binders`, which will report a scan's
`lane`/`step` binders once they exist — `Kernel_elab.admit` needs its own
check because it does not consult `Fold.binders` at all.

## RHS rendering

`Me_detail.of_value` builds one expression root per
`local.Region_local.value` (`me_detail.ml:78-100`) and `Region_trace.pp_local`
prints `Expr.Pp.value local.value` (`region_trace.ml:94-95`) — neither works
for a descriptor, which has no single expression root. Render a scan local
with `init` and `update` as scoped children in both places; do not fabricate
a projection at dummy indices to force it through the existing
single-expression path. The explorer fixture gains an **unspecialized** scan
local so this path is exercised before any specialization rewrite runs.

**Landed.** `Expr.Pp` gains `scan`/`scan_open`, factored out of `value_open`'s
existing `Scan_at` case (`pp.ml`'s `at`/`scan_body`/`guard_at` are now
mutually recursive top-level functions taking `~names` explicitly, rather
than closures private to `value_open`): `scan_body` renders `init`/`update`
with the same scoped `lane`/`step`/`prev` naming a real `Value.Scan_at` read
gets, but stops before the trailing `@[row,lane]` a read's own `at` case
appends — there being no row/lane to show for a plain declaration.
`Region_program.pp` and `Region_trace.pp_local` (the two **text** renderers)
special-case `Region_local.Rhs.Scan` to call `Expr.Pp.scan_open`/`scan` on
the descriptor directly, never `Rhs.value`'s placeholder-wrapped wrapper.
`Me_detail.of_value` (the **graph-node** renderer) instead splits a scan
local into two named roots, `l<i>-init-e`/`l<i>-update-e`, each holding
`s.Expr.Scan.init`/`.update` directly — no wrapper needed at all, since a
root has no enclosing projection to fabricate — and extends its local-naming
map with `s.Expr.Scan.prev -> "<local-name>_prev"` so `update`'s `prev`
occurrences (reached unwrapped, so `Pp`'s own scan-scope naming never
applies) still get a readable name instead of a raw identity.

## `divmod` for Stage 2

Not part of the scan primitive itself, but recorded here per the
foundation-contracts step so Stage 2 has one shared implementation rather
than several ad hoc ones: encode a remainder as
`x - K * floor_div_pos(x, K)`, built from the existing `Add`/`Scale`/
`Floor_div_pos` delta nodes and converted back to a nonnegative range with
`Clamp_low` — never `Assume_position`. This is sound, not a papering-over:
for `x >= 0` and `K > 0` the remainder is provably in `[0, K)`, so the clamp
never actually fires; that argument, not the clamp itself, is what makes the
encoding trustworthy. One shared helper serves every quotient/remainder
split the LSTM gate math needs (`lane mod K`, `j mod K`, `c mod K`, `a mod
R`), tested with `K > 1`, `R = 2`, and `Q > 1` with unequal dimensions so a
quotient/remainder mixup would be observable.

## Plumbing

`Kernel.Limits.t` (`kernel.ml:55-64`) gains all four fields
(`max_local_slots`, `max_scan_state`, `max_scan_updates_per_key`,
`max_scan_updates_total`) with `Hard` counterparts in `Kernel.Limits.Hard`
(`kernel.ml:76-110`), validated by `create` (`kernel.ml:122-153`;
`kernel.mli`'s `Limits` signature starts at line 71). `create` also builds
and stores an `Expr.Scan_limits.t` from `max_scan_state -> max_state` and
`max_scan_updates_per_key -> max_updates` — the aggregate total has no
meaning for one descriptor — so `Kernel.Limits.scan_limits : t ->
Expr.Scan_limits.t` is a pure accessor that cannot fail. `Kernel.Limits.default`
derives its scan fields from `Expr.Scan_limits.default` so the two constants
cannot drift apart; a test pins that derivation.

The same configured value must reach symbolic construction:
`Eval_symbolic.run` currently hardcodes `Kernel.Limits.default`
(`eval_symbolic.ml:50`) — the same hardcoded-default smell also appears in
`Ground_eval.body_at`'s call to `pixel_body` (`ground_eval.ml:418`), worth
fixing alongside since it is the identical defect — while
`Region_kernel.of_graph ?limits` (`region_kernel.ml:1-2`) passes its limits
only to `Kernel_adapt`. Thread `?limits` through `Eval_symbolic.run` and pass
the same value from `Region_kernel.of_graph`, with tighter- and
wider-than-default regressions through Direct, Symbolic, and
`Region_kernel.of_graph`.

`checked_slot_total` (`region_program.ml:227-239`) switches to billing
`max_local_slots` and its comment is rewritten to match (it currently argues
the opposite). All existing `Kernel.Limits.create` call sites in
`depth_probe.ml`, `fusion_test.ml`, `kernel_test.ml`, and
`region_compute_test.ml` (each using today's 8-argument form) gain the four
new arguments.

## Validation

```sh
opam exec -- dune build lib/expr                                 # the Expr-only checkpoint
NO_COLOR=1 opam exec -- dune runtest test/expr test/native
make precommit
make jsoo.runtest && make jsoo.inline-runtest
```

No code changes accompany this record. The array-capacity measurement
(`Array.make`/fill/read-back at `8192`/`65536`/`196,608`/`1,000,000`/
`50,000,000` elements, natively and under `js_of_ocaml`+node) was run from a
standalone script outside the dune tree, since `Kernel.Limits`' scan fields
do not exist yet for a `depth_probe.ml`-style committed regression to pin
against; a committed probe following that file's pattern belongs with the
implementation once the fields exist (tracked as part of landing the
primitive, not a gap in this record).
