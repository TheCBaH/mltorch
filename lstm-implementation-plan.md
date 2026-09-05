# `lstm.input` — implementation plan (rev. 19)

## Context

`_ai_/todo-ops.md` and `_ai_/ops-progress.md` record `lstm.input` as investigated 2026-09-05 and
deliberately **not** landed; `_ai_/lstm.md` holds the scoping notes and `_ai_/lstm-plan.md` an
expanded-scope plan written the same day. This plan grounds those against the actual source and
turns them into an executable sequence.

`sequencer2d_s` (911 nodes) is `native_builds:false`, blocked at `torch.ops.aten.lstm.input` — 36
occurrences, its first frontier (`test/data/pt2_json_model_support.jsonl:82`).

The blocker is real: `Semantics.SEMANTICS` exposes only `sum` and `max_reduce`
(`semantics.ml:152-155`), both folding independently-computable terms with a fixed associative
combine. An LSTM step is a non-associative recurrence over two hidden *vectors*. `cumsum` looked
like the same shape and was not (`_ai_/lstm.md:95-104`).

Decisions: the scan becomes a **new `Expr` AST node**, it lands **alone first**, and state
footprint gets its **own budget dimensions**.

Eighteen review rounds are folded in; every finding was checked against the source and all held —
including five that corrected censuses, claims, or a test design I had published.

---

## Corrections to `_ai_/lstm-plan.md` found by reading the source

1. **Region-authored ops are single-output today** — `check_output` rejects `output <> 0`
   (`region_computation.ml:23-27`).
2. **Corpus-scale traces exceed the Region budget** — `Region_program.check` bills total slots
   against `max_size` (`region_program.ml:227-239`); `Limits.default.max_size` is 4096
   (`kernel.ml:150`). Sequencer2D needs ~6500.
3. **Do not flatten the trace into `Local_at`** — `Expr.Index` has no modulo (`index.ml:17-31`).
4. **Blast radius is small** — only `ground_eval.ml`, `me_detail.ml` and one incidental site in
   `vec6.ml` match `Expr.Value` constructors outside `lib/expr*`.
5. **`expr_api.ml` is at 749/1000 lines**; `file-size-exceptions.txt` is empty.
6. **`Const_ssa` needs no work** (`const_ssa.ml:35,136-138`) — keep `Lstm` off `allows`.
7. **`Tensor[]` arguments cost an ATen walk recipe** (`aten_walk_gen.ml:94-96`).
8. **Stage 2 needs a modulo `Expr.Index` does not have** — decided below.

### Things I got wrong

Five published claims were false. Three were censuses from loose or truncated greps — the lesson is
to match an *application form* and count, never a bare name with `head`:

- **The grounding budget does not exist.** `body_at` calls the unbudgeted recursive `ground`
  (`ground_eval.ml:417-428`); only `expand ~budget` is metered. The comment at `:226` says
  otherwise and is wrong.
- **`Kernel.Limits.create`**: I published "68 sites across 22 files", having swept up `Me_limits`,
  zip limits and Model Explorer tests. Measured: **11 invocations across 4 test files**
  (`depth_probe.ml` 2, `fusion_test.ml` 1, `kernel_test.ml` 5, `region_compute_test.ml` 3) plus
  `kernel.ml`'s own `create`; four other hits are comments.
- **`Stage_program.ground`**: I published "four callers" (a `head -10` truncation), then "43 calls"
  (an unbracketed comment slipped past the filter). Measured by matching the application form:
  **42 calls across 12 files**, plus **9 non-call references**, out of 51 textual hits in `.ml` and
  `.mli`.
- **A static gate cannot bound scan cost.** Rev. 8 claimed construction-time and rewrite-time
  checks sufficed; they do not (1c).
- **My "shared metering" test was vacuous.** A program whose locals and emitter individually fit
  but jointly exceed the per-key budget is rejected by the *preflight* this plan also requires, so
  it never reaches a runtime charge and proves static aggregation, not meter ownership (1d).

Also corrected below, each verified in source: `Scan_at` cannot serve as a Region read;
`Local_scan_at` needs its own public constructor; standalone scans and expanding rewrites bypass
every limit; no execution API is a validation gate; `Kernel_elab.admit` does not reject a scan; the
option callback cannot carry typed bounds errors; Model Explorer has no root for a scan RHS; a
pre-built `Scan.t` collides with the Region binder namespace; a nominally colliding prebuilt
fragment cannot be repaired after composition; the Region builder had neither an error channel nor
limits; grounding must meter *logical* size across *both* terms of a comparison;
`Region_eval.materialize` retains one slot array per key; my scope rule rejected valid captures;
update accounting ignored vector extents, emitters, key counts and reduction context;
specialization escapes the limits it was admitted under; peak state double-charged a trace RHS; the
bundle never named the recursive representation boundary; the payload convention was misapplied to
malformed-extent and scope errors; and the total-update number was justified only by a per-key
census.

---

## Prerequisites

- **Commit the in-flight `cumsum` landing** (25 modified + 4 untracked files).
- **Fix a live defect in `Region_eval`.** `region_eval.ml:59` dispatches locals on the numeric slot
  count (`| offset, 1 -> ...`), the bug fixed for `Region_execution` in `ac11eb8` and pinned by
  `region_compute_test.ml:311-321`. A `Vector` local of extent 1 (SDPA at `Wk = 1`) takes the
  scalar branch with no `~reducer` bound and raises `Unbound_reducer`. Reachable via
  `Region_execution.value_at` (`:203-204`) → `Kernel_eval` per cell (`kernel_eval.ml:231-237`).
  `fixup! 99872ea` plus a `Wk = 1` regression on `value_at`.
- **ATen feasibility probe** — static-link check plus a tiny three-output oracle, no LSTM landing.
  `conv3d` shipped with `at::native::slow_conv3d` stubbed and cost a session.

---

## Stage 1 — the scan primitive

### 1a. Specify before implementing

Write into the tracked `.ai/` record **before the Expr commit**: the representation and its place in
the recursive group, the construction API and both public projection constructors, the exact error
declarations and their propagation, the per-child scope rules, the two-namespace binder rule and
the caller's freshening obligation, **the meter API including its charge operation, boundary rule,
four reset units and propagation**, the static measures with their formulas/censuses/numbers and
what they do and do not guarantee, the validated-execution token, specialization and rewrite
re-measurement, the grounding meter and verdict mapping, the RHS rendering path, and the `divmod`
encoding for Stage 2.

### 1b. `Expr`: the scan node

**Representation — it joins the one recursive group.** `Scan.t` contains `Value.t` and
`Value.Scan_at` contains `Scan.t`, so the cycle is owned where the existing one is:
`lib/expr_internal/expr_repr.ml`, "the one intentionally recursive representation boundary".

| File | Change |
|---|---|
| `expr_repr.ml` | add `scan` to the recursive group; add both projections to `value` |
| `scan.ml` (new, internal) | alias + smart constructors |
| `scan_limits.ml` (new, internal) | the checked limit record and the meter |
| `expr_api.ml` | extend `module rec Bool … and Reduction … and Value` with `Scan`; add `Scan_limits`, `Scan_meter`, `Builder.scan`, both value constructors, the admission check |
| `expr.ml` | façade exports `Scan`, `Scan_limits`, `Scan_meter` |

```ocaml
and scan = {
  width : int;  steps : int;
  lane : Reduce_var.t;   (* bound in init AND update *)
  step : Reduce_var.t;   (* bound in update only *)
  prev : Local_var.t;    (* previous-row reader, bound in update only *)
  init : value;  update : value;
}
(* value: *)
| Local_scan_at of Local_var.t * Role.Position.t Index.t * Role.Position.t Index.t
| Scan_at       of scan       * Role.Position.t Index.t * Role.Position.t Index.t
```

`trace[0,l] = init[lane:=l]`; `trace[s+1,l] = update[step:=s, lane:=l, prev:=trace[s,·]]`. All
lanes at a step read the same completed previous row. `prev` is read as `Local_at (prev, i)`. Row
and lane are **separate index arguments**, never flattened.

**Both projections need public constructors.** `Value.t` is `private` (`expr_api.ml:240`), so
`Region_program` — a different library — cannot build `Local_scan_at` itself:

```ocaml
val Value.scan_at       : Scan.t      -> row:position Index.t -> lane:position Index.t -> Value.t
val Value.local_scan_at : Local_var.t -> row:position Index.t -> lane:position Index.t -> Value.t
```

`Local_scan_at` names a trace local, exactly as `Local`/`Local_at` name scalar/vector locals, so
dependency order (`region_program.ml:241-250`), shape agreement (`:196-215`) and slot lookup have
something to key on. `specialize_pixel` rewrites it into `Scan_at`, so `Expr.Rewrite.local_binding`
(`expr_api.ml:501-503`) gains a `Scan of Scan.t` case. **`dune build lib/expr` is an explicit
checkpoint**, followed by a **Native compile/integration checkpoint** that exercises the
cross-library boundary — a scan-backed Region program built, checked, printed and executed purely
through the public `Expr` surface — so a signature that Native cannot actually use is caught in
the Expr commit rather than three commits later.

**Construction API and error type.** The "payload is the LIMIT, not the measure" convention
(`check.ml:14-16`) applies **only** to early-stopping budget errors:

```ocaml
val Builder.scan :
  limits:Scan_limits.t -> width:int -> steps:int ->
  init:(lane:position Index.t -> Value.t Builder.t) ->
  update:(step:position Index.t -> lane:position Index.t ->
          previous_at:(position Index.t -> Value.t) -> Value.t Builder.t) ->
  (Scan.t, Scan.error) Err.t Builder.t

module Scan : sig
  (* Exposed as a private record in the recursive signature, exactly as
     [Reduction.t] is today (expr_api.ml:219-234). Native needs the binders and
     both children: [Region_program.check] and its folds, the printer/rendering
     path, and [Region_execution.evaluate_locals] — which reads [width], [steps],
     [init] and [update] to run the trace directly rather than through a
     projection. The internal [scan.ml] alias is not reachable across the library
     boundary, so a signature exposing only [error] would make the Native work
     impossible to write. *)
  type t = private {
    width : int;  steps : int;
    lane : Reduce_var.t;  step : Reduce_var.t;  prev : Local_var.t;
    init : Value.t;  update : Value.t;
  }

  type error =
    | Bad_width  of int            (* the offending value *)
    | Bad_steps  of int
    | Step_in_init                 (* payload-free: a freshly minted internal ordinal *)
    | Prev_in_init
    | Unbounded_reduction_context
    | State_over_limit   of { limit : int }     (* early stop: the LIMIT *)
    | Updates_over_limit of { limit : int64 }
  val pp_error : Format.formatter -> error -> unit
end
```

**The caller must freshen a prebuilt fragment before composing it — the API cannot do it.** A
callback may return an opaque value built from an independent supply (`Builder.return`). If its
nominal identities collide with `lane`/`step`/`prev`, a captured free read is structurally
indistinguishable from an intended bound read: in a bound child it is *already* captured, and in
`init` a collision with `step`/`prev` yields a false scope error. The codebase already documents
this — `expr_api.ml:446-456` states that freshening the combined tree afterwards "repairs nothing —
once a nominal collision has captured a reference, there is no record of which binder it meant."
Threading `run_from` prevents collisions for computations the callbacks *mint*; it cannot repair an
opaque prebuilt value. State the obligation on `Builder.scan`, and write the regression to
**freshen the captured fragment inside the callback, from the shared builder state, before
combining** — then assert the three binders are fresh and the captured read still names the earlier
Region local.

**Region propagation — a program-producing continuation, with limits.** The Region builder is
`type 'a t = Expr.Builder.state -> Region_local.t list -> 'a * Expr.Builder.state`
(`region_program.ml:388-390`): `scalar`/`vector` always invoke their continuation and only `finish`
returns an `Err.t`, so there is no error channel for a failing `Expr.Builder.scan`; its state holds
no limits either.

```ocaml
val Region_program.Builder.scan :
  limits:Expr.Scan_limits.t -> width:int -> steps:int ->
  init:… -> update:… ->
  ((row:index -> lane:index -> Expr.Value.t) -> (program, error) Err.t t) ->
  (program, error) Err.t t
```

On failure it short-circuits with `Err.fail` and never invokes the continuation.
`scalar`/`vector`/`run` keep their signatures — they are polymorphic in `'a`, and every chain ends
at `finish`. `scan` must take the **callbacks**, not a `Scan.t`: the Region builder mints through
`run_from`, while `Expr.Builder.run` restarts at ordinal 0, and `builder.ml:1-7` states outright
that two computations run from `initial` deliberately reuse ordinals.

**Scope rules — reject the minted identities, pass everything else through.**

| Binder | **Bound in** | Rejected at construction in |
|---|---|---|
| `lane` | `init`, `update` | — |
| `step` | `update` | `init` |
| `prev` | `update` | `init` |

The constructor rejects membership of the **newly minted** `step`/`prev` identities in `init` and
nothing else — it must not require the bodies to be closed over only the scan binders, since a
Region scan's update legitimately reads earlier Region locals. Correspondingly
`Region_program.check` must not dump the scan's binders into one global `allowed_free`, which loses
the per-child distinction; the descriptor traversal supplies per-child masking.

**Two binder namespaces.** `prev` makes `Local_var.t` a binder for the first time. Keep
`Fold.binders : Value.t -> Reduce_var.t list` (`expr_api.ml:432`) as the reducer list — reporting
`lane` **twice** (two sibling scopes) and `step` once — and add `Fold.local_binders`. Split
`Duplicate_binder` into reducer and local cases (`check.ml:8`). Mirror in `freshen`'s free-identity
avoidance (`rewrite.ml:175-179`, no local counterpart today), comparison, hashing, printing, alpha
normalization and substitution.

**Remaining bundle work:**

| File | Work |
|---|---|
| `value.ml:136-147` | `tag` gets 11 and 12 — **append only** |
| `value.ml:166-207`, `:235-279` | de Bruijn levels for `step`/`lane` **and** a parallel `Local_var` channel for `prev` |
| `eval.ml:236-314` | interpreter, plus metering below |
| `fold.ml:94-119` | `walk` recurses into `init`/`update` and both projections |
| `fold.ml:206-259`, `:323-333`, `:347-405` | **per-child** scope masking for `free_reducers`, `locals`, `scalar_locals`/`vector_locals` (unscoped `walk`s today) and a new scan-local set |
| `rewrite.ml:96-153` | `rebuild`: `~on_reduce` plus a local-binding channel; `freshen` mints all three; `substitute_locals` skips `prev`, gains the `Scan` case |
| `check.ml:29-103` | duplicate binders in both namespaces; the two `init` rules |
| `pp.ml:70-157` | both projections; scoped naming across both namespaces |

Scan order is semantically significant: no reordering, reassociation or tree reduction.

**Projection evaluation runs exactly `row` updates.** Two row buffers hold only the last completed
row. Bounds-check `row`/`lane`, run exactly `row` updates, read the lane. Row 0 returns `init`; row
`steps` the final state.

**The evaluator callback must carry typed bounds errors.** `Expr.Eval`'s callbacks are
`?(local = fun _ -> None) ?(local_at = fun _ _ -> None)`, turning `None` into `` `Unbound_local ``
(`eval.ml:217,245,249`). A cached scan reader of that shape cannot distinguish an unknown local
from an out-of-range row or lane, and `Local_scan_at` carries no descriptor for `Eval` to check
itself. Define `~scan` over a closed variant preserving at least `Unknown_local`,
`Row_out_of_range`, `Lane_out_of_range`; same contract in `Region_execution`/`Region_eval`.

### 1c. What actually bounds scan cost: an explicit meter

**Static analysis cannot be the enforcement point.** Each attempt has a bypass: `Builder.scan` sees
reductions *inside* a descriptor, but a caller can build a valid `Scan.t` and return
`Value.scan_at d …` from an ordinary `Builder.reduction` whose extent is dynamic —
`Builder.reduction : … -> (index -> Value.t t) -> Value.t t` (`expr_api.ml:341-351`) takes neither
limits nor an error channel, and a *constant* outer reduction multiplies real updates just as
invisibly. `Expr.Rewrite` rebuilds with **raw** constructors and inserts subtrees documented as not
re-traversed (`expr_api.ml:490-516`). Public combinators compose checked projections. And
`Expr.Eval.value` accepts raw `Value.t`.

**So the guarantee is a runtime meter — with a public charge operation, since the Region executor
lives in another library and runs trace descriptors itself, outside `Expr`'s `Scan_at` evaluator.**

```ocaml
module Scan_limits : sig
  type t
  type error = Invalid of { name : string; value : int64 }
  val create : max_state:int -> max_updates:int64 -> (t, error) Err.t
  val max_state : t -> int
  val max_updates : t -> int64
  val default : t
  val pp_error : Format.formatter -> error -> unit
end

module Scan_meter : sig
  type t
  type error =
    | Updates_exhausted of { limit : int64 }   (* early stop: the LIMIT *)
    | State_over_limit  of { limit : int }
  val create : limits:Scan_limits.t -> t
  val charge_update : t -> (unit, error) Err.t
  val remaining : t -> int64
  val pp_error : Format.formatter -> error -> unit
end
(* State reservation is INTERNAL to the Expr implementation — not in this signature. *)

(* Expr.Eval.value gains: *)  ?scan_meter:Scan_meter.t
```

`t` is abstract and built only through `create`, which rejects negative fields as `Invalid` and
applies the same **exclusive `v >= hard`** rule `Kernel.Limits.create` uses (four comments in the
tree cite that convention). **Zero is valid and meaningful for both fields**: `max_state = 0`
forbids every scan, since each reserves `2 * width >= 2`; `max_updates = 0` admits only descriptors
that can never update — those with `steps = 0`, whose trace is just the initial row.
`Kernel.Limits.create` builds and stores the `Scan_limits.t` at construction, so
`Kernel.Limits.scan_limits` is a pure accessor that cannot fail — which removes the question of how
a caller constructs the abstract value downstream.

**Construction and static admission use the descriptor's worst case; only the meter is
projection-sensitive.** These three boundaries must be stated together or they contradict each
other. `Builder.scan` sees no projection at all, and `Scan_admission.check` measures occurrences
with `U(d)`, whose formula carries `steps` and not the projection's `row`. Since
`U(d) >= steps * width` for any positive-width descriptor, **a positive-step descriptor is rejected
at both boundaries under a zero update limit, even when a particular runtime projection would
execute row zero and the meter would charge nothing.** That asymmetry is deliberate, not a defect:
`row` is an index expression and is symbolic on exactly the paths that matter, so a
projection-sensitive static measure (`U_at(d, row)`, exact for a constant row and `steps` otherwise)
would add machinery that almost never fires while giving up early rejection at construction. Pin
the intended outcomes by running **the same positive-step, row-zero case at all three boundaries**:
`Builder.scan` → `Updates_over_limit`; `Scan_admission.check` → `Updates_over_limit`; and the
evaluator, given a wider admission limit and a narrow meter, → succeeds having charged nothing.

**Update boundary rule, stated once:** exactly `limit` charges succeed, and the next charge fails
**before its update body is evaluated**. The standalone `Scan_at` evaluator and both Region
executors call the *same* `charge_update`, so their off-by-one behaviour cannot drift.

**State needs its own enforcement, because the update meter cannot see it.** `Expr.Eval.value`
accepting a meter is not enough: a `Scan.t` constructed under a wider `max_state` can be evaluated
with a narrower configured meter and allocate its two row buffers above the limit — and a scan
nested in an *initializer*, or any row-zero projection, consumes that state while performing **no
update charges at all**, so no amount of update metering catches it. Standalone evaluation would
then honour only half the configured resource contract. So the meter reserves state: entry checks
and reserves `2 * width` before the buffers are allocated, exit releases it, and the reservation
tracks the true nesting peak. Reserving is O(1) per scan entry, which is why this is preferred
over re-running a whole-value admission traversal at an `Expr.Eval.value` boundary called once per
coordinate and per lane.

**That reservation pair stays internal to the Expr implementation and out of the public
signature.** A public `leave_scan : t -> width:int -> unit` independent of entry would let any
caller release without reserving, release twice, or pass a mismatched width, driving the recorded
live state below zero so that a later entry admits more buffers than `max_state` — defeating the
reason the meter and limits are abstract at all. Only `Expr.Eval` ever allocates those buffers:
the Native Region executor needs the public `charge_update` because it runs trace locals itself,
but it writes them **directly into the preflighted slot range**, which `max_local_slots` and the
peak-state preflight already bound — the same fact recorded above as "a trace RHS allocates no
old/next buffers". So the public surface is exactly `create`, `charge_update`, `remaining` and
`pp_error`. (If a future caller genuinely needs reservation, make entry return an opaque token
that release consumes, rejecting stale or foreign reservations — never an independent
width-taking release.)

Release must happen on **every** success *and* error path — with `Err.Escape` in play the scan
evaluator restores the previous reserved level in both branches, `protect`-style; this is the easy
thing to get wrong. Compute the reservation with `Int64`-checked arithmetic: `width` is validated
at construction (`Bad_width` rejects nonpositive), but `2 * width` summed across nesting is an
aggregate, and a check on a wrapped result is not a bound. Test failed entry, nested entry,
exceptional unwinding, and — as an internal invariant test, since the public API cannot express it
— attempted double and mismatched release.

Regression: construct a descriptor under a wider limit and evaluate it with a narrower one,
**including a nested row-zero projection that performs no update charges** — the case that proves
the update meter alone is insufficient.
`Scan_meter.error` maps into `Expr.Eval.error` and into the Region execution error row with one
conversion each. `Expr.Eval.error` also gains `` `Scan_meter_required ``: a stateless
`?on_reduction`-style hook cannot aggregate across calls, and silently defaulting to
`Scan_limits.default` inside `Eval` would break the wider-than-default contract by letting
construction and admission accept a wider custom limit that execution then rejects. So encountering
a scan node with **no** meter is that typed error rather than a silent default. Making
`?scan_meter` optional keeps the 9 existing `Expr.Eval.value` call sites *compiling* unchanged, but
every one of them is updated in this work to pass the meter its reset unit requires —
`region_execution.ml:133,150,175` and `region_eval.ml:62,81,95` (per key, or per `value_at`
invocation), `schedule.ml:51` and `kernel_eval.ml:227,289` (per output coordinate). The optionality
is a migration convenience, not a set of sites left alone.

**Four reset units, each with an owner. `lowered` stores only immutable limits — never a meter.**

- **Standalone `Expr.Eval.value`** — one meter for the whole raw value.
- **Region materialization** — **one meter per region key**, shared by every local body, every
  vector lane, every nested `Scan_at`, and the emitter, matching the declared `per_key` unit. This
  is load-bearing: a scan-backed Region local is executed by the new loop in
  `Region_execution.evaluate_locals`, not by evaluating a `Scan_at`, so its `steps * width` updates
  happen **outside** `Expr.Eval` and that loop calls `charge_update` before each lane update; and
  the loop already makes separate `Expr.Eval.value` calls per scalar body
  (`region_execution.ml:133`) and per vector lane (`:150`), with another for the emitter (`:175`),
  so a per-call meter would reset three ways inside one key. `Region_eval` mirrors the same unit
  across its own split calls (`region_eval.ml:62,81,95`) so reference and production agree.
- **`value_at` — one fresh meter per invocation**, shared across every local evaluation and the
  selected emitter within that call. This path is easy to miss and is live:
  `Kernel_eval.eval_value` calls `Region_execution.value_at` for an on-demand Region value
  (`kernel_eval.ml:230-237`), and the public `Kernel_eval.value_at` reaches it independently of
  fusion admission. Neither obvious alternative works — calling `Expr.Eval.value` with no meter
  makes every scan-backed value fail as `Scan_meter_required`, while storing a *mutable* meter in
  `lowered` makes separate `value_at` calls history-dependent so they eventually exhaust one
  another. Per-invocation is also what the operation already claims to be: `Region_eval.value_at`
  evaluates all of the key's locals and then emits, and `region_eval.mli` calls it a fresh scalar
  projection. `Region_eval.value_at` takes the same rule.
- **Pixel execution — one fresh meter per output coordinate.** A Pixel program is the singleton
  partition, so one coordinate *is* one key; reusing one mutable meter across the tensor would
  silently convert the per-key limit into a tensor-wide total and reject a multi-element output
  after enough individually valid pixels.

**Pixel propagation needs a contract change, not just a parameter.** `Schedule.ground` calls
`Err.or_raise ~pp_error:Expr.Eval.pp_error` once per pixel inside `Tensor.materialize`
(`schedule.ml:42-53`), whose comment explains the deliberate wrapper-drop: `Tensor.materialize`
takes `Vec6.coord -> float`, and threading a result through it would make Direct — the hot path —
pay for the symbolic one. Passing a meter through unchanged would therefore turn
`Updates_exhausted`/`Scan_meter_required` back into exceptions, contradicting
`Stage_program.ground`'s move to `Err.t`. So `Schedule.ground` creates one meter **inside** each
pixel callback and returns an `Err.t` by escaping from `Tensor.materialize` with `Err.Escape` —
the sanctioned mechanism, and exactly the shape `Region_execution.materialize` already uses
(`region_execution.ml:180-201`). Update that comment to record why the contract widened: a resource
limit is a typed outcome, unlike `Dim.index`'s `Invalid_argument` programming error. Both Kernel
Pixel arms (`kernel_eval.ml:227,289`) take the same fresh-per-coordinate rule. And because
`Region_execution.t = Pixel_loop of Expr.Value.t | Region_loop of lowered` exposes a raw value in
the Pixel branch, `lower` must carry the validated limits on **both** branches, not only inside the
abstract `lowered`.

Also provide `Expr.Scan_admission.check : limits:Scan_limits.t -> Value.t -> (unit, Scan.error)
Err.t` over a whole `Value.t`, seeing every enclosing reduction context and aggregate occurrence;
Region preflight requires it. Tests: a previously constructed scan wrapped in a **constant** outer
reduction and in a **dynamic** outer reduction — statically rejected where provable, bounded by the
meter otherwise. Plus a **multi-coordinate Pixel scan where every coordinate consumes exactly the
limit and the whole tensor succeeds** (proving per-coordinate reset), and a typed exhaustion test.

**Static measures — admission and cost reporting, not guarantees.** They let a compact AST be
rejected before execution, which the Region path needs.

| Field | Bounds |
|---|---|
| `max_local_slots` | Region trace **storage**: total slot count, `(steps+1)*width` per scan local |
| `max_scan_state` | **Peak live** scan state |
| `max_scan_updates_per_key` | Recurrence **iterations** for one Region key |
| `max_scan_updates_total` | Summed `keys * per_key` across a Kernel's logical values |

The measure is update *count* and makes no runtime claim: an update body may hold an arbitrary
`Reduce`, and `max_size` bounds syntax nodes, not iteration counts. **Record the pre-existing gap**
— unbounded reduction extent is not introduced by scans and is out of scope. A nested scan's
updates are multiplied by an enclosing reduction's extent when `hi - lo` is statically constant; one
beneath a statically unbounded reduction is rejected. LSTM's own scans sit at Region-local top
level, so this costs the target nothing. Narrow the "identical `width`/`steps`, different reduction
extents get the same budget" test to reductions containing **no** scans, and add a
nested-scan-under-reduction boundary test.

```
U(d)    = width * init_updates(d) + steps * width * (1 + update_updates(d))
per_key = Σ scalar-RHS U(d) + Σ vector-RHS extent*U(d) + Σ Rhs.Scan U(d)
        + outputs_per_key * Σ emitter U(d)
S(d)    = 2*width + max(nested standalone state in init, in update)   (* executable Scan_at *)
T(d)    =           max(nested standalone state in init, in update)   (* trace RHS *)
scan_peak = 0                                                        if the program has no scan
          = total_slots + max(trace-RHS T(d), scalar/vector/emitter S(d))   otherwise
```

**The scan-state measure is zero for a program with no scan** — no trace RHS and no executable
`Scan_at`. Without that guard the formula is unconditional, so an existing scan-free Region program
with any scalar or vector local (SDPA's own `s`/`p` vectors, for instance) has all scan terms zero
but `total_slots > 0`, and setting `max_scan_state = 0` to *disable the new feature* would instead
reject pre-existing Region computations — while also making `max_scan_state` shadow
`max_local_slots` where no scan state exists, contrary to the table's separate dimensions. When a
scan *is* present, `total_slots` enters as the resident baseline, because the slot array stays live
while the scan runs.

A trace RHS writes directly into its own slot range — row `s+1` written while row `s` is read from
the same array — so it allocates no old/next buffers, while every executable `Scan_at` does. Updates
sum over *occurrences*: two projections carrying the same inline descriptor execute twice. A cached
`Local_scan_at` read is constant-time and is **not** charged. All products and running sums use
`Int64`-checked or saturating arithmetic — `lib/expr*` is jsoo-reachable. Regression for the guard:
a scan-free Region program with scalar *and* vector locals is admitted with **both scan limits set
to zero**, while the otherwise-equivalent program containing a scan is rejected. Boundary test:
configure the state limit **between** `total_slots + T(d)` and `total_slots + 2*width + T(d)`, so an
accidental double charge is observable. The trace product belongs to `max_local_slots` alone.

**A validated token, because no execution API is a gate.** `Eval_direct.region_result` goes
`Region_computation.program` → `Region_execution.materialize` (`eval_direct.ml:100-141`) and
`Stage_program.ground` materializes without checking (`stage_program.ml:60-80`). Nor is
`Region_computation.program` sufficient: the per-key measure depends on an output shape the executor
is handed *later*. Today `Region_execution.lower : Region_program.t -> t` takes no limits while
`materialize`/`value_at` accept any `~output_shape` (`region_execution.mli:14-33`), and
`region_execution.mli:31` already *claims* callers pass "an already-validated lowered Region
program" — nothing enforces it. Make the claim true:

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

`lowered` stores the validated shape and limits, and is what the per-key meter is built from;
`materialize`/`value_at` stop accepting a shape. Mirror in `Region_eval`'s public entries, and
invoke `preflight` from `Region_computation.program`, `Stage.check` and `Kernel.create`.

**`lower_region` must change too, or the token proves nothing.** It is currently
`Region_program.t -> lowered` (`region_execution.mli:15-19`) — a second public constructor that
mints the supposedly validated type with no output shape, no limits and no preflight, so
`materialize`/`value_at` could not rely on the invariant. Give it the same validated,
result-valued signature (keeping it for its stated purpose: a caller that already knows
structurally that the program is not a plain pixel expression). Migration is six call sites: the
`lower_region` use in `kernel_eval.ml:170`, and the `lower` uses in `eval_direct.ml:136`,
`native4d/eval_direct4.ml:141`, `stage_program.ml:71`, `test/native/region_program_test.ml:219`
and `test/native/region_compute_test.ml:699`. (`region_eval.ml:17` is a comment.)

One wrinkle to handle explicitly: `Kernel_eval.converted` lowers a program whose output
`Region_program.with_output` has already rewritten with `Result_conversion.apply`
(`kernel_eval.ml:166-173`), so a proof attached at `Kernel.create` time covers the *unconverted*
form, not the one actually lowered. Either revalidate inside `lower_region` — the simpler choice,
and cheap since preflight is shape arithmetic — or have `Kernel.create` validate the converted
form and store the proof-bearing artifact in the Kernel. Do not leave a public unchecked
constructor in place as the reconciliation.

**`Stage_program.ground` gains `?limits`, preflights every stage before materializing the first, and
returns `(…, error) Err.t`** — raising on a resource limit would cross an `Err.t` into an exception
boundary by hand, which CLAUDE.md forbids. **The commit must carry the whole migration: 42 call
sites across 12 files** — `lib/native_op_walk/native_verify.ml` (the one library caller, which
propagates) plus `test/native/graph_symbolic_{activation,pointwise,shape,norm,pad_slice,pool,conv,
combine}_test.ml`, `test/native/{stage_program,kernel_eval}_test.ml` and
`test/native4d/compute_test.ml`, whose sites unwrap with the sanctioned `Err.or_raise ~pp_error`.
Nine further textual hits (51 total across `.ml`/`.mli`) are comments or interface references.

**`max_scan_updates_total` is narrowed to Kernel execution** and says so rather than implying a
guarantee it cannot enforce: there is no whole-graph choke point before `Eval_direct.run` and
`Stage_program.ground` begin materializing. `Kernel.create` sums across the Kernel's logical values;
Direct and Stage ground are covered by the per-key bound and the runtime meter.

**Numbers — three censuses, before the defaults are chosen.** Arithmetic over known corpus shapes,
so they belong in the design commit and are re-verified at M6: (1) max per-key count for one
admitted Region program — Sequencer2D `2 × 16 × 192 ≈ 6.1k`; (2) summed `keys * per_key` over the
live logical values after the liveness/selection stage Kernel adaptation uses — batch keys (16 or
32) multiply each, one program per live `output`/`h_n`/`c_n`, 36 LSTM occurrences; (3) the Direct
path's total, recorded separately since it materializes every node output.
`max_scan_updates_per_key` derives from (1) with stated headroom, `max_scan_updates_total` from (2),
with (3) documenting what the Kernel-only narrowing leaves uncovered. Add a corpus test asserting
the default total admits the intended landing while a deliberately tighter total rejects it.
`max_local_slots` must clear ~6500, so its default starts at 8192. Slot and state ceilings come from
bytes and peak live arrays at 8 bytes/float, measured natively and under node, reported the way
`Hard.depth`/`eval_depth` already are via `test/native/depth_probe.ml`. Do not borrow `Hard.size`'s
magnitude — its unit is expression nodes.

**Distinct typed errors that survive the operation boundary.** One per measure: total local slots
(the renamed `Local_words_over_limit`), peak scan state, updates per key, and — on `Kernel` — total
updates. An oversized trace is reported by `checked_slot_total`, not by the Scan constructor, and
per-key aggregation is a Region-level failure, so preserving only `Scan.error` would not satisfy the
exit condition. `Region_context.program` is `Err.map_error (fun _ -> Invalid_program)`
(`region_context.ml:67`) and `Region_computation` maps that onward (`:53-58`): **wrap the whole
`Region_program.error` rather than selecting one case**. Tests assert the exact payload through
`Region_computation.program`.

**Plumbing.** Add all four fields to `Kernel.Limits.t` with `Hard` counterparts
(`kernel.ml:82-110,148-152`, `kernel.mli:70-130`), validated by `create`. `Kernel.Limits.create`
also builds and stores an `Expr.Scan_limits.t` from `max_scan_state -> max_state` and
`max_scan_updates_per_key -> max_updates` (the aggregate total has no meaning for one descriptor),
so `Kernel.Limits.scan_limits : t -> Expr.Scan_limits.t` is a **pure accessor that cannot fail**.
`Kernel.Limits.default` derives its scan fields from `Expr.Scan_limits.default` so the two cannot
drift; a test asserts the derivation. The same configured value must reach symbolic construction:
`Eval_symbolic.run` hardcodes `Kernel.Limits.default` (`eval_symbolic.ml:50`) while
`Region_kernel.of_graph ?limits` passes its limits only to `Kernel_adapt` (`region_kernel.ml:1-2`),
so thread `?limits` through `Eval_symbolic.run` and pass the same value from
`Region_kernel.of_graph`, with tighter- and wider-than-default regressions through Direct, Symbolic
and `Region_kernel.of_graph`. Switch `checked_slot_total` (`region_program.ml:227-239`) to bill
`max_local_slots` and rewrite its comment — it argues the opposite. Also thread the new fields
through the **11 `Kernel.Limits.create` invocations in 4 test files**.

### 1d. Region: scan-backed locals, and specialization re-measurement

- **Make the local's RHS a variant** — scalar expression | vector body | scan descriptor — with
  extents derived from the RHS. `{ shape; value }` cannot honestly carry a scan.
- `Region_program.Builder.scan` per 1b, handing back a reader built with `Value.local_scan_at`.
- `Region_program.check`: per-child masking from the descriptor traversal; shape agreement gains a
  third read kind; dependency order and region-invariance keep applying.
- `Region_execution.evaluate_locals` (`:100-161`) gets a scan arm running the descriptor **once per
  key**, calling `Scan_meter.charge_update` before each lane update; `slot_reader` (`:85-98`) gains
  a 2D accessor resolving `Local_scan_at` **by local id** over the resolution variant. **Dispatch on
  the declared RHS, never the slot count.**
- **Specialization must re-measure.** Region validation admits `Local_scan_at` as a constant-time
  cached read; `specialize_pixel` then replaces it with an inline `Scan_at`, so in a chain a later
  scan that cheaply read an earlier trace becomes a descriptor that **re-executes** the earlier scan
  inside its own update. Cost multiplies past the admitted limits while the AST stays compact — and
  today's contract carries only `max_size`/`max_depth` (`region_program.mli:39-40`). So
  `specialize_pixel` takes scan limits and re-measures the completed rewritten value, with a typed
  error, threaded through `Stage.pixel_body` and `Ground_eval`. Likewise
  `substitute_loads`/`substitute_locals` take `limits` and re-measure; `freshen`, `alpha_normalize`,
  `substitute_output`, `substitute_reducer` and `map_sources` stay cheap.
- **Stream `Region_eval.materialize`.** It retains one `values` array per key in an unbounded
  `Hashtbl` (`region_eval.ml:118-144`), so peak scratch is `keys * total_slots` while
  `max_local_slots` bounds only one array. Restructure it to stream keys as
  `Region_execution.materialize` already does; it is the reference path, so this changes
  performance, not results.
- Mirror in `Region_trace`, `Region_program.pp` and `Fold`.

**Testing that metering is shared needs two distinct tests, because the obvious one is vacuous.** A
program whose locals and emitter individually fit but jointly exceed `max_scan_updates_per_key` is
rejected by `preflight`, which aggregates exactly those occurrences — it never reaches a runtime
charge and proves static aggregation, not meter ownership. So:

- keep that case as a **preflight** regression; and
- add a **runtime** test that observes the shared meter directly: extend the existing optional
  `counters` instrumentation (`region_execution.ml:16-22`, already "optional test instrumentation …
  ordinary execution passes none") with a `scan_updates` field, run one valid key containing
  multiple locals *and* an emitter, and assert the **exact combined charge** — equivalently, the
  meter's final `remaining` — in **both** `Region_execution` and `Region_eval`. A test seam that
  injects a deliberately smaller meter than the admission limit is the alternative form.
- add a **`value_at` independence** regression: call it twice for two coordinates sharing one
  Region key, each call consuming exactly the limit, and assert both succeed with the same result
  and the same charge count. This is what fails if a mutable meter is ever stored in `lowered`.

**Other tests** (`test/native/region_scan_test.ml`): materialized trace vs. scalar reference;
partition coverage and no duplicate outputs; one scan start per key; a scan reading an earlier
trace; chained scans catching recursive replay; dependency-order rejection; specialized AST size
independent of `steps`/`width`; slot-, state- and update-budget rejection; bad cached projections;
many keys; **a two- or three-scan chain inside the efficient Region budget whose specialized replay
exceeds the update limit — efficient materialization must succeed while specialization returns the
budget error without starting the replay**; both substitution regressions, one evaluated directly
through `Expr.Eval`; production cost assertions.

### 1e. Grounding, fusion, and rendering

**One meter per proof attempt, tracking the current pair size — not cumulative work.**
`Map_verify.Budget.max_nodes` caps the **pair**, recomputed each round as
`Ground_expr.size lhs + size rhs` (`map_verify_check.ml:244-246`) — *after* both terms are built,
while the roots come from independent, entirely unmetered `Ground_eval.at` calls (`:382-383`) and
`expand` can call `body_at` with part of the allowance already spent. Three rules make a meter
match that quantity:

- **Accounting is on the current `lhs_size + rhs_size`.** Expansion does not append a disjoint
  tree; it *replaces* cells inside terms whose sizes were already charged, so replacing a one-node
  `Cell` with a body of size `n` changes the pair by `n - 1`. Charge that delta. Unchanged
  ancestors are already counted and must never be recharged, and a cached body must not be debited
  in full at each replacement — either mistake accumulates historical work and rejects a pair whose
  actual logical size is still within the limit.
- **The meter's scope is one proof attempt, not `compare_at`.** `compare_at` runs an unqualified
  Structural attempt and, unless it proves, reruns with constants bound
  (`map_verify_check.ml:399-414`); the first attempt's non-proof verdicts are *discarded by design*,
  as a soundness requirement. The two attempts build different pairs, so each gets the **full**
  limit — a meter owned by `compare_at` would let the discarded attempt starve the authoritative
  one.
- **Within an attempt the two roots share that one limit**, and it carries into every
  `expand`/`body_at` of that attempt. Return the cached logical size with grounded terms so the
  verifier stops rediscovering it with `Ground_expr.size`.

**Two accounts, because retained pair size and construction scratch are different quantities.**
Every unrolled construct is metered, charging before construction — ordinary nodes, `Reduce`
unrolling (`ground_eval.ml:295-318`), `Max_pool` window expansion, and scans alike, threaded
through `at`/`body_at`/`ground` — but not all of it against the same budget. Unrolling builds
logical trees that the result does not retain: a `Scan_at` computes two rows across every lane and
returns one projected lane, and the max-pool grounder builds **both** `best` and `best_index` at
every window position before returning only the requested one (`ground_eval.ml:387-412`). Charging
those against the pair meter would measure temporary work, so a final `lhs_size + rhs_size` at or
below `max_nodes` could still fail; charging only the retained result would leave the explosive
temporary construction unbounded, contradicting the requirement to stop *before* building a
crossing subtree. So:

- the **pair meter** (`Budget.max_nodes`) receives only the **returned** logical tree — that is
  what the verdict's budget has always meant; and
- a separately named **scratch bound** (a new `Budget` field) covers every constructed node
  including discarded ones: full-row scan evaluation, unselected lanes, and the discarded max-pool
  accumulator. Its exhaustion is its own `Unproved` reason carrying that limit — not `Max_nodes`,
  whose payload states the pair cap.

The max-pool asymmetry pre-dates scans; scans only make it load-bearing.

**Charge logical size, not allocations.** `Ground_expr.t` is a plain tree with no sharing node and
no memoization: `size`, `cells` and `project` all recurse (`ground_expr.ml:129-138,205-232`). A
`previous_at` read embeds the *same* prior pointer at many sites, so a dense coupled recurrence
allocates almost nothing while every later traversal sees a tree growing exponentially across steps.
Carry a checked logical size with each grounded state value and **debit that cached size at each
embedding**. (A bounded DAG/let form in `Ground_expr` is the escape hatch; it needs every consumer
made sharing-aware.) Never call an unmetered `Ground_expr.size` to discover the cap was exceeded.

**Verdict mapping.** The generic conversion yields `Unproved (Eval …)` (`map_verify_check.ml:173`);
the budget outcome is `Unproved (Max_nodes n)` (`:246`, type at `map_verify.mli:194`). Add a typed
grounding-budget error mapped specially to `Max_nodes`, carrying **the budget** — an early stop, so
the limit convention applies. Verifier callers take `Map_verify.Budget.max_nodes`; define
`Ground_eval.default_budget` for other callers at the same value.

Tests: an oversized **reduction** at the root; an oversized scan at the root; an oversized scan
behind a cell expanded in a later round; a **dense coupled recurrence whose prior lanes are read
repeatedly**; **each root consuming between ½ and ¾ of `max_nodes` with the pair exceeding it**; and
**a later round where the first expanded cell consumes most of the allowance and the next would
cross it**. All must stop before constructing the crossing subtree and report
`Unproved (Max_nodes budget)`.

Two further tests pin the accounting model itself, and both fail under a naive cumulative meter:
**a pair that sits exactly at `max_nodes` and still succeeds across several expansion rounds**
(proving replacement deltas, not re-debited history), and **a discarded Structural attempt that
nearly exhausts its limit while the constant-bound attempt still receives a fresh full allowance**.

Three more separate the two accounts: **a sparse multi-lane scan whose projected lane stays small
succeeds at the pair limit**, **`Max_pool.Value` succeeds at the pair limit** despite its discarded
index accumulator, and **discarded intermediate scan state still trips the scratch bound** — the
last being the one that proves the scratch account is not merely decorative.

**Reject scans in the shared fusion admission rule.** `Kernel_elab.admit` rejects only when
`pixel_expression` is `None` (`kernel_elab.ml:71-76`), and its comment says the planner and direct
callers share that one rule — a singleton `Scan_at` program is still a pixel expression. Add an
explicit scan/recurrent-effect summary; test both planner-driven and direct admission.
`Fusion_plan`'s `pointwise` test (`fusion_plan.ml:146-148`) covers the planner only because
`Fold.binders` reports scan binders.

**Give Region locals an RHS rendering path.** `me_detail.of_value` builds one expression root per
`local.Region_local.value` (`me_detail.ml:78-100`) and `Region_trace.pp_local` prints
`Expr.Pp.value local.value` (`region_trace.ml:94-95`); neither works for a descriptor. Render it
with `init` and `update` as scoped children and use that in both; do **not** fabricate a projection
at dummy indices. The explorer fixture must include an **unspecialized** scan local.

**JS backends.** `make jsoo.runtest` and `make jsoo.inline-runtest` catch different things.

### Stage 1 exit condition

A synthetic coupled scan agrees across `Region_execution`, `Region_eval`, `specialize_pixel` and
`Ground_eval`; its specialized AST size is independent of `steps` and `width`; scan starts are one
per key, and instrumentation shows the **exact combined charge on one shared per-key meter** in both
executors; repeated `value_at` calls on one key are independent, each succeeding at exactly the
limit; a multi-coordinate Pixel scan consuming exactly the limit at every coordinate succeeds,
proving per-coordinate reset; **the meter bounds a raw composed `Value.t` that no static gate saw,
including a scan under a dynamic outer reduction, and a wider-than-default program both constructs
and executes under the wider value**; a descriptor built under a wider `max_state` is refused by a
narrower meter even when it performs no updates (the nested row-zero case); no public constructor
can mint a `lowered` without shape, limits and preflight; oversize traces, scan state, per-key and
total updates
(including the specialized-replay and both substitution cases) and out-of-range projections fail
with **typed errors whose exact payload survives to the operation boundary**; the same over-limit
program is rejected through Direct, Stage ground, Region execution and Kernel, each before starting
a scan; grounding terminates `Unproved (Max_nodes …)` on the oversized-reduction, scan-at-root,
later-expansion, dense-reuse and both split-budget paths; fusion rejects a scan through both entry
points; `Region_eval.materialize` holds one slot array at a time; an unspecialized scan local
renders in the explorer and the trace; both JS checks pass.

"Round-trips" means **structural compare/hash equality after freshening** — Expr, Region and Stage
programs have no parser or codec.

### Suggested commits (`_ai_/` is gitignored and its own repo — ledger updates go there separately)

1. `fixup! 99872ea` — `Region_eval` declared-shape dispatch + `Wk = 1` regression.
2. `.ai/` — representation, construction and error contracts, the freshening obligation, the meter
   API with its charge operation/boundary rule/four reset units/propagation, the static measures
   with their three censuses and reduction-context rule, the validated-execution token,
   specialization and rewrite re-measurement, grounding meter, RHS rendering, `divmod`.
3. `feat(expr): add a bounded ordered scan` — 1b plus `Scan_meter`, the admission check and the
   `Expr.Eval` metering, plus `test/expr/scan_test.ml`. `dune build lib/expr` is the checkpoint.
4. `feat(native): budget Region slots and scan state/updates` — 1c, including `preflight`, the
   validated `lowered` with limits on both branches, **both `lower` and `lower_region` made
   result-valued with their six call sites**, the `scan_limits` accessor, the `Schedule.ground`
   `Err.Escape` contract change, meter propagation to both Kernel Pixel arms and to
   `Region_execution.value_at`/`Region_eval.value_at` (a fresh meter per invocation, with the
   repeated-call independence regression), `?limits` through
   `Eval_symbolic.run`, and the `Stage_program.ground` signature change with **all 42 call sites in
   12 files**.
5. `feat(native): add scan-backed Region locals` — 1d.
6. `feat(native): meter grounding, reject scan fusion, render scan locals` — 1e.

---

## Stage 2+ — `lstm.input` itself (outline)

Not started until Stage 1 is landed and green. Follows `_ai_/lstm-plan.md` §§2, 4–7 with the
corrections applied.

**Decide the modulo encoding first (correction 8).** One shared `divmod` helper: remainder as
`x - K * floor_div_pos(x, K)` from the existing `Add`/`Scale`/`Floor_div_pos` delta nodes, converted
back with `Clamp_low`, never `Assume_position`. `Clamp_low` is sound rather than a papering-over:
for `x >= 0` and `K > 0` the remainder is provably in `[0, K)` so the clamp never fires — state that
argument in the design record. Reuse the one helper for `lane mod K`, `j mod K`, `c mod K`,
`a mod R`, and test with `K > 1`, `R = 2`, `Q > 1` and unequal dimensions so a quotient/remainder
mixup is observable.

- **M0 — binding.** `op "lstm" ~overload:"input"`, on the probe already run. Precedent: `cat`/`stack`
  take `Tensor[]` (`aten_ops_gen.ml:75-76`), `native_layer_norm` returns three tensors (`:87`). Add
  the `Walk_meta`/recipe from correction 7. Oracle fixtures across layers, biases, directions,
  layouts, `dropout = 0 / 0.5 / 1`.
- **M3 — the Native op.** `lib/native/ops/lstm.ml`; register in `graph_ir`, the registry,
  `graph_shape` (three shapes), `eval_op` (ordinal-selected), `graph_builder`, `output_transfer`.
  Widen `Region_computation` past correction 1; one Region program per output ordinal. Keep `Lstm`
  off `Const_ssa.allows`, documented.
- **M3b — stacked layers and configuration coverage** (`_ai_/lstm-plan.md:363-381`). Required for the
  landing, not a follow-up.
- **M4 — both importers.** `op_bridge_recurrent.ml`, `native_interp_lower_recurrent.ml`. Tensor-list
  precedent: `Interp_decode.tensors_arg_result` (`interp_decode.ml:429-434`) and
  `native_interp_decode.ml:87-96`. `verify.ml:204-221` permits a fixed tuple to expose *fewer*
  outputs — assert the output count explicitly.
- **M5 — Native4D counterpart.** Reuse the Native payload if it names no axis and carries no
  `Shape4.t` — the `Sdpa` precedent. Rank-3 operands sit on `H/W/C` with `N=T=D=1`
  (`aten_shape.mli:1-4`), so `_ai_/lstm.md:163-165`'s "reject at Native4D" is wrong. Native4D has no
  `Discard` (`op.ml:11-18`), so dead state outputs must be gone first.
- **M6 — evidence and ledger.** Native op walk; both corpus shape families; re-verify the three
  update censuses against the real programs; regenerate `make pt2.json-model-support`. Clearing 36 of
  911 nodes does not mean the graph builds — record Sequencer2D's actual next frontier separately for
  Native, Native4D and Kernel, and preserve the scoreboard (92/58/66/44) until a sweep shows a
  change.

---

## Verification

```sh
opam exec -- dune build lib/expr                                 # the 1b checkpoint
opam exec -- dune build
NO_COLOR=1 opam exec -- dune runtest test/expr test/native      # Stage 1's own suites
NO_COLOR=1 opam exec -- dune runtest                             # whole tree
make precommit                                                   # build+format+runtest+file-size+whitespace
make jsoo.runtest && make jsoo.inline-runtest                     # the new Expr node crosses the JS boundary
make pt2.json-model-support                                       # Stage 2 only; expect no diff before M6
```

Run the non-promoting form first so failures stay visible, and review any golden diff before
`dune promote`.
