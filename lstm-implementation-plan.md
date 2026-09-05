# `lstm.input` — implementation plan

## Context

Status: planned. Land the bounded scan primitive first, then the full inference-only,
three-output LSTM implementation, including stacked layers and Native4D. This plan is the
implementation contract; [lstm-plan.md](lstm-plan.md) supplies the tensor and arithmetic details.
The open contracts identified in [lstm-review.md](lstm-review.md) are specified below.

`sequencer2d_s` (911 nodes) is `native_builds:false`, blocked at `torch.ops.aten.lstm.input` — 36
occurrences, its first frontier (`test/data/pt2_json_model_support.jsonl:82`).

`Semantics.SEMANTICS` exposes `sum` and `max_reduce`, each with a fixed combine over
independently computable terms. LSTM requires an ordered recurrence over two hidden vectors.
It cannot use those reductions as a state-carrying fold. Preserve evaluation order, including
floating-point association; mathematical associativity does not authorize reassociation.

The scan is a new `Expr` AST node, authored through the existing Region computation path.
Storage, peak state, update counts, and grounding construction have separate budget dimensions.

Broader design proposals are in [project_design_ideas.md](project_design_ideas.md). They are
separate from the work required to land LSTM.

## Prerequisites

- **Fix a live defect in `Region_eval`.** `region_eval.ml:59` dispatches locals on the numeric slot
  count (`| offset, 1 -> ...`). `Region_execution` already dispatches on the declared shape.
  A `Vector` local of extent 1 (SDPA at `Wk = 1`) takes the
  scalar branch with no `~reducer` bound and raises `Unbound_reducer`. Reachable via
  `Region_execution.value_at` (`:203-204`) → `Kernel_eval` per cell (`kernel_eval.ml:231-237`).
  Dispatch on the declared shape and add a `Wk = 1` regression on `value_at`.
- **ATen feasibility probe** — static-link check plus a tiny three-output oracle, no LSTM landing.
  Execute the binding: successful linking alone does not establish that its kernel path avoids
  throwing stubs.

---

## Stage 1 — the scan primitive

### 1a. Record the contracts

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
| `expr.ml` | façade exports `Scan`, `Scan_limits`, `Scan_meter`, `Scan_admission` |

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
(`expr_api.ml:501-503`) gains a `Scan of Scan.t` case. Build `lib/expr` and compile a consumer
outside that library which constructs and inspects the descriptor through public APIs.
After Region scan locals land, extend that fixture to build, check, print and execute a
scan-backed Region program through the public surface.

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
    | Bad_steps  of int
    | Bad_width  of int            (* the offending value *)
    | Prev_in_init
    | State_over_limit   of { limit : int }     (* early stop: the LIMIT *)
    | Step_in_init                 (* payload-free: a freshly minted internal ordinal *)
    | Unbounded_reduction_context
    | Updates_over_limit of { limit : int64 }
  val pp_error : Format.formatter -> error -> unit
end
```

**Freshen prebuilt fragments before composition.** Thread `run_from` for identities minted by
callbacks. Independently built fragments can already contain colliding reducer/local identities;
freshening the combined tree cannot recover which binder a captured reference meant. State this
obligation on `Builder.scan`. The regression freshens a prebuilt fragment from the shared builder
state inside the callback, before combining it, then asserts all three scan binders are fresh and
the captured free read still names the earlier Region local. Extend freshening's avoidance of free
identities to both namespaces so it also avoids the captured local.

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

**Projection evaluation runs exactly `row` time steps.** Two buffers hold the completed and
next row. Bounds-check row/lane, run `row * width` lane updates, then read the selected lane.
Row 0 returns initialized state; row `steps` returns final state. Nested scans add their own
charges to the same meter.

**Exact projection error contract.** Add the following inside `Expr.Eval`. The optional local id
distinguishes a cached trace from an inline descriptor; `extent` is the exclusive upper bound
(`steps + 1` rows or `width` lanes). It is observed shape data, not a budget limit.

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
type scan_reader =
  Local_var.t -> row:int -> lane:int -> (float, scan_error) Err.t
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

`` scan_error e = `Scan_projection e `` is the single conversion. The evaluator maps callback
results with `Err.map_error scan_error`, preserving detection provenance. `Region_execution`
and `Region_eval` provide readers with this exact type and widen the resulting `Expr.Eval.error`
unchanged; they do not define competing projection errors. A missing callback fails with
`Unknown_local id`. A reader first resolves the id in its trace table (a scalar/vector id is
not a trace), then checks row, then lane, before indexing. Inline `Scan_at` applies the same
bounds rules with `local = None`; cached reads report `Some id`. Compute the row extent and
flattened storage offset with checked arithmetic. Static Region shape checks still reject a
scalar/vector local used as a trace before execution.

Boundary tests assert the entire error payload for unknown ids, negative and upper-bound
indices, simultaneous row/lane failures (row wins), and inline versus cached projections.

### 1c. Scan admission and runtime budgets

**Runtime metering is required even after static admission.** A checked scan can be composed
under another reduction, inserted by a raw rewrite, or passed as a raw `Value.t` to the evaluator.
Construction-time checks cannot bound all those contexts. Export the charge operation because
Native runs trace descriptors directly, outside the inline `Scan_at` evaluator.

```ocaml
module Scan_limits : sig
  type t
  module Field : sig
    type t = Max_state | Max_updates
  end
  module Invalid : sig
    type t = { field : Field.t; value : int64 }
  end
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
    | Updates_exhausted of { limit : int64 }   (* early stop: the LIMIT *)
  val create : limits:Scan_limits.t -> t
  val charge_update : t -> (unit, error) Err.t
  val remaining : t -> int64
  val pp_error : Format.formatter -> error -> unit
end
(* State reservation is INTERNAL to the Expr implementation — not in this signature. *)

```

`t` is abstract and built only through `create`, which rejects negative fields as `Invalid` and
applies the same **exclusive `v >= hard`** rule `Kernel.Limits.create` uses.
The printer renders `Max_state` as `max_state` and `Max_updates` as `max_updates`; strings
are presentation only. Extend `Kernel.Limits.error` with
`` `Scan_limits of Expr.Scan_limits.error `` and use `Err.map_error` to preserve this payload
when constructing its stored scan limits. Test both fields at negative and hard-boundary values
through `Scan_limits.create` and `Kernel.Limits.create`.
**Zero is valid and meaningful for both fields**: `max_state = 0`
forbids every scan, since each reserves `2 * width >= 2`; `max_updates = 0` admits only descriptors
that can never update — those with `steps = 0`, whose trace is just the initial row.
`Kernel.Limits.create` builds and stores the `Scan_limits.t` at construction, so
`Kernel.Limits.scan_limits` is a pure accessor that cannot fail — which removes the question of how
a caller constructs the abstract value downstream.

**Construction and static admission use the descriptor's worst case; only runtime metering is
projection-sensitive.** `U(d) >= steps * width`, so a positive-step descriptor fails construction
and admission under a zero update limit even if a particular projection reads row zero. Test
that same row-zero case at all three boundaries: `Builder.scan` and `Scan_admission.check`
return `Updates_over_limit`; evaluation of a descriptor built under a wider limit succeeds
with a zero-update meter and no charges, provided its initializer contains no updating scan.

**Update boundary rule, stated once:** exactly `limit` charges succeed, and the next charge fails
**before its update body is evaluated**. The standalone `Scan_at` evaluator and both Region
executors call the *same* `charge_update`, so their off-by-one behaviour cannot drift.

**Reserve state before allocation.** On entry to inline scan evaluation, reserve `2 * width`
against the meter's current live state; on exit, restore the previous level on every success,
error and `Err.Escape` path. This enforces the nesting peak even for row-zero projections and
scans in initializers that perform no updates. Check arithmetic before reserving or allocating.

Keep reservation/release internal to Expr so callers cannot under-count state with unmatched,
duplicate or wrong-width releases. Native trace locals write directly into preflighted slots;
they need the public update charge, while Region preflight bounds resident storage and nesting.
Test nested/failed entry, error unwinding and internal release invariants. A descriptor built
under a wider state limit must fail under a narrower evaluation meter, including a nested
row-zero case.

Extend `Expr.Eval.error` with `` `Scan_meter of Scan_meter.error `` and
`` `Scan_meter_required ``. Export
`Eval.scan_meter_error : Scan_meter.error -> Eval.error`, wrapping with the `Scan_meter` tag,
and use it with `Err.map_error` in Expr and Native's direct trace loop.
Encountering an inline `Scan_at` without `?scan_meter` returns `Scan_meter_required` before
reserving state or evaluating bodies. Cached `Local_scan_at` reads require no update charge
and use the reader contract above. For inline projections with a meter, check row then lane
before reserving state. Do not silently choose default limits. Pass the meter to every
production `Expr.Eval.value` call in
`region_execution.ml`, `region_eval.ml`, `schedule.ml` and both `kernel_eval.ml` Pixel arms.

**Meter ownership.** Store only immutable limits in lowered programs.

| Entry point | Fresh meter scope | Calls sharing it |
|---|---|---|
| Standalone `Expr.Eval.value` | Whole raw value | All nested evaluations |
| Region materialization | One Region key | All scalar/vector/trace locals, nested scans and every emitter for that key |
| Region `value_at` | One invocation | All locals and the selected emitter |
| Pixel execution | One output coordinate | The complete pixel expression |

Both Region evaluators use these rules. Their trace loops call `charge_update` before each
lane's update body, since those updates happen outside the inline `Scan_at` evaluator.
Separate `Expr.Eval.value` calls within one key must not create separate meters.
Repeated `value_at` calls remain independent even when their coordinates share a key;
`Kernel_eval.value_at` reaches this path independently of fusion admission.

**Preserve typed Pixel failures.** Make `Schedule.ground` return `Err.t`, using `Err.Escape`
to cross `Tensor.materialize`'s float-returning callback. Replace the current `Err.or_raise`
there and update its comment. Create the meter inside each pixel callback; both Kernel Pixel
arms use the same reset rule. Change `Region_execution.t`'s Pixel branch to carry validated
limits as well as the expression, so custom limits survive lowering on either branch.

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

A scan-free program has `scan_peak = 0`, even if it owns scalar/vector slots: disabling
scans must not reject existing Region operations. When a scan is present, `total_slots` is
the resident baseline. A trace RHS writes row `s+1` directly into its slot range while reading
row `s`, so it allocates no additional old/next buffers. Inline `Scan_at` does allocate them.

Updates sum over occurrences: two inline projections execute twice; a cached `Local_scan_at`
read costs no update. Use checked or saturating `Int64` products and running sums throughout.

Tests: admit a scan-free scalar/vector program with scan-state and update limits zero; reject
its scan-bearing counterpart. Set the state cap between `total_slots + T(d)` and
`total_slots + 2*width + T(d)` to catch double-counting trace row buffers. Charge the trace
product as storage through `max_local_slots` and include those resident slots once in peak state.

**Validate the executable artifact.** `Eval_direct.region_result` goes
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

**Validate both constructors.** `lower_region` also becomes result-valued and checks the same
shape and limits. Migrate the
`lower_region` use in `kernel_eval.ml:170`, and the `lower` uses in `eval_direct.ml:136`,
`native4d/eval_direct4.ml:141`, `stage_program.ml:71`, `test/native/region_program_test.ml:219`
and `test/native/region_compute_test.ml:699`.

Revalidate inside `lower_region` after `Kernel_eval.converted` rewrites the emitter with
`Result_conversion.apply`. Validation of the unconverted form does not cover a changed
executable expression.

**`Stage_program.ground` gains `?limits`, preflights every stage before materializing the first, and
returns `(…, error) Err.t`**. Carry the complete caller migration:
`lib/native_op_walk/native_verify.ml` (the library caller, which
propagates) plus `test/native/graph_symbolic_{activation,pointwise,shape,norm,pad_slice,pool,conv,
combine}_test.ml`, `test/native/{stage_program,kernel_eval}_test.ml` and
`test/native4d/compute_test.ml`, whose sites unwrap with the sanctioned `Err.or_raise ~pp_error`.
Re-enumerate application sites when implementing; counts in a working plan are not a completion
check. Compilation must cover every changed public signature.

**`max_scan_updates_total` is narrowed to Kernel execution** and says so rather than implying a
guarantee it cannot enforce: there is no whole-graph choke point before `Eval_direct.run` and
`Stage_program.ground` begin materializing. `Kernel.create` sums across the Kernel's logical values;
Direct and Stage ground are covered by the per-key bound and the runtime meter.

**Numbers — three censuses, before the defaults are chosen.** Arithmetic over known corpus shapes,
so they belong in the design commit and are re-verified against the finished LSTM programs:
(1) max per-key count for one
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
through `Kernel.Limits.create` callers in `depth_probe.ml`, `fusion_test.ml`, `kernel_test.ml`
and `region_compute_test.ml`.

### 1d. Region locals and specialization

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

**Test admission and runtime ownership separately.** A program whose locals and emitter
individually fit but jointly exceed `max_scan_updates_per_key` is rejected by preflight.
To also establish that execution shares one meter:

- keep the combined over-limit case as a **preflight** regression; and
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

Grounding currently enters an unmetered `ground` through `at` and `body_at`;
`expand` meters its outer traversal but can enter that same unmetered construction.
Update the implementation and the misleading comments in `ground_eval.ml`/`.mli`.
Bound roots and expansion before constructing a crossing subtree, including ordinary
reductions and max-pool.

**Two exact accounts.** Construction uses cumulative fuel, not peak live scratch.
Discarding a row does not refund it. Name that public field `max_ground_nodes`.

| Account | Unit and charge | Reset |
|---|---|---|
| `max_nodes : int` | Current logical nodes in `lhs + rhs`; a new root costs its size, replacing a one-node cell with size `n` costs `n - 1` | One proof attempt |
| `max_ground_nodes : int64` | Cumulative logical node occurrences constructed or copied, including discarded intermediate results | One proof attempt |

Ground the left root then the right root, and expand them in that same explicit sequence;
do not rely on tuple argument evaluation order for charging or error precedence.
Both roots and all expansion rounds share both accounts. The Structural attempt and the
subsequent constant-bound attempt each receive fresh accounts; a discarded attempt must not
consume the authoritative attempt's allowance. A new coordinate/member comparison also starts
fresh. Neither field is a whole-verification time or memory guarantee.

For construction fuel, a newly created node costs one plus any logical subtrees copied from
cached state. Attaching a freshly constructed child, or moving an accumulator into its sole
successor, transfers its already-charged occurrences without charging them again. Reusing a
cached previous lane or cell body costs its cached logical size **at every embedding**, even
if the implementation reuses a pointer. Rebuilding an ancestor during expansion costs one;
retaining an unchanged child transfers it. Unselected lanes, discarded rows, and both max-pool
accumulators consume fuel; dropping them refunds nothing. This avoids charging every growing
accumulator in full at every fold step while still accounting for duplicated subtrees.

The pair account charges only retained results and replacement deltas, never unchanged
ancestors again. Thread its remaining allowance into construction contributing to the retained
tree, reserving the known node/delta before construction. Temporary scan rows and discarded
max-pool results use construction fuel until their selected result is retained. If that
selection exceeds the remaining pair allowance, refuse it before embedding it in the pair.
A cached logical size permits this check without walking the oversized value.

All size additions, products, and charges are checked before arithmetic can overflow, including
under js_of_ocaml. Store a checked logical size beside **every** grounded intermediate.
`Ground_expr` has tree semantics: pointer sharing alone does not bound recursive consumers.
Never use an unmetered `Ground_expr.size` to discover that a recurrence has already exceeded
the cap. A sharing-aware ground IR is a separate project design idea.

**Public grounding API.** Define this in `Ground_eval`, which must not depend on `Map_verify`.
The verifier explicitly converts its larger public budget into these two fields.

```ocaml
module Budget : sig
  type t = { max_ground_nodes : int64; max_nodes : int }
end
module Meter : sig
  type t
  val create : Budget.t -> t
end
module Term : sig
  type t
  val expression : t -> Ground_expr.t
  val size : t -> int64
end
val default_budget : Budget.t
(* Extend [Ground_eval.error] with exactly these two budget cases: *)
(* | `Ground_nodes_over_limit of int64 | `Pair_nodes_over_limit of int *)
val at :
  meter:Meter.t -> Env.t -> Tensor_id.t -> Vec6.coord -> (Term.t, error) Err.t
val expand :
  meter:Meter.t ->
  boundary:(Ground_expr.Origin.t -> Cluster_var.t option) ->
  Env.t -> Term.t -> (Term.t, error) Err.t
```

`at` registers a root in its meter. `expand` replaces that registered root and returns its
successor; the old term is no longer active. Keep registration and replacement in `Ground_eval`
so callers cannot forget the pair delta. Terms belong to their creating meter; passing a
foreign or replaced term is an `invalid_arg` programming error, checked by an internal
identity/generation token. A failed construction ends that attempt; callers cannot probe a
partial frontier or reuse its meter. Internal `body_at`/`ground` receive the same meter.
Non-verifier callers explicitly create a meter from `default_budget` for each independent root
and its expansions, and use `Term.expression` for existing tree consumers.

Nonpositive budget fields behave as zero allowance, refusing the first positive charge;
exactly the limit succeeds. Check pair admission before construction fuel when both reject
the same retained-node insertion, giving deterministic error precedence. Budget errors carry
the configured field value; tests use nonnegative budgets for ordinary boundary cases.

**Profiles and migration.** Add `max_ground_nodes : int64` to the concrete public
`Map_verify.Budget.t` in both `map_verify.mli` and `map_verify_types.ml`. Use these initial
policy values (ten times each pair cap); they are starting allowances, not measured LSTM
capacity claims:

| Profile | Existing `max_nodes` | New `max_ground_nodes` |
|---|---:|---:|
| `Budget.default` | 200,000 | `2_000_000L` |
| `Budget.cumulative` | 1,000,000 | `10_000_000L` |
| `Budget.release` | 50,000 | `500_000L` |
| `Effort.Quick` | 20,000 | `200_000L` |
| `Effort.Standard` | 50,000 | Uses `Budget.release` |
| `Effort.Thorough` | 200,000 | `2_000_000L` |

`Ground_eval.default_budget` is `{ max_ground_nodes = 2_000_000L; max_nodes = 200_000 }`.
Derive the matching `Map_verify.Budget.default` fields from it. Keep other existing profile
fields. A record update changing only `max_nodes` retains its construction allowance; the
ratio is not applied implicitly at runtime.

Migrate all five default/profile literals and the full external literal in
`test/native/verify_boundary_test.ml` (use `2_000_000L` there). Audit other full literals and
budget overrides. Update `Budget.pp` to print
`{coords<=%d ground_nodes<=%Ld nodes<=%d rounds<=%d sample=%a}` and review affected goldens.
Migrate every `Ground_eval.at`/`expand` caller from raw trees and `~budget:int` to the meter
and term contract. `map_verify_check.ml` threads the meter through `attempt`/`settle` and
uses cached root sizes. Add profile/default printer coverage if no existing golden exercises it.

**Exact verdict mapping.** Add `Max_ground_nodes of int64` to `Unproved.t` in the public
signature and implementation. Extend the single `unproved_of_eval_error` conversion:

| Grounding error | Verdict |
|---|---|
| `` `Ground_nodes_over_limit limit `` | `Unproved (Max_ground_nodes limit)` |
| `` `Pair_nodes_over_limit limit `` | `Unproved (Max_nodes limit)` |
| Other existing errors | `Unproved (Eval error)` |

Both budget payloads are the configured limit. The current post-expansion path supplies the
observed size to `Max_nodes`; replace that path too, so its meaning is consistent.
`Unproved.pp` prints `over max_ground_nodes (%Ld)`, `reason` returns `over max_ground_nodes`,
and `reasons` gains that exact label. Update `test/native/outcome_label_test.ml`'s complete
verdict samples, canonical label golden, and counts (17 becomes 18). Check derived
`Verdict.labels`, `Outcome.label`, report/tally and explorer expectations. Keep affected
variant declarations and their exhaustive matches/lists in the repository's alphabetical order.

**Grounding regressions.** Isolate each budget by giving the other enough allowance.

- Pair exhaustion: oversized root reduction, root scan, scan behind a later expanded cell,
  dense repeated-lane recurrence, two roots each using between half and three quarters of the
  cap, and a later expansion where successive cells jointly cross it. Expect
  `Unproved (Max_nodes limit)`; retained construction stops before the crossing insertion.
- Pair accounting: a pair exactly at its cap survives several replacement rounds when
  construction fuel suffices. A sparse scan projection and `Max_pool.Value` fit at the pair
  limit despite discarded intermediates.
- Construction exhaustion: discarded scan rows or max-pool accumulators exhaust fuel while
  the pair fits. Repeated cached embeddings spend their logical size; discarded rows do not
  refund fuel. Expect `Unproved (Max_ground_nodes limit)` and the exact payload.
- Reset and boundaries: both attempts receive the full two-field budget; both roots share it
  within an attempt; exactly-limit and next-charge cases distinguish the two reasons.

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

### Stage 1 exit conditions

- A synthetic coupled scan agrees across production/reference Region execution, specialized
  Expr evaluation and grounding. Its specialized AST size is independent of width and steps.
  Freshening preserves structural compare/hash equality; Expr/Region/Stage have no codec.
- Instrumentation shows one trace execution per key and the exact combined update charge
  across locals and emitters in both evaluators. Repeated `value_at` calls are independent;
  multi-coordinate Pixel execution receives a fresh allowance at each coordinate.
- The runtime meter bounds raw composed values, including scans under dynamic reductions.
  Wider custom limits reach construction and execution; narrower state meters reject nested
  row-zero scans before allocating beyond their allowance.
- Every public lowering constructor validates shape and limits. Slot, state, per-key update,
  Kernel-total update, specialization and substitution failures preserve their exact typed
  payloads. Per-key over-limit programs fail before scan execution through Direct, Stage,
  Region and Kernel paths; total-update rejection is Kernel-specific.
- Projection failures preserve exact local/row/lane/extent payloads and the defined precedence
  through the evaluation boundary.
- Pair-limited grounding returns `Unproved (Max_nodes limit)`; construction-limited grounding
  returns `Unproved (Max_ground_nodes limit)`. Both roots share budgets within an attempt,
  and Structural/constant-bound attempts each receive fresh budgets. All regressions in 1e pass.
- Both fusion admission entry points reject scans. Reference materialization holds one slot
  array at a time. Unspecialized trace locals render in the explorer and textual trace.
- Native public-API integration and both JavaScript checks pass.

### Implementation sequence

1. Fix `Region_eval` declared-shape dispatch and add the `Wk = 1` regression.
2. `.ai/` — representation, construction and error contracts, the freshening obligation, the meter
   API with its charge operation/boundary rule/four reset units/propagation, the static measures
   with their three censuses and reduction-context rule, the validated-execution token,
   specialization and rewrite re-measurement, grounding meter, RHS rendering, `divmod`.
3. `feat(expr): add a bounded ordered scan` — 1b plus `Scan_meter`, the admission check and the
   `Expr.Eval` metering, plus `test/expr/scan_test.ml`. `dune build lib/expr` is the checkpoint.
4. `feat(native): budget Region slots and scan state/updates` — 1c, including `preflight`, the
   validated `lowered` with limits on both branches, **both `lower` and `lower_region` made
   result-valued with their caller migrations**, the `scan_limits` accessor, the `Schedule.ground`
   `Err.Escape` contract change, meter propagation to both Kernel Pixel arms and to
   `Region_execution.value_at`/`Region_eval.value_at` (a fresh meter per invocation, with the
   repeated-call independence regression), `?limits` through
   `Eval_symbolic.run`, and the full `Stage_program.ground` signature migration.
5. `feat(native): add scan-backed Region locals` — 1d.
6. `feat(native): meter grounding, reject scan fusion, render scan locals` — 1e, including
   the public budget, term/meter API, verdict and golden migrations.

Keep `_ai_/` ledger updates in its separate repository. Publish durable contracts in the
tracked `.ai/` design record with the implementation, following `CLAUDE.md`.

---

## Stage 2+ — `lstm.input` itself (outline)

Start after Stage 1 is landed and green. Use [lstm-plan.md](lstm-plan.md) §§2, 4–7 for the
tensor and arithmetic contract, with this document taking precedence for the scan APIs and
integration requirements. Support stacked inference, optional bias pairs, either direction
count and input layout, and finite dropout in `[0,1]` (inactive when `train=false`). Preserve
all three outputs, including live final states. Reject training, packed/unbatched input,
projections, unsupported dtypes, invalid dimensions/configuration and malformed operand lists.

**Use one shared `divmod` helper.** Encode remainder as
`x - K * floor_div_pos(x, K)` from the existing `Add`/`Scale`/`Floor_div_pos` delta nodes, converted
back with `Clamp_low`, never `Assume_position`. `Clamp_low` is sound rather than a papering-over:
for `x >= 0` and `K > 0` the remainder is provably in `[0, K)` so the clamp never fires — state that
argument in the design record. Reuse the one helper for `lane mod K`, `j mod K`, `c mod K`,
`a mod R`, and test with `K > 1`, `R = 2`, `Q > 1` and unequal dimensions so a quotient/remainder
mixup is observable.

- **Binding.** `op "lstm" ~overload:"input"`, using the prerequisite probe. Precedent: `cat`/`stack`
  take `Tensor[]` (`aten_ops_gen.ml:75-76`), `native_layer_norm` returns three tensors (`:87`). Add
  a `Walk_meta`/recipe for `Tensor[]` inputs; `aten_walk_gen.ml`'s default recipe excludes them.
  Oracle fixtures across layers, biases, directions,
  layouts, `dropout = 0 / 0.5 / 1`.
- **Native operation.** `lib/native/ops/lstm.ml`; register in `graph_ir`, the registry,
  `graph_shape` (three shapes), `eval_op` (ordinal-selected), `graph_builder`, `output_transfer`.
  Replace `Region_computation.check_output`'s output-zero restriction with per-output shape
  validation; use one Region program per output ordinal. Keep `Lstm`
  off `Const_ssa.allows`, documented.
- **Stacked layers and configuration coverage** (the corresponding section in `lstm-plan.md`). Required for the
  landing, not a follow-up.
- **Both importers.** `op_bridge_recurrent.ml`, `native_interp_lower_recurrent.ml`. Tensor-list
  precedent: `Interp_decode.tensors_arg_result` (`interp_decode.ml:429-434`) and
  `native_interp_decode.ml:87-96`. `verify.ml:204-221` permits a fixed tuple to expose *fewer*
  outputs — assert the output count explicitly.
- **Native4D counterpart.** Reuse the Native payload if it names no axis and carries no
  `Shape4.t` — the `Sdpa` precedent. Rank-3 operands sit on `H/W/C` with `N=T=D=1`
  (`aten_shape.mli:1-4`), so rank-3 recurrence is representable in Native4D. Native4D has no
  `Discard` (`op.ml:11-18`), so dead state outputs must be gone first.
- **Evidence and ledger.** Native op walk; both corpus shape families; re-verify the three
  update censuses against the real programs; regenerate `make pt2.json-model-support`. Clearing 36 of
  911 nodes does not mean the graph builds — record Sequencer2D's actual next frontier separately for
  Native, Native4D and Kernel. Change the support scoreboard only on regenerated sweep evidence.

---

## Verification

```sh
opam exec -- dune build lib/expr                                 # the 1b checkpoint
opam exec -- dune build
NO_COLOR=1 opam exec -- dune runtest test/expr test/native      # Stage 1's own suites
NO_COLOR=1 opam exec -- dune runtest                             # whole tree
make precommit                                                   # build+format+runtest+file-size+whitespace
make jsoo.runtest && make jsoo.inline-runtest                     # the new Expr node crosses the JS boundary
make pt2.json-model-support                                       # Stage 2 corpus verification
```

Run the non-promoting form first so failures stay visible, and review any golden diff before
`dune promote`.
