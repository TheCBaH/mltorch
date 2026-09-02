# Region computation for SDPA — analysis and staged plan

Working note.  Grounded against `unbind-int` at `e37ff39` (Region computation
landed for RMSNorm/LayerNorm/Softmax).  Companion to
`region-compute-design.md` §6 and to `.ai/region_compute_design.md`'s
"Design headroom for deferred work"; this note is the SDPA-specific
derivation those two defer.

Related working note: `native4d-sdpa-compatibility-plan.md`, which depends on
Stage A of this plan.

## 1. Where SDPA's cost actually is

`lib/native/ops/attention.ml:450-466` evaluates `score_at` three times per
output coordinate — once inside `m`, once inside `z`, once inside `numer` —
and each `score_at` is an `E`-term dot (`attention.ml:426-448`).  Under
`Direct`, `Compute.pixel` is applied once per output coordinate by
`Schedule.evaluate` (`lib/native/eval_direct.ml:250-255`,
`lib/native/schedule.ml:22`), so nothing is shared across the `E` outputs of
one query row — not even `sf`, which is re-derived at every coordinate
(`attention.ml:419`).

Total work is

```text
3 * N*T*D*H*Wq*Wk*E*E
```

which is exactly the eight-factor ceiling `output_shape` already enforces
through `Attention.Sdpa.total_work_bounded`.  The `Theta(Q*K*E^2)` figure in
`region-compute-design.md` §6 is this, with the batch-like factors dropped.

The structure that fixes it is the one Softmax already exploits: `m` and `z`
are constant across the output feature axis `C`.  That is a Region partition,
literally — `Whole [Axis.C]`.

## 2. Stage A — scalar locals only, no language change, bitwise identical

### 2.1 The program

```text
region [C = Whole]
  let sf : scalar = sqrt(scale)
  let m  : scalar = max_reduce_{k<Wk} score_at(k)
  let z  : scalar = sum_{k<Wk} exp(score_at(k) - m)
  emit select (-inf < m)
         (sum_{k<Wk} (exp(score_at(k) - m) / z) * value[.., W:=k, C])
         0
```

`score_at(k)` is `attention.ml:426-448` unchanged: the `E`-term dot of
`query[.., C:=e] * sf` and `key[.., W:=k, C:=e] * sf`, plus the mask read at
the score coordinate.

### 2.2 Why the locals are legal

`Region_program.check`'s `Non_invariant_local`
(`lib/native/region_program.ml:186-200`) forbids a local from depending on an
output axis that is `Whole` in the partition.  The whole axis here is `C`, and
`score_at` overrides `C` with the reduction variable in every load it
performs:

- query at `Coord.set out C e`;
- key at `Coord.set (Coord.set out C e) W k`;
- mask at `Coord.set out C k`, through the broadcast reduction.

So `Expr.Fold.output_axes` on `m`'s and `z`'s bodies yields `{N,T,D,H,W}` and
never `C`.  The check passes.  The emitter reads `C` freely — that is what
emitters are for — through `value[.., W:=k, C]`.

`sf` depends on no output axis at all, so it is trivially invariant.  It is
loop-invariant across the whole tensor rather than merely across the row, but
the program has no notion of a tensor-level constant; a per-key local is the
available hoist and is enough to take it off the inner reduction path.

### 2.3 Cost

Per key (one key = one query row, `N*T*D*H*Wq` of them):

| part | cost |
| --- | ---: |
| `m` | `Wk*E` |
| `z` | `Wk*E` |
| `E` emitters, each `Wk*E` | `Wk*E*E` |

Total `N*T*D*H*Wq*Wk*E*(E+2)` against the current `3*N*T*D*H*Wq*Wk*E*E`:
**~3x on the dominant term**, plus `sf` moving from per-output to per-row.

### 2.4 The claim is `Identical`, and that is provable with existing machinery

`Region_program.specialize_pixel` substitutes the locals back and yields *the
current expression, unchanged* — no reduction changes order, grouping or
bounds; the locals only name subterms that were already there.  Therefore
`Region_program.reconstructs ~pixel:(Legacy_pixel ...) program` is `true`,
results are bitwise equal, and none of the following moves:

- `Verify.atol_for_target`'s single `1e-5` entry for this target;
- the flash-oracle argument in `.ai/attention_design.md` §7 and §10;
- `Native_verify`'s bitwise `Float.equal` Direct-vs-Symbolic comparison.

This is a free win against an already-satisfied proof obligation.

### 2.5 Secondary payoff

`Stage_program.Stage.computation` gets the structural program instead of the
flattened tree `.ai/attention_design.md` §6 measured at depth 12 / size 121,
so Model Explorer shows attention's row sharing.  Transform verification and
Kernel adaptation consume the same structure.

### 2.6 File-level inventory for Stage A

1. `lib/native/ops/attention.ml`
   - rename `Sdpa.Compute` to `Sdpa.Legacy_pixel` (test oracle only), exactly
     as `Reduce.Softmax` did at `lib/native/ops/reduce.ml:639-655`;
   - add `Sdpa.Computation.program ~limits params ~query ~key ~value ~mask`
     authoring the program above.
2. `lib/native/region_computation.ml`
   - `is_region_authored` gains `Sdpa`;
   - `program` gains an `Sdpa` arm resolving query/key/value by role and
     filling an absent mask.
   - `synthetic_role` gains `Sdpa_mask`.  This is the existing mechanism, not
     a new one: `eval_op.ml:224-233` already open-codes "absent mask becomes a
     `0.`-filled all-ones-shaped tensor", and `Rms_weight`/`Layer_weight`/
     `Layer_bias` are the same pattern.  Moving it into `fill` makes the
     default value and shape decided in one place.
3. `lib/native/eval_direct.ml:93-95` — add `Sdpa_mask` to the synthetic-id
   role list.
4. `lib/native/eval_op.ml:218-236` — the `Sdpa` arm becomes an explicit
   impossible arm (`invalid_arg "Eval_op.pixel: Sdpa is Region-authored"`),
   matching the three existing ones at `eval_op.ml:157`, `:214`, `:250`.
5. `lib/native/region_context.ml` — needs an `Expr.Coord`-level
   `broadcast_coord`.  `Pointwise_binary.broadcast_coord`
   (`pointwise_binary.ml:36-40`) is polymorphic in the index but operates on
   `Vec6.t`, and the Region program's coordinates are
   `Role.Position.t Index.t Coord.t`.  Share one definition or the two
   spellings will drift; a broadcast bug here is silent (an out-of-bounds read
   or a dropped mask), which is the failure `Max_op` exists to prevent
   elsewhere.
6. `lib/native4d/region_computation4.ml` — no change needed for Stage A
   itself; see the companion note for what an `Sdpa4` would add.

### 2.7 Verification for Stage A

- `reconstructs` against `Sdpa.Legacy_pixel` over the `Sdpa.Walk` config
  space, the same differential test the three landed ops use.
- `Region_trace` coverage: keys, local evaluations, emitter visits, ownership
  agreement.  A dense result comparison alone is not sufficient evidence —
  `.ai/region_compute_design.md` says why.
- `lib/native_op_walk/sdpa_nwalk.ml`'s Direct-vs-Symbolic walk, unchanged.
- `Recipe_sdpa`'s ATen oracle run, unchanged and expected bit-identical.
- Counter table for `bin/region_compute_bench.exe`, adding SDPA rows.

### 2.8 What Stage A does NOT fix

Each of the `E` emitters still recomputes all `Wk` scores.  The asymptotic
`E^2` is untouched; only the constant `3` becomes `1 + 2/E`.

## 3. Stage B — one shaped-local constructor, flash asymptotics, still `Identical`

`Region_local.Shape.t` is a one-constructor variant *on purpose*
(`lib/native/region_local.ml:2-4`); `.ai/region_compute_design.md` records
that leaving the field out would make shaped locals a language change instead
of a new constructor.  SDPA is the case it was left in for.

### 3.1 The program

```text
region [C = Whole]
  let s : vector[Wk] = score_at(i)                        -- Wk*E
  let m : scalar     = max_reduce_{k<Wk} s[k]             -- Wk
  let z : scalar     = sum_{k<Wk} exp(s[k] - m)           -- Wk
  let p : vector[Wk] = exp(s[i] - m) / z                  -- Wk
  emit select (-inf < m) (sum_{k<Wk} p[k] * value[.., W:=k, C]) 0   -- Wk
```

Per key `Wk*(2E+3)` against the current `3*Wk*E*E`: **speedup ~1.5*E**, i.e.
~96x at `head_dim = 64`, ~7.5x at the walk's `e = 5`.

Caching only `s` and not `p` also reaches `Theta(Wk*E)` — `Wk*(4E+2)` — with
one fewer vector local, if footprint matters more than the constant.  Both are
asymptotically right; `p` is the better constant.

### 3.2 The numerical point, which the existing design record understates

`region-compute-design.md` §6 lists three strategies (score cache, output
accumulator, blocked online softmax) and then discusses the online-softmax
numerical boundary as though it applied to all three.  It does not.

- Strategies 1 and 2 **only cache values**.  No reduction changes order,
  grouping or bounds.  Substituting the vector local back through
  `specialize_pixel` reproduces the current expression exactly, so the claim
  stays `Identical` and results stay bitwise equal.
- Strategy 3 (blocked online softmax) is the only one that reassociates, and
  its payoff is avoiding the `Wk`-element row in memory, not FLOPs.

So the asymptotic win is reachable **without** adopting a tolerance policy, a
second oracle, or an `Equivalent` claim.  Online softmax stays a genuinely
separate, later, evidence-bearing step, motivated by footprint rather than by
work.  Fold this correction into the tracked record when Stage B is planned.

### 3.3 What Stage B costs to build

In dependency order:

1. **`Region_local.Shape.t` gains `Vector of int`**, plus
   `Region_local.vector ~id ~var ~extent ~value` whose body may mention the
   binder.
2. **`Expr.Value.t` gains one node**:
   `Local_at of Local_var.t * Role.Position.t Index.t`, alongside the existing
   `Local` (`lib/expr/expr_api.ml:244`).  Reads reuse `Reduce_var` scoping.
3. **`Expr.Builder` must be able to mint a binder outside a reduction.**
   Today only `Builder.reduction` mints a `Reduce_var.t`, and that is
   deliberate — `expr_api.ml:30-34` says the absent public constructor is what
   makes reducer scope enforceable.  This is the one place the extension
   touches `lib/expr`'s trust boundary and it deserves the same care: a
   `Builder`-only minter, never a public constructor.
4. **`Region_program.check`** gains shape agreement — `Local` on a vector
   local, or `Local_at` on a scalar local, is a typed error — and keeps
   `Non_invariant_local` unchanged, since the rule is identical for a vector
   body.  New rows follow CLAUDE.md's payload rule: a named record, not prose.
5. **`Region_execution`**'s per-key slot array becomes scalar-or-vector, and
   `evaluate_locals` loops the extent binding the local's own var;
   `Expr.Eval`'s `local` lookup gains an indexed form.
6. **`Expr.Rewrite.substitute_locals` gains an indexed variant** that
   substitutes the binder with the read index.  This is what keeps
   `specialize_pixel`, `reconstructs`, transform verification and Kernel
   adaptation working, and it is what makes §3.2's `Identical` claim provable
   rather than asserted.  Without it, Stage B is unverifiable by the existing
   differential test and the whole argument collapses.
7. **A locals-footprint bound.**  `Kernel.Limits.max_extent` bounds one
   vector; total local words across several vector locals is an *aggregate*
   and needs its own divide-before-multiply fold, per CLAUDE.md's 32-bit rule
   — `lib/native` is js_of_ocaml-reachable and a check on a wrapped product is
   not a bound.
8. **`Region_trace` / `Region_execution.counters`** currently increment
   `locals` once per local regardless of shape.  Left alone, a vector local
   silently under-reports and the counter table in
   `.ai/region_compute_design.md` stops being evidence of what it claims.
   Either count elements or add a separate `local_elements` counter, and say
   which in the record.

### 3.4 Interaction with the resource ceiling

After Stage B the eight-factor `total_work_bounded` is conservative by
~`E/2`.  **Do not loosen it in the same change**: loosening admits graphs that
are currently rejected, which is a scope change with its own verification
obligation, and the new bound should be derived from the region program's cost
model rather than hand-edited.

## 4. Recommendation

Land Stage A alone first.

- ~3x, no new language, program shape reused verbatim from
  `Reduce.Softmax.Computation`.
- Its `Identical` proof is the existing `reconstructs`-against-`Legacy_pixel`
  test — the same migration the three landed ops already went through.
- It exercises the Region path on an op with a **broadcast operand** and a
  **two-level reduction**, neither of which the current three cover.  That is
  real coverage of the IR, not just a speedup.
- It is the prerequisite for `native4d-sdpa-compatibility-plan.md`, which
  needs a Region-authored SDPA to avoid a second numeric kernel.

Stage B then lands as a language extension whose payoff is measured against a
Stage A baseline rather than against the pixel path.

## 5. Open questions

1. Should `sf` be a local at all, or should the IR gain a program-level
   (per-tensor, not per-key) constant?  A per-key local is correct but
   over-general; a tensor-level constant is a fourth binding class and
   probably not worth it for one `sqrt`.
2. Does `Whole [C]` remain the right partition once N/T are used as real
   batch-like axes?  The partition is over output coordinates, so `Whole [C]`
   is right regardless — but confirm the key count `N*T*D*H*Wq` is what the
   benchmark should sweep.
3. Stage B's vector extent is `Wk`, a runtime extent.  `Region_local.Shape`'s
   `Vector of int` takes a static `int`; confirm the extent is known at
   program-construction time in every route (it is, from `key_shape`, but
   `Region_computation` is the boundary that must prove it).
