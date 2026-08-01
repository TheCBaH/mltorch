# Result/Option census

Stage 0 of `.ai/result_option_migration_plan.md`. This is the per-site
inventory that sizes the migration; every later stage's site list must be a
subset of the `reduce`/`defect` rows here.

The machine-readable inventory is `.ai/result_option_census.tsv` — one row per
site, `file<TAB>line<TAB>cluster<TAB>shape`, **603 rows, each in exactly one
cluster**. This file is the readable summary plus the full listing of every
actionable site.

## Baseline (raw — preserved, not replaced)

Measured at `native-transform-verify` d4d7e1f:

```sh
# 603 — arm heads at line start (the migration's unit of work; = the TSV)
git grep -cE '^[[:space:]]*\|[[:space:]]*(Error|None)\b' -- '*.ml' '*.mli' \
  | awk -F: '{s+=$2} END{print s}'
# 653 — every `| Error`/`| None` occurrence, inline arms included
git grep -oE '\|[[:space:]]*(Error|None)\b' -- '*.ml' '*.mli' | wc -l
# 120 files, of 301 tracked .ml/.mli
```

The two figures differ by regex, which is why both are recorded. Neither is a
quality metric: a type declaration and a reducible match contribute equally.

## Classified baseline

| verdict | n | meaning |
|---|---:|---|
| `defect` | 1 | outright wrong; loses a detection backtrace |
| `reduce:*` | 95 | mechanically reducible to a standard or project helper |
| `audit:*` | 13 | needs a decision, not a mechanical rewrite |
| `exempt-lowering` | 15 | deliberate `Core.Error.t` drop at a framework boundary |
| `emitted-text` | 15 | inside `{\| … \|}`; generated OCaml, not runtime code |
| `keep:*` | 464 | real branching, error leaves, or combinator definitions |
| **total** | **603** | |

**The migration is 96 sites** (1 defect + 95 reduce), not the ~615 the raw grep
suggests and not the ~145 an earlier estimate claimed. 82% of the tree's
`Error`/`None` arms are already correct.

### Method, and its confidence

Clusters were assigned by script (`classify.py` → `refine.py`, kept with the
scratch work, not in-tree) and then **verified by reading the source** for
every `defect`, `reduce`, `audit`, `exempt-lowering` and `emitted-text` row —
the 140 rows that drive work. Two clusters needed sibling analysis rather than
single-line matching:

- `reduce:of_option` requires the sibling arm to be exactly
  `Some x -> Core.return x` *and* the match to have exactly two arms. 21
  candidates were rejected on that test (`keep:of_option-no`) — almost all are
  3+-arm decoders in `lib/interp/interp_decode.ml` where a middle arm handles a
  wrong-kind argument.
- `print-arm` splits on whether **every** arm of the match is pure rendering.
  Only 24 of 160 are (`reduce:render`); the other 136 have an `Ok` arm that
  continues into nested work, and stay explicit per the migration plan's own
  rule.

The 464 `keep:*` rows were classified by shape and sampled, not each read in
full. Where a shape-based rule was wrong, reading caught it — see the false
positives recorded below.

## Actionable sites

### `defect` (1) — Stage 1 — **fixed**

| site | shape |
|---|---|
| `lib/native/transform/graph_view.ml:352` | `Error e -> Core.fail (\`Invalid_sig (id, e.Core.Error.kind))` |

`D.validate_sig` returns a `Core.result`; rebuilding it by hand makes
`Error.make` capture a *fresh* callstack at line 352, discarding the original
detection site. `Core.map_error (fun e -> \`Invalid_sig (id, e))` fixes it.
This is verbatim the wrong form documented in `.ai/fmt_migration_plan.md`
136-140 — the bug that doc warns about had a live instance.

Introduced by `6ce482f` (`git blame -L 352,352`), so the fix is a `fixup!`.

> **Found while implementing:** the defect was confirmed by observation, not
> inferred. Covered by `test/native4d/shape4_test.ml`, "view4: Invalid_sig keeps
> the detection site of the dialect check"; reverting the fix alone turns it red
> (`detected_in_dialect=false view_is_only_the_caller=false`).
>
> The first draft of that test asserted the wrong frames — that the trace would
> name `shape4.ml` and *not* `graph_view.ml`. Both were wrong:
> `Shape4.of_vec6` is **inlined** into `Dialect4.validate_sig`, so the detection
> frame reads `dialect4.ml:32`; and `graph_view.ml` legitimately appears below
> it as the caller, before and after the fix. The property that actually
> discriminates is frame **order** — frames print innermost-first, so "detected
> inside the dialect" is "dialect4 appears above graph_view".
>
> This is direct evidence for the migration plan's warning about `Core.of_option`
> and frame indices: which frame survives inlining is not predictable in advance,
> so Stage 2 must observe the compiled behaviour before documenting it.

### `reduce:of_option` (25) — Stages 2–3

> **Found while implementing (Stage 2):** `Core.of_option` exists, and the frame
> question the plan flagged as unpredictable is now measured rather than guessed.
> Observed on 4.14.3:
>
> ```text
> Raised by primitive operation at Core.Error.make   lib/core/core.ml:11
> Called from Core.fail                              lib/core/core.ml:43 (inlined)
> Called from Core.of_option                         lib/core/core.ml:45
> Called from …lookup_missing                        <caller>          (inlined)
> Called from …                                      <caller's caller>
> ```
>
> `fail` inlines **into** `of_option`, while `of_option` keeps its own frame —
> the opposite of the review's prediction that a tail-position `fail` would let
> `of_option` be elided. Net effect: a bridged option sits one named frame deeper
> than a direct `Core.fail`, and the caller stays visible either way.
>
> So the eager-payload caveat stands, but the provenance caveat is resolved: no
> Stage 3 site needs to be skipped to protect frame-exactness. `core.ml`'s
> comment now states the rule without indices.

> **Found while implementing (Stage 3):** all 21 `lib/` sites converted; none
> was skipped, because every payload is a cheap constructor over values already
> in scope — no `Fmt.str`/`asprintf` among them.
>
> The eager-payload cost was **measured, not assumed**. `of_option` allocates
> the payload before inspecting the option, so `eval_direct.find_tensor` and the
> `native_interp` output collection now allocate one small block per lookup on
> the success path. resnet18 inference, 50 samples per variant, reverting only
> the three hot-path files for the baseline:
>
> | variant | mean | min |
> |---|---:|---:|
> | with `of_option` | 269.7 ms | 254.2 ms |
> | hand-rolled match | 264.3 ms | 257.5 ms |
> | with `of_option`, repeated | 263.7 ms | 255.2 ms |
>
> The repeat is the point. A single paired run showed a ~4% regression, which
> the control run shows is machine noise: the two `of_option` runs differ by
> more than either differs from the baseline, and `of_option` holds the lowest
> minimum. **No measurable cost.** These lookups are per-node, not per-element,
> and the workload is dominated by conv2d — so this is the expected result, but
> it is now observed rather than argued.

`Some x -> Core.return x | None -> Core.fail e`, two arms exactly.

| area | n |
|---|---:|
| lib/native | 9 |
| lib/native4d | 4 |
| test/native | 3 |
| lib/pt2, lib/native_interp, lib/native_graph, lib/interp | 2 each |
| test/interp_run.ml | 1 |

21 in `lib/`, 4 in `test/` — the plan scoped Stage 3 to `lib/` only; the four
test sites ride along or are left, at implementation's discretion.

### `reduce:or_raise` (34) — Stage 6 — **done**

| area | n |
|---|---:|
| test/native | 21 |
| test/native4d | 10 |
| test/native_bridge_test.ml | 2 |
| lib/native/ops/conv.ml:15 | 1 |

The single `lib/` site is not a test fixture and needs its own judgement.

> **Found while implementing:** two things worth carrying forward.
>
> **The census misses `Result.fold`.** `test/native/compute_test.ml:681,738`
> have the same "unwrap or die" semantics, written as
> `|> Result.fold ~ok:Fun.id ~error:(fun e -> failwith …)`. That is not an arm
> head, so a census keyed on `^\s*\| (Error|None)` cannot see it — the second
> blind spot of that kind, after `tensor_bridge.ml:34`'s pattern destructure.
> Both were found only by grepping the *semantics* (`Core.Error.kind` near a
> raise) once the arm-head work was done. Any future audit should run both
> passes.
>
> **A correct replacement count does not prove a scripted edit was correct.**
> Seven sites shared an identical trailing block, so they were rewritten by
> regex with an assertion on the number of replacements. The assertion passed —
> and one replacement in `permute_passes_test.ml` was still wrong: the
> non-greedy body ran from one `match` to a *later* block's `with`, deleting a
> `match` belonging to an unrelated expression 350 lines away and orphaning its
> `with`. The count was right because the number of matches was right; the
> matches themselves were not. The compiler caught it; reverted and redone with
> `Edit`. CLAUDE.md's rule (prefer `Edit`, which fails loudly on a miss) covers
> the miss; this is the adjacent failure — a hit in the wrong place.
>
> `or_invalid_arg` in `conv.ml` changes `Invalid_argument` to `Failure`, since
> `Core.or_raise` raises the latter. It is `lib/`, not a test, but nothing
> catches it: full suite green.

### `exempt-lowering` (15) — Stages 4–5

Deliberate drops of the `Core.Error.t` wrapper where the output leaves the
framework. Each becomes a **named, commented helper** so a future checker can
exempt it by symbol rather than guess.

| site(s) | target type | stage |
|---|---|---|
| `bin/native_graph.ml` :219 :229 :242 :276 :545 :565 :570 :582 :662 :668 | Cmdliner `(_, string) result` | 4 → `to_cli` |
| `lib/native/transform/pattern.ml:164` (`region`), `:199` (`run`) | the pattern monad's `failure` | 5 → `of_core` (**done**) |
| `lib/native_aten_bridge/tensor_bridge.ml:34` | `(_, string) result` | 5 |
| `test/native/dce_test.ml:150` `:153` | one printer over two error rows | keep; comment at :146-147 already explains |

`tensor_bridge.ml:34` is worth singling out: it drops the wrapper by
**destructuring in the pattern** — `| Error { Core.Error.kind = e; _ } ->` — so
it never writes the string `e.Core.Error.kind` and no grep for that expression
finds it. Any future checker must match the destructuring form too.

The `bin/native_graph.ml` line list supersedes an earlier draft that named
`:504` and `:672`; `:504` is `Fmt.pf ppf "provenance error: %a"`, a printer.

### `reduce:render` (24) — Stage 7 — **done, mostly as non-findings**

> **Found while implementing: this cluster is ~75% false positives.** Checking
> all 24 against the source, only 6 were value printers worth converting. The
> heuristic ("every arm of the match calls a print function") is too weak, and
> the reasons it fails are worth naming:
>
> - **`lib/native/transform/pass.ml:28` is not a result at all.** Its `problem`
>   type is `Error of Map_verify.error | Rejected of …` — `Error` is a *local
>   variant constructor*. A census keyed on the token `| Error` cannot tell that
>   from `Stdlib.Error`, and no amount of arm-shape analysis would. Converting
>   it would have been nonsense.
> - **Arms that print *and* return a value** are control flow, not rendering:
>   `dce_test.ml:24` prints then yields `None`/`Some g`.
> - **The `Ok (Rewrite.Step (final, map)) -> <several prints>` family** (~14
>   sites across pack_test, drop_pool_indices_test, fold_const_test,
>   permute_passes_test, pass_test, mutation_test, verify_test, pass4_test)
>   destructures a rich value and emits multiple lines. Routing that through
>   `~ok:` puts the destructuring inside a lambda and reads worse than the match
>   it replaces. These belong with the 136 `keep:render` sites, not against them.
> - **`test/pt2_test.ml:50,159`** bind a value in the `Ok` arm and interleave
>   `Printf` (stdout) with `Format` (std_formatter); converting risks reordering
>   output between two buffers for no gain.
>
> **Converted (6):** `test/native/tensor_test.ml:56` → `Option.fold` (it
> returns a string, so it was never a printer either);
> `test/native_bridge_test.ml` ×2 → `Core.Pretty.option_or`;
> `test/native/core_test.ml` ×2 and `test/pt2_test.ml:101` →
> `Core.Pretty.core_result`.
>
> **Reverted:** `lib/aten_schema/aten_func_ast.ml:138`. `Fmt.option` would fit,
> but `aten_schema` depends only on jsont/yamlt — adding `fmt` to a minimal
> library for a cosmetic printer change is not a trade worth making.
>
> Goldens did not move, which was the acceptance criterion.

### `reduce:option-map` (10) — Stage 8 — **done, 7 of 10**

> **Found while implementing:** three are multi-arm matches the `| None -> None`
> line cannot distinguish from a two-arm map — `aten_decode_gen.ml:185` (a
> guarded `Some (Tensors_ret nret) when …` arm), `aten_spec_gen.ml:247` (a
> `Some T_skip` arm), and `native_interp.ml:138`, which the do-not-touch
> register had already caught. All three stay. This is the same false-positive
> shape as `reduce:render`, and it is now the third cluster where a
> single-line rule over-reports.
>
> The other seven converted to `Option.map`/`Option.bind`.
> `map_verify.ml:531`'s two explanatory comments were preserved in place; its
> inner `under_test` match also folded to `Option.fold ~none:false`.
>
> **The census under-reported here too.** `lib/aten_spec_run` :289 :305 :396
> :503 :552 were classified `keep:render`, but reading them shows the printed
> prefix repeats verbatim across both arms —
> `"  aten   %s = %a"` against `"  aten   %s = <missing>"` — which is exactly
> `Core.Pretty.option_or ~none:"<missing>"`, and the file already uses that
> combinator at `:341`. All five converted. Where Stage 7 meant overruling the
> classifier's *findings*, this meant overruling its *dismissals*; both need the
> source read.

`| None -> None` with a single mapping sibling → `Option.map`/`Option.bind`:
`lib/aten_gen/aten_config_gen.ml` :71 :100 :225 · `aten_decode_gen.ml:185` ·
`aten_spec_gen.ml:247` · `lib/native/transform/ground_eval.ml:140` ·
`map_verify.ml:373` `:531` · `passes/fold_const.ml:60` ·
`lib/native_interp/native_interp.ml:628`.

`map_verify.ml:531` carries an explanatory comment that must survive the
rewrite.

### `reduce:transparent` (2) — Stage 5 — **done**

> **Found while implementing:** `region` and `run` were not two independent
> crossings — both compute the same `Rgn.of_nodes s.view (claimed ∪ shared)`.
> So one named crossing (`of_core`) plus one shared `region_of` collapses
> `:164`, `:195` and `:199` together, and `Core.Error.kind` now appears exactly
> once in the file. The transparent arm at `:195` fell out as `Result.bind`
> without needing to be addressed separately.
>
> The `Error e -> Error e` arms remaining at `pattern.ml:63,66` are the monad's
> own `let*`/`let+` definitions — the same `keep:combinator-def` category as
> `lib/core/core.ml:82`, and equally out of scope.

`lib/native/transform/cluster_relation.ml:345` is success *mapping*
(`Ok () -> Ok rel`), so `Result.map (fun () -> rel)`.
`lib/native/transform/pattern.ml:195` is `Result.bind` — **not** the module's
own `let*` at `:63`, which is a state-monad operator over
`state -> ('a * state, failure) result` and does not typecheck there.

### `audit:*` (13) — Stage 8

Decisions, not rewrites.

- `audit:error-to-none` (5) — **resolved**.
  `lib/native/transform/passes/fold_const.ml:42,52` already carried a comment
  justifying the discarded diagnostic and were the model.
  `test/native/permute_passes_test.ml:26,29,52` now carry one too: the plan
  offered `Core.or_raise` as the alternative, but these helpers return an
  `option` that every caller pattern-matches, so raising would force a
  restructure of each call site for a diagnostic none of them read. A failure
  there still fails the test — it flips a boolean equivalence check against a
  golden expecting `true` — so the loss is message quality, not detection. The
  comment says so.
- `audit:option-fold` — **converted** (all but one). `| None -> false` with a
  computing `Some` arm is `Option.fold ~none:false ~some:f`, **not** `is_some`
  and **not** `Option.value` (the `Some` arm applies a function).

  Stage 8 dismissed this cluster as "reads fine as a match". That was wrong,
  and incoherent besides: the same stage *introduced*
  `Option.fold ~none:false ~some:(Cluster_var.equal v)` at `map_verify.ml:538`
  while leaving seven identical siblings as matches. Converted on review:
  `ground_eval.ml` ×3 (`stored_f32` is the clearest — its body already used
  `Option.bind`, so the trailing `match` was pure wrapping), `region.ml:90`,
  `json_util.ml:81`, `float32.ml:23`, plus two more inline instances the
  arm-head census never saw (`pointwise.ml:113`, `payload.ml:114`).

  Kept: `aten_gen/aten_spec_gen.ml:401`, whose `Some` arm calls `emit_module`
  for its side effect and then returns `true`. Folding a side-effecting
  predicate inside `List.filter` hides the effect; the match shows it.
  Kept too: `permute_passes_test.ml:1111,1178`, nested inside a *tuple* match
  where the fold clarifies nothing.

Note the repo has **zero** `Some _ -> true | None -> false` predicates — it
already uses `Option.is_some`.

## Registers

### Do-not-touch

Sites that look mechanical and are not. Each was a false positive of a
shape-based rule, caught by reading:

- `lib/native_interp/native_interp.ml:138` — `| None -> None` in a **five-arm**
  match where `Some { arg = Argument.None _; _ }` also maps to `None` and a
  further arm raises. Not `Option.map`. (`keep:multi-arm`.)
- `lib/native/transform/pattern.ml:75,85` — `| Error (Invalid _ as e) -> Error e`
  is deliberate two-tier backtracking: `Invalid` propagates, `Mismatch`
  backtracks. Only `:195` is transparent.
- `lib/core/core.ml:82` — inside `Core.List.map`, i.e. a combinator definition.
  The `Core` module defines the very helpers this migration recommends
  (`map_error` at :46-48 is `Error e -> Error { e with … }`, `or_raise` at
  :50-52 is a `Core.result` reaching `failwith`), so it is excluded wholesale.
  (`keep:combinator-def`.)
- `lib/aten_gen/aten_emit.ml` :107 :120 :189 :206 :222 — all five arm heads sit
  inside the `{|%s … |}` region spanning lines 84-232 and are *emitted* OCaml
  text. The `%%d` escapes at :107, :120 and :206 prove it. 15 arm heads across
  the tree are emitted text (`aten_emit.ml` 5, `aten_spec_gen.ml` 4,
  `aten_config_gen.ml` 1, `test/codegen_test.ml` 5).
- The 136 `keep:render` sites — the `Ok` arm continues into nested work, so the
  match is operational control flow that also reports, not a value printer.
  `.ai/fmt_migration_plan.md` Stage 3 established this distinction.

### Wrapper drops

Every hand-rolled unwrap of the `Core.Error.t` wrapper in the tree: 1 defect
(above), 15 deliberate lowerings (above). There are no others — this is the
complete set, and it is the design input for the deferred checker.

## Outcome

All stages landed (`3d32481..03115e6`). **Arm heads 603 → 515.** Adoption:
`Core.or_raise` 22 → 55, `Core.of_option` 0 → 23, `Core.map_error` 52 → 55,
`Core.Pretty.*` 30 → 44. One real bug fixed (`graph_view.ml:352`). Goldens
never moved.

Final semantic sweep — the second pass this document argues for — is clean:
zero raises over `Core.Error.kind`, and the one hand-rolled wrapper drop left
(`dce_test.ml:147`) is the registered, commented one.
`test/func_parser_test.ml:3`'s `Result.fold ~error:failwith` is over a plain
`(_, string) result`, so `Core.or_raise` correctly does not apply.

### The method, graded

The census was wrong in both directions, and the corrections are the reusable
part:

| | |
|---|---|
| over-reported | `reduce:render` ~75% false positives; `reduce:option-map` 30% |
| under-reported | wrapper drops written as pattern destructures; `Result.fold`; **inline arms** |

**The fourth blind spot, and the least excusable: `Option.iter` was never
checked for at all.** The high-level plan's classification table lists
"apply a side effect only when present → `Option.iter`". The classifier has no
rule for `| None -> ()`. Eleven sites, every one filed `keep:*` or
`emitted-text`; eight were genuine:

`lib/native/graph_ir.ml` ×3 (the `pp_*_annotation` family) ·
`bin/native_graph.ml:467` · `lib/aten_schema/aten_func_ast.ml:115` ·
`lib/aten_native_verify/verify.ml:83` · `lib/aten_gen/aten_walk_gen.ml:120` ·
`test/interp_run.ml:47`.

Converting `graph_ir`'s annotation printers also removed a superfluous
unit-lambda: `Printer.t`'s fields already have a printer's shape, so
`Fmt.pf fmt " {%a}" p.node id` replaces
`Fmt.pf fmt " {%a}" (fun fmt () -> p.node fmt id) ()`. The match had been
hiding that.

Kept: `interp_verify.ml:20` (three arms — `None`, `Some (Error _)`,
`Some (Ok _)`), `native_interp.ml:150` (three arms with a guard), and
`aten_config_gen.ml:201` (emitted text).

Unlike the other three, this is not a limit of syntactic analysis. The rule was
written down in the plan and simply not implemented — which is its own lesson
about auditing with a tool you also wrote.

**The third blind spot, found after this document first claimed completeness.**
An arm head has to start a line to be counted, so the whole
`function None -> x | Some y -> f y` / `match o with None -> x | Some y -> f y`
family — written on one line, or with the `None` arm trailing `function` — was
invisible. Nine sites; eight are genuine `Option.fold`/`map`/`bind`:

| site | form |
|---|---|
| `lib/native/json_util.ml:55` | `Option.map` |
| `lib/aten_gen/aten_walk_gen.ml:42` | `Option.bind` |
| `lib/native/graph_ir.ml:612`, `tensor_sig.ml:41` | `Option.fold ~none:[]` |
| `lib/native_graph/pt2_native_graph.ml:231` | `Option.fold ~none:[]` |
| `lib/native4d/graph.ml:45` | `Option.fold ~none:"?"` |
| `bin/native_graph.ml:204` | `Option.fold ~none:""` |
| `lib/native/transform/pattern.ml:181` | `Option.fold ~none:(-1)` |

Note the shape is `Option.fold`, **not** `Option.value ~default`: the `Some`
arm *applies a function*, so the default and the mapped value have different
types before folding. `Option.value` fits only when the arm is `Some x -> x`.

Not converted, and instructive: `lib/aten_gen/aten_config_gen.ml:56` matches
`Aten_func_ast.Default.t`, whose **first constructor is literally named
`None`** — a multi-arm variant match, not an option. It is the exact option
analogue of `pass.ml:28`'s local `Error`, and the second proof in this audit
that constructor *names* cannot be trusted without types.

Four more (`lib/aten_gen/aten_emit.ml` :113 :126 :131 :138) are inside the
`{|%s … |}` template spanning 84–232 — **emitted OCaml text**, so converting
them would rewrite generated bindings rather than runtime code. They stay, per
the do-not-touch register. The `Map.update` callbacks
(`graph_ir.ml:269`, `coeff_form.ml:35`, `map_verify.ml:405,447`) return an
option by contract and are already idiomatic; folding them reads worse.

The single sharpest failure: `pass.ml:28` matches on a **local variant
constructor named `Error`**. No token-level or arm-shape rule can distinguish
that from `Stdlib.Error` — only types can, which is the strongest argument for
the deferred typed checker.

The single sharpest lesson about tooling: a scripted rewrite with an assertion
on the replacement *count* still landed one edit in the wrong place. A right
count is not a right match.

## What this changes in the plan

- The migration is **96 sites**, and 82% of the tree is already correct. The
  high-level plan's "615 arms in 123 files" overstates the work ~6x.
- Stage 7 is the *minority* case it predicted: 24 reducible renders against 136
  that stay.
- Stage 3 gains 4 `test/` sites the plan scoped to `lib/`; Stage 6 gains 1
  `lib/` site the plan scoped to tests.
- Stage 5 gains `tensor_bridge.ml:34`, which no grep for `e.Core.Error.kind`
  could find.
