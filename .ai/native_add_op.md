# Adding a native operation (native + graph + ATen bridge)

This is the step-by-step for wiring a new op into the pure-OCaml native engine,
the graph IR, and the ATen bridge. It is the generalisation of the changes that
added `aten.mul` (the elementwise twin of `add`). For the design rationale of
each layer see `native_compute_design.md`, `native_graph_design.md`, and
`native_aten_bridge_design.md` / `native_aten_bridge_layout.md`.

Worked example below: a pointwise binary op `Mul`. Structured ops
(conv/pool/linear/norm) follow the same site list; only the per-op `Compute`
functor and `output_shape` differ.

## Ordering convention (applies to every site here)

Variant constructors **and** the per-op functions/match arms are kept in **global
alphabetical order** — so any op is easy to locate by name — unless there is a
concrete reason to deviate (e.g. a runner that references another, or a default
`| _ ->` fallthrough that must stay last). When you add an op, slot it into its
alphabetical position at every site below; don't append.

## The sites to touch

1. **Native op** — `lib/native/ops/<group>.ml`
   The op module is the single source of truth for everything about the op except
   the shape/eval dispatch. Expose:
   - a payload record `type t` — its `params` (if any) plus its operand refs as
     `Tensor_ref.t` / `Tensor_ref.t option` fields;
   - `name` (the JSON case tag, e.g. `"Mul"`), `jsont : t Jsont.t` (a payload codec
     reusing `params_jsont` and `Tensor_ref.jsont`), `operands`, `map_operands`,
     and `pp pp_ref fmt t` (operand refs printed via the passed `pp_ref`, params via
     a local `pp_params`);
   - `output_shape` and a `Compute (S : Semantics.SEMANTICS)` functor with a `pixel`
     function written against the abstract `S` value/index domain (so it runs
     unchanged under `Direct` and `Symbolic`).
   Elementwise ops live in `pointwise.ml`; reductions in `reduce.ml`; etc.
   `pointwise.ml` has two shared payload helpers: `Bin` (`{ a; b }`, for the
   binary ops) and `Scalar_bin` (`{ x; scalar }`, for the scalar-parameter twins
   `Add_scalar`/`Div_scalar`). Reuse whichever fits rather than restating the
   codec/dataflow/pp boilerplate. Reuse the value basis in
   `semantics.ml` — `mul`/`add`/`sub`/`div`/`select`/`lt`/… already exist, so a new
   pointwise op usually needs no new primitive.
   **A structural op usually needs none either, and the two kinds of addition to
   `SEMANTICS` are not the same size.** `index_max` (added for `Pad`'s reflect
   mirror) cost three lines, because `Expr` already had the `Max` constructor
   with every eval/pp/compare/traversal arm. An index *comparison* would not:
   `Expr.Bool` has only `Value_lt` and `Index_eq`, so `index_lt` would be a
   genuine new AST node. Before reaching for a boolean primitive, check whether
   the clamp-and-compare idiom (`.ai/native_compute_design.md` §1) expresses the
   test — it is also what keeps a `load` in bounds, since `select` evaluates
   both arms.
   **Decide where a parameter becomes canonical, and say so in the payload's
   type.** `Slice`'s bounds are non-negative, ordered and inside the axis
   because `Aten_shape.resolve_slice` applied the defaulting, the negative-index
   normalization and the clamping *before* the payload existed — one helper, so
   the bridge and the importer cannot drift. What is deliberately not delegated
   is the decision that the result is EMPTY: that is a fact about the extent, so
   it lives in `output_shape`, where `Graph_builder` and a JSON-decoded graph
   meet it too. The same split applies to `Pad`'s crop. A rule that only the
   importers enforce is a rule the builder and the JSON decoder do not have.

   That leaves a check the op still owns: `Slice.output_shape` rejects bounds
   outside `0 <= start <= stop <= extent`, unreachable from either importer and
   reachable from both of the other two constructors — and it is what keeps
   `Compute`'s read in bounds, since `clamp_low` bounds the source coordinate
   below and nothing bounds it above.

   Keep the native graph surface one-to-one with ATen overloads when an overload
   has a distinct target and user-visible parameter contract. It is fine for the
   implementation to delegate internally after translating params (for example,
   `Conv2d_padding` lowers `"valid"`/`"same"` into concrete `Conv2d` windows),
   but the graph constructor, JSON tag, builder function, and bridge arm should
   still name the distinct overload.
   - `Mul`: `type t = Bin.t`, `jsont = Bin.jsont ~name`, etc.;
     `output_shape = broadcast_output_shape` (the shared equal-or-1 broadcast rule
     in `pointwise.ml`), and `pixel ~a_shape ~b_shape a b out` reads each operand
     through `Pointwise.broadcast_coord ~index_zero:S.index_zero <shape> out`. `load` is
     **strict** (an out-of-bounds index raises), so a binary elementwise op must
     reduce the output coord against each operand's own shape first — that is what
     fans an extent-1 axis out without an OOB read. A unary op whose input already
     has the output shape (relu) reads at `out` directly. When the op takes operand
     shapes, thread them from `Eval_op` (`shape_of`) and the bridge runner.

2. **Graph IR** — `lib/native/graph_ir.mli` and `graph_ir.ml`
   - Add the constructor `| <Op> of <Group>.<Op>.t` to `type 'g gop` (alphabetical)
     in **both** files.
   - Add one entry to `op_registry` in `graph_ir.ml` — a `(module struct include
     <Group>.<Op>  let inject t = <Op> t  let project = function <Op> t -> Some t |
     _ -> None end : OP)` in alphabetical position. That single line wires up JSON
     encode/decode, `operands`, `map_operands`, and `pp`: those are folded over the
     registry, so there are **no** per-op match arms to add for them. (`Subgraph` is
     the lone exception, handled inline because it carries the recursive graph.)

   These two — `Graph_shape` and `Eval_op` — are the only places that still match
   per op, because they need shape / semantics context the payload can't carry.
   Their record patterns qualify the first label so the payload's module is
   unambiguous (the `node.Node.outputs` convention), e.g.
   `| Mul { Pointwise.Bin.a; b } -> …`.

3. **Shape dispatch** — `lib/native/graph_shape.ml`
   Add the arm calling your op's `output_shape` on the operand signatures.

4. **Eval dispatch** — `lib/native/eval_op.ml`
   Add the arm instantiating `<Op>.Compute (S)` and calling `C.pixel`. This is the
   single functor applied once at `Direct` and once at `Symbolic`. Absent optional
   operands are materialised with `fill` (zeros bias / ones weight).

5. **Claim transfer** — `lib/native/transform/output_transfer.ml`
   Add the op to `classify`, per output: `Reindexing` if every output element is
   **copied from an input element with no arithmetic** — a permutation
   (`Permute`, `Reshape`) or a selection (`Unbind`, one slice per output) both
   qualify, because the `Approximate` claim it carries is per-element, so it does
   not depend on the whole value multiset surviving; `Discontinuous` if an
   arbitrarily small input change can switch the result (an argmax); otherwise
   `Continuous`. The
   match is exhaustive with no default arm on purpose — a defaulting classifier
   would silently mis-transfer a new op, and mis-transferring means a verifier
   asserting an equality the graph does not guarantee. See
   `.ai/native_transform_design.md` §8.

5b. **Permute allowlists** — `lib/native/transform/passes/sink_permute.ml` and
   `reuse_permute.ml`
   Both match an explicit list of ops rather than reaching for
   `Graph_ir.operands`, because only an op with no axis-specific semantics
   commutes with a permutation. If your op reads and writes each output slot
   independently — every pointwise op, whatever its parameters — add it to
   `Sink_permute.elementwise`, and to `Reuse_permute.binary_elementwise` too if
   it has more than one tensor operand. Unlike the `Output_transfer` match these
   have a default arm, so omitting an op is silent: nothing breaks, but the op
   becomes a barrier that strands a permute at every occurrence, which for an
   activation means one per layer. `permute_passes_test.ml`'s
   `sink_permute_allowlist` fixture is the regression — extend it, and the
   whole graph should collapse to a single trailing permute.

   Scalar op *parameters* (a clamp bound, an `Add_scalar` scalar) do not disturb
   this: they are per-op constants, not per-axis data.

6. **Builder** — `lib/native/graph_builder.ml` and `graph_builder.mli`
   Add the constructor function (alphabetical) in both. It just appends a node via
   `op1 ?name ~kind:"<op>" (<Op> { … })` (first label qualified); output shapes are
   computed by `Graph_shape`, never passed in.
   - `let mul ?name a b = op1 ?name ~kind:"mul" (Mul { Pointwise.Bin.a; b })`

7. **ATen bridge** — `lib/native_aten_bridge/op_bridge.ml`
   - One `dispatch` arm (alphabetical by op name) matching the ATen
     `node.target` string(s) — include both the functional and in-place variants
     when they share semantics (e.g. `"…aten.mul.Tensor" | "…aten.mul_.Tensor"`).
   - The arm builds a small native graph rather than calling the `Compute`
     functor itself: `native_tensor_arg` for each tensor operand, then
     `build_g ~name [operands] (function [ids] -> … | _ -> assert false)` with a
     `Graph_builder` body, piped through `some_graph`. That is what lets an arm
     insert relayout `permute`s around the op (see the batch-norm arm) instead of
     being limited to one node. Decode non-tensor args with the `Interp_decode`
     (`D.`) helpers as the reductions do.
   - Bridge only when operand/result layouts already line up under `of_aten`'s
     right-aligned positional mapping. Ops needing NCHW↔NHWC relayout are the
     deferred class documented in `native_aten_bridge_layout.md`.
   - A schema `Scalar` arg crosses as either `Argument.Int` or `Argument.Float`
     — never assume one. Decode with `D.scalar_arg_result` /
     `D.scalar_opt_arg_result` (importer side: the local `scalar_arg` /
     `scalar_opt_arg` in `native_interp.ml`), not `float_arg`, which rejects
     the integer spelling. Both hand back an `Aten_scalar.t`, which also admits
     `Bool`, so narrowing to a native float is an explicit checked step.
   - A **Tensor-typed arg may hold a bare scalar**: the exporter writes
     `add.Tensor(x, 3)` with `as_int` rather than lifting a constant tensor.
     The ATen path materialises those via `full_like`
     (`Interp_decode.tensor_or_scalar_arg`); the native path routes them to a
     scalar-parameter op instead. Any binary arm whose ATen counterpart accepts
     a Number needs the tensor-vs-scalar branch, or it will fail to decode a
     real graph.
   - Decode every argument the schema declares, including the ones you do not
     implement. `alpha` on add/sub is the cautionary case: ignoring it computes
     `self + other` while the schema says `self + alpha * other`. Reject with
     `Validation_failure` rather than dropping it.

8. **Random-walk fuzzing** (optional) — `lib/native_op_walk/`
   A `<op>_nwalk.ml` assembling the op's `Walk` config space into a subject, plus
   an entry in `native_op_walk.ml`'s list. Optional because it is a fuzz harness,
   not a correctness site, and it is not always appropriate: `Div` is deliberately
   not registered, since a random divisor hits zero and the resulting inf/NaN
   would make the walk flaky rather than informative.

## Multi-output ops and dead outputs

Most ops produce one output; a few ATen ops return a tuple. If yours does:

- `Graph_shape.output_shape` returns one shape **per output**; `Node.outputs`
  holds one id per output. `Eval_direct`/`Eval_symbolic` loop over them, so the
  builder must allocate all output edges (not just via the single-output `op1`).
  For a **fixed** arity, hand-roll the builder and name the edges (see
  `max_pool2d_with_indices`). For an arity derived from the operand signature,
  use `Graph_builder.opN`, which allocates one edge per inferred shape and
  imposes no expected count.
- If the outputs need **different** per-pixel computations, thread an output
  selector into `Eval_op.pixel` and have the op's `Compute` expose one pixel
  function per output (e.g. values vs argmax indices). If they run the *same*
  computation and differ only in a coordinate, take `~output` as an ordinary
  parameter of one `pixel` instead (`Unbind`).
- **If the output count comes from model data, bound it inside the op's
  `output_shape`, before the list is built.** A ceiling in the builder is too
  late — the builder calls `Graph_shape` first, so the allocation it means to
  prevent has already happened. `Shape_error.Output_count` is the row, the limit
  is `Kernel.Limits.Hard.outputs`, and the test is `>=`, matching
  `Kernel.Limits.create`. See `native_multi_output_design.md` §1a.
- Route a genuinely-dead output into a `Discard` sink
  (`Graph_builder.discard`); drop a size-0 ATen output entirely (the engine has
  no empty tensors). See `native_multi_output_design.md` for the full rationale.

## Tests (mirror the existing `add` cases)

- `test/native/compute_test.ml` — `Direct: <op>`: evaluate `Compute (Direct)`
  on a small hand-made tensor, pin the result with `[%expect]`.
- `test/native/graph_test.ml` — `Direct graph: <op> of …`: build a one-node graph
  with `Graph_builder`, run `Eval_direct.run`, pin the output.
- `test/native/graph_symbolic_test.ml` — `Symbolic graph: <op> …`: print the
  `Stage_program` DAG and assert `ground` matches `Eval_direct` for the same
  inputs (covers the Symbolic path through the shared `Eval_op` functor).
- `test/native/graph_json_test.ml` — a codec round-trip, if the payload
  introduces a JSON shape the other ops do not already have (a bare number, an
  independently-optional field). Use a value that is not f32-exact (0.1) so the
  test distinguishes a narrowed scalar from an unnarrowed one — printing alone
  does not.
- `test/native/permute_passes_test.ml` — extend the `sink_permute_allowlist`
  fixture (site 5b).
- `test/native_bridge_test.ml` — `dispatch: <op>…`: drive `Op_bridge.dispatch`
  through `dispatch_print` with hand-derived expected values (independent of ATen
  as oracle). Needs the `aten` C++ build.

The first suites are pure OCaml (`dune runtest test/native`); the bridge test
needs the libtorch-backed `aten` library.

### Getting a real ATen oracle — which harness actually compares

Three harnesses look like they compare against ATen; only some do, so pick
deliberately rather than assuming coverage:

- **`lib/native_op_walk`** compares `Direct` against `Symbolic`. Both instantiate
  the *same* `Compute` functor, so it validates staging, scheduling and shape
  machinery — **not** the pixel math. A wrong definition passes it.
- **The generated ATen walks** (`test/native_walk_test.ml`,
  `test/pt2_op_native_walk_cram.t`) are the real oracle. Most ops get one for
  free: once your bridge arm exists, the op flips from `skipped (no native
  impl)` to `matched`. Check whether yours is in the `needs_meta` list at the
  end of `native_walk_test.ml` — if it is, the generator could not synthesise a
  valid config (clamp's schema defaults are the both-`None` pair ATen rejects),
  and you must add a `Walk_meta` entry in `lib/aten_gen/walk_meta.ml` plus a
  recipe under `lib/aten_walk_recipes/`. Even when a default walk exists it may
  only exercise the schema defaults — hardtanh's are `(-1, 1)`, never
  MobileNet's `(0, 6)` — so an override is worth it when the parameters you
  care about lie outside them.
  - Make correlated parameters **one** axis, not several. Independent axes can
    step into a combination the op rejects, which the walk then reports as a
    spurious mismatch; a candidate list of whole valid configurations makes the
    invalid state unreachable instead.
- **`test/pt2_node_bridge_cram.t`** is ATen-vs-native over real serialised nodes,
  but only over `test/data/resnet18/`. Note that `bin/pt2_spec_gen` **skips** any
  node whose Tensor-typed argument holds a bare scalar, so the tensor-or-scalar
  path is unreachable from fixtures. Cover that with a dual-path expect test in
  `test/native_bridge_test.ml`: `Interp_verify.dispatch ~verify:true` runs the
  node through real ATen *and* through `Op_bridge` + `Eval_direct` and compares
  with `Verify.verify_node`. Silence means agreement — so confirm the test can
  fail before trusting it.

**An in-place target needs a NON-idempotent value to actually exercise the
oracle.** `Interp_verify.dispatch` and `Aten_spec_run.run` (the engines behind
`test/native_bridge_test.ml`'s `verify_print`/`verify` and every generated ATen
walk) both now capture native's `Op_bridge` read of the input BEFORE calling
`Interp_dispatch.dispatch`, because a real in-place ATen op (`Tensor(a!) self`)
mutates its argument through the very handle the harness holds — comparing
after that call would silently compare against ATen's own output. This was a
live bug until op5.md's Group 5 exposed it: `relu_`/`hardtanh_` never caught it
because clamping is idempotent (`f(f(x)) = f(x)`), so applying the mutated
value twice was invisible. Adding an in-place op whose function is NOT
idempotent (most of them) is what actually tests this ordering — do not treat
a green `relu_`-style precedent as proof the harness reads pre-mutation state.

## Verify and commit

```sh
make format        # mandatory before commit
dune runtest test  # native + bridge inline tests (pt2/interp stay gated)
```

Optional follow-up: add an ATen-vs-native cross-check in
`test/aten_spec_run_test.ml` (+ a codec round-trip in `test/aten_spec_test.ml`)
so the verify harness exercises the op against the real ATen oracle — see
`aten_spec_design.md`.
