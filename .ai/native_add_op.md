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
   Add a module exposing `output_shape` and a `Compute (S : Semantics.SEMANTICS)`
   functor with a `pixel` function written against the abstract `S` value/index
   domain (so it runs unchanged under `Direct` and `Symbolic`). Elementwise ops
   live in `pointwise.ml`; reductions in `reduce.ml`; etc. Reuse the value basis
   in `semantics.ml` — `mul`/`add`/`sub`/`div`/`select`/`lt`/… already exist, so a
   new pointwise op usually needs no new primitive.
   - `Mul`: `output_shape = broadcast_output_shape` (the shared equal-or-1
     broadcast rule in `pointwise.ml`), and `pixel ~a_shape ~b_shape a b out` reads each operand
     through `Pointwise.broadcast_coord ~index_zero:S.index_zero <shape> out`. `load` is
     **strict** (an out-of-bounds index raises), so a binary elementwise op must
     reduce the output coord against each operand's own shape first — that is what
     fans an extent-1 axis out without an OOB read. A unary op whose input already
     has the output shape (relu) reads at `out` directly. When the op takes operand
     shapes, thread them from `Eval_op` (`shape_of`) and the bridge runner.

2. **Graph IR** — `lib/native/graph_ir.mli` and `graph_ir.ml`
   - Add the constructor to `type 'g gop` (alphabetical) in **both** files. Name
     operand fields as typed fields; optional operands are `tensor_ref option`.
   - Add the arm to `operands` and to `map_operands` in `graph_ir.ml`.

3. **Shape dispatch** — `lib/native/graph_shape.ml`
   Add the arm calling your op's `output_shape` on the operand signatures.

4. **Eval dispatch** — `lib/native/eval_op.ml`
   Add the arm instantiating `<Op>.Compute (S)` and calling `C.pixel`. This is the
   single functor applied once at `Direct` and once at `Symbolic`. Absent optional
   operands are materialised with `fill` (zeros bias / ones weight).

5. **Builder** — `lib/native/graph_builder.ml` and `graph_builder.mli`
   Add the constructor function (alphabetical) in both. It just appends a node via
   `op1 ?name ~kind:"<op>" (<Op> { … })`; output shapes are computed by
   `Graph_shape`, never passed in.
   - `let mul ?name a b = op1 ?name ~kind:"mul" (Mul { a; b })`

6. **ATen bridge** — `lib/native_aten_bridge/op_bridge.ml`
   - A `run_<op>` Direct-scheduler helper (alphabetical) that materialises the
     output via `Schedule.evaluate (output_shape …) (C.pixel …)`.
   - A `dispatch` arm (alphabetical by op name) matching the ATen `node.target`
     string(s) — include both the functional and in-place variants when they share
     semantics (e.g. `"…aten.mul.Tensor" | "…aten.mul_.Tensor"`). Use the `unary`
     / `binary` helpers for tensor-only ops; decode extra args with the
     `Interp_decode` (`D.`) helpers as the reductions do.
   - Bridge only when operand/result layouts already line up under `of_aten`'s
     right-aligned positional mapping. Ops needing NCHW↔NHWC relayout are the
     deferred class documented in `native_aten_bridge_layout.md`.

## Tests (mirror the existing `add` cases)

- `test/native/compute_test.ml` — `Direct: <op>`: evaluate `Compute (Direct)`
  on a small hand-made tensor, pin the result with `[%expect]`.
- `test/native/graph_test.ml` — `Direct graph: <op> of …`: build a one-node graph
  with `Graph_builder`, run `Eval_direct.run`, pin the output.
- `test/native/graph_symbolic_test.ml` — `Symbolic graph: <op> …`: print the
  `Stage_program` DAG and assert `ground` matches `Eval_direct` for the same
  inputs (covers the Symbolic path through the shared `Eval_op` functor).
- `test/native_bridge_test.ml` — `dispatch: <op>…`: drive `Op_bridge.dispatch`
  through `dispatch_print` with hand-derived expected values (independent of ATen
  as oracle). Needs the `aten` C++ build.

The first three suites are pure OCaml (`dune runtest test/native`); the bridge
test needs the libtorch-backed `aten` library.

## Verify and commit

```sh
make format        # mandatory before commit
dune runtest test  # native + bridge inline tests (pt2/interp stay gated)
```

Optional follow-up: add an ATen-vs-native cross-check in
`test/aten_spec_run_test.ml` (+ a codec round-trip in `test/aten_spec_test.ml`)
so the verify harness exercises the op against the real ATen oracle — see
`aten_spec_design.md`.
