# Adding a Native4D operation

The Native4D twin of `native_add_op.md`, and much shorter — the dialect is
deliberately reduced, and the op registry removes most of the parallel matches.
Read `native_add_op.md` first if you are adding the *Native* op this legalizes
from; this file covers only the four-axis side.

## Before anything: does it belong?

Native4D is a **deliberately reduced** representation (`native4d_design.md` §1).
A Native graph it cannot express is outside its domain, not a gap. So the first
question is never "how do I add this" but "should the converter reject it
instead". Adding an op widens what the dialect promises to represent, and every
op is a new obligation for the verifier, the shape rules and the evaluator.

Two forms the answer usually takes:

- **legalize, don't add.** `Linear` and single-batch `Bmm` become `Conv2D`;
  `Mean keepdim=false` becomes `MeanKeepDims` plus `Reshape4`. No new op.
- **reject.** General grouped convolution, batched `Bmm` and live max-pool
  indices are all rejected on purpose; `native4d_design.md` §8 lists what the
  smallest honest extension would be for each, if that ever changes.

## Ordering convention

Constructors **and** every per-op match arm stay in **global alphabetical
order**, as in Native. Slot the new op into position at each site; do not
append.

## The sites

1. **Payload** — `lib/native4d/ops4.ml`, *only if the op needs one*.

   Most do not. Eleven of the nineteen registry entries are Native's payload
   modules **unchanged** — `Pointwise.Add`, `Pool.MaxPool2d` and friends already
   expose the `name`/`jsont`/`operands`/`map_operands`/`pp` an entry wants, and
   their parameters name no axis and carry no shape, so there is nothing
   four-axis about them to restate. Reuse the Native module directly.

   Write a new payload only when one of these holds:

   | trigger | what changes |
   |---|---|
   | the op **names axes** | the field is `Axis4.t`, not `Axis.t`, so T and D are unnameable |
   | the op **carries a shape** | the field is `Shape4.t`, so it cannot leave the dialect |
   | grouping or a mode is involved | it moves from a parameter into the choice of **constructor** — the IR should not be able to hold a value the dialect rejects |

   That third row is the convolution case: `Conv2D`/`DepthwiseConv2D`/
   `TransposedConv2D` are three constructors precisely so `groups` and
   `transposed` are not numbers a graph can carry.

2. **Variant + registry** — `lib/native4d/op.ml`.

   One constructor, one registry entry:

   ```ocaml
   (module struct
      include <Module>
      let inject t = <Op> t
      let project = function <Op> t -> Some t | _ -> None
    end : OP);
   ```

   That single entry wires up JSON encode/decode, `operands`, `map_operands`
   and `pp`, all of which fold the registry. There are **no** per-op match arms
   for those — and unlike `Graph_ir`, no inline `Discard` case to work around,
   because the dialect has no zero-output op.

3. **Shape** — `lib/native4d/graph_shape4.ml`.

   Delegate to the **Native** op's `output_shape` and wrap the result through
   `four` (i.e. `Shape4.of_vec6`). Do not restate the shape rule: it would be a
   second definition, free to drift from the one the compute actually uses. The
   wrap is where an op whose output leaves the dialect is caught.

4. **Evaluation** — `lib/native4d/eval_op4.ml`.

   Translate parameters and call the **existing** Native `Compute (S)` functor.
   Native4D adds no numeric kernels; `native4d_design.md` §2 lists
   "reimplementing numeric kernels already expressed by the Native `Compute (S)`
   functors" as a non-goal. If a translation needs to permute a tensor to line
   up with shared compute, something is wrong with the layout choice — the
   weight layouts are Native's unchanged for exactly this reason (§6.1).

5. **Claim transfer** — the dialect's `Output_transfer` table.

   Per output: `Reindexing` if every output element is **copied from an input
   element with no arithmetic** — a permutation (`Permute4`, `Reshape4`) or a
   selection (`Unbind`) both qualify, since the claim is per-element and copying
   preserves it; `Discontinuous` if an arbitrarily small input change can switch
   the result; otherwise `Continuous`. Exhaustive with no default arm, same
   discipline and same reason as Native's.

6. **Builder** — `lib/native4d/builder.ml`, alphabetical, taking `Shape4.t` and
   `Axis4.t` in its public signature so an invalid graph is not constructible.
   `op1` for a fixed single output; `opN` when the arity comes from the operand
   signature (`Unbind`). `op1`'s arity check is not vestigial — it is what stops
   an op that grew an output from silently dropping it.

7. **Legalization** — `lib/native4d/lower.ml`, plus a row in the domain contract
   table if the new op changes what `Domain.check` accepts. State the claim
   (`Identical`/`Equivalent`) and *why* it is that and not stronger.

## Tests

- `test/native4d/op_json_test.ml` — add a sample. The **count check** in that
  file compares the sample list against the registry length, so a new op fails
  it until sampled; that is the one thing a round-trip test cannot catch by
  itself, since the case it would fail on is the case nobody wrote.
- `test/native4d/compute_test.ml` — Direct values, hand-computed. Not
  Native-as-oracle: both sides would instantiate the same functor, so agreement
  would prove staging rather than arithmetic.
- Direct versus grounded Symbolic, bitwise — in `compute_test.ml`, not in a
  `graph_symbolic_test.ml`; this file used to name one that does not exist.
  `Fixtures4.per_op` is count-checked against the registry the same way the JSON
  samples are, so a new op fails until it has a graph there.
- **If the op is multi-output**, check what the harnesses compare. Several took
  `List.hd` of `Graph.outputs` and so only ever looked at ordinal 0;
  `compute_test.ml`'s `agree` and `verify_test.ml`'s end-to-end comparison now
  render every output. A fixture with two or more outputs is what keeps that
  honest.
- `test/native4d/mutation_test.ml` — for a multi-output op, swap two of a NODE's
  output ids while leaving `Graph.outputs` alone. Nothing structural sees it;
  only `Map_verify` does.
- `test/native4d/verify_test.ml` — if the op is a legalization target, a mutation
  in its family, **observed failing before the fix is restored**.

## Verify

```sh
make format
opam exec -- dune runtest test/native4d test/native
```
