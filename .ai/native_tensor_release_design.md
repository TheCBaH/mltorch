# Releasing tensor payloads after their last use — design

Covers the direct evaluators of both dialects: `Eval_direct.run`
(`lib/native`) and `Eval_direct4.run` (`lib/native4d`). The shared piece is
`lib/native/release_schedule.ml`.

## 1. Problem

Every node output is a fresh, dense, exclusively owned `Bigarray.Array1`
(see `native_tensor_design.md`). The evaluators fold over `g.nodes` and bind
each result into `env : Tensor.packed Tensor_id.Map.t`. Before this change
nothing was ever removed, so every intermediate stayed reachable until `run`
returned and peak payload memory was the **sum of all intermediates** rather
than the largest set alive at one time.

## 2. Contract: `Retain.t`

```ocaml
module Retain : sig
  type t =
    | All                       (* every edge in the result: the default *)
    | Only of Tensor_id.Set.t   (* these edges plus g.outputs; the rest released *)
end
```

`run` takes `?retain`, defaulting to `All`, so a caller that passes nothing
sees exactly the old env. Under `Only s`:

- the result holds exactly `g.outputs` and the members of `s` that were
  bound;
- every other edge — intermediates, inputs and constants alike — leaves
  `env` right after its last reader runs, so its payload is garbage from then
  on;
- outputs are `Tensor.equal_bits` to an `All` run, and every error is the same
  error at the same node.

Inputs and constants are released from `env` too, for uniformity; that frees
nothing while the caller still holds them. Neither setting ever binds an
index output nothing reads (the second output of the argmax-style ops): it is
not allocated at all.

`Only Tensor_id.Set.empty` is the inference setting. `All` stays the default
because some callers read intermediates: `fold_const` reads a named non-output
edge (it passes `Only {out}`), and `Lower.evaluate` returns the whole env.

## 3. One definition of "reader"

`Release_schedule.schedule ~operands ~is_sink ~retain g` is one backward pass
over `g.nodes`. The first non-sink node met that reads an edge is its last
reader. The schedule answers three questions:

- `after s n`: edges to drop once node `n` has run — each operand whose last
  reader is `n`, and each output of `n` nothing reads;
- `initial s`: graph inputs nothing reads, dropped before the first node
  (there is no node to drop them after);
- `has_reader s id`: a graph output, or an operand of a non-sink node.

Retained edges and `g.outputs` are never scheduled; under `All` nothing is,
but `has_reader` still answers. A `Discard` sink is not a reader.

Allocation and release both read this one schedule: the evaluator's
dead-index-output filter calls `has_reader`, so the two cannot disagree about
what a reader is. The function is generic over the op type
(`~operands`/`~is_sink`) rather than a functor, because the two dialects share
`Graph_common.Graph` but not their op type, and there are only two callers.
It is keyed by `Node_id.t`/`Tensor_id.t`, never by a position, so it needs no
bound for js_of_ocaml's 32-bit `int`.

A node's releases happen after `hooks.on_end`, so a hook runs while the
node's operands are still bound.

## 4. Why dropping references is enough

A payload has no views, strides or shared buffers, so it is reachable only
through `env`, the per-node `operand_env`, and whatever the caller holds.
Removing a binding cannot free memory another edge still uses. An op that
returns its operand's `packed` unchanged makes two ids point at one block,
and the GC keeps it until both are dropped. There is no double free or use
after free to guard against: this is a scheduling problem, not an ownership
one. "Release" means dropping the last reference — OCaml has no `free` for a
Bigarray.

A `Node_executor`/`Region_executor` that keeps an operand beyond its call (a
cache, a shadow comparison) keeps that payload alive. That is only less
saving, not a hazard — until buffer reuse, where it becomes one.

## 5. `peak_bytes`

`Release_schedule.peak_bytes ~is_index_output g s` simulates the fold using
only `Tensor_sig`s: `g.inputs` are resident throughout; each allocated output
adds `numel × cell_bytes` when produced; a released node output subtracts it.
The peak is sampled after a node's outputs exist and before its releases,
since both are live while it runs. `numel` goes through
`Vec6.numel_bounded` below `Kernel.Limits.Hard.numel`, the arithmetic is
`int64`, and the running sum has its own overflow check
(`` `Peak_bytes_overflow ``), since bounded terms do not bound their sum.

It turns the saving into a deterministic figure rather than a GC measurement.
Measured on the cram models (MiB, inputs and constants included), `All` →
`Only empty`: mobilenetv2_050 lowered 112.7 → 13.0, canonical 35.7 → 12.6;
fastvit_sa12 lowered 380.7 → 54.1, canonical 200.6 → 53.9.

Observed peak RSS for the lowered graph (`native_graph eval`, native, MiB),
`All` → `Only empty`: mobilenetv2_050 129.0 → 55.9, regnetx_002 98.3 → 57.5,
fastvit_sa12 398.3 → 145.8, with outputs byte-identical and wall time
unchanged. Under `Only`, RSS sits 15–27 MiB above the process floor plus
`peak_bytes`: the GC reclaims released Bigarrays without explicit `Gc` calls,
with a modest lag. That lag is the input to any arena/buffer-reuse decision;
it has not been measured under node.

## 6. Verification

- `test/native/release_schedule_test.ml`: the schedule and `peak_bytes` on
  hand-built graphs over a toy op.
- `test/native/eval_direct_release_test.ml` and its Native4D twin: `All` vs
  `Only empty` equivalence over shared fixture graphs, a mid-graph error, and
  the exact key set of an `Only` result.
- `test/native_gc/`: native only, since js_of_ocaml's `Weak`/`Gc` give no
  such guarantee. On Native an executor stashes an intermediate in a `Weak`
  slot and, when a later node starts, forces a major collection: the slot is
  empty under `Only` and full under `All`. Native4D has no per-node seam, so
  its twin watches an input only the env holds, after the run. Removing the
  release fold turns both red.

## 7. Future work this enables

- **Buffer reuse.** A released payload of the same format and cell count
  could back the next output. The schedule is the lifetime proof reuse needs,
  but an executor that retained an operand would see it overwritten, so reuse
  needs its own design.
- **In-place elementwise ops** when an operand's last reader is the op
  itself; same hazard.
- **Ordering for memory.** `g.nodes` is one topological order among many;
  choosing one that lowers `peak_bytes` is a separate problem that §5 makes
  measurable. The proposed
  [memory-aware scheduling design](native_tensor_arena_schedule_design.md)
  separates dependency order from storage readers, regenerates releases for
  each order, and checks actual arena pool sizes before runtime acceptance.
