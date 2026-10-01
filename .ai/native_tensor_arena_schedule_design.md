# Memory-aware topological scheduling — design record

Status (2026-10-01): proposed; implementation not started. Detailed working
design, staged implementation plan and tracker are in `ai/`. This record holds
the durable contract; existing allocator/evaluator behavior is described in
[the arena record](native_tensor_arena_design.md),
[release record](native_tensor_release_design.md) and
[interval allocator record](interval_alloc_design.md).

## Goal and optimization boundary

Reorder the nodes of a validated Native graph to reduce simultaneous live
tensor payload and accept the order for arena execution only when checked
placement shows no increase in allocated pool bytes. Preserve every operation,
operand, id, signature, graph input/output order, structural group and
provenance binding. Run after graph transforms and shape inference, before
planning. Leave the rewrite framework's stable topological sorter unchanged.

The scheduling engine optimizes exact simultaneous target payload bytes.
It scores Kahn-ready nodes using two constructive policies (peak first and
live first), with optional deterministic bounded beam search. Original node
positions determine ties. Keep the original order as incumbent; always return
a complete valid permutation. No recomputation, pruning, fusion, in-place
mutation or concurrent execution is part of this pass.

Ready-node weights come from existing `Alloc_script.Alloc.bytes`, using
`Core.Storage_units.Byte_size` checked arithmetic. Alignment never changes a
scheduling weight. Three distinct metrics must remain visible: simultaneous
payload peak, sum of payload peaks per pool, and actual placed pool bytes.
They do not imply one another. Only actual checked pool comparison establishes
arena savings; none establishes an RSS reduction.

## Lifetime adapters and verification

Intermediate mode obtains allocation facts from `Eval_direct.dry_run` with
explicit effective retention. Dependencies include every operand, including
`Discard`; readers are distinct non-sink consumer nodes. Duplicate operands
count once. Every output is allocated before releasing last-read operands;
multi-output nodes are indivisible. Unread ordinary outputs overlap their
producer's inputs; skipped unread index outputs are absent. Graph outputs and
retained tensors remain live; quantized outputs count outside eligible pools.
Inputs/constants are fixed caller storage outside this script.

Role mode obtains facts from `Eval_direct.storage_script` under unchanged
layout, ownership, retention and alignment. `Storage_script.Block.arena`
determines pool membership, never `alloc.eligible`. Pools are identified by
logical arena and Bigarray kind. Replay the fixed preparation/population
prefix, initial frees and publication/release suffix. Target execution arenas
exclude persistent `Constants`. Copied reclaimable inputs start live and free
after their last reader; borrowed/outside inputs follow existing non-reclamation
rules. Forwarded inputs/constants are single blocks with their longest lifetime.
Outputs and retained intermediates remain pinned through result release.

Candidate frees must be recomputed, not moved with old script chunks. Validate
the original and selected graphs with `Graph_view.of_graph`. Compare predicted
metrics to fresh scripts before accepting an order: exact eligible replay and
`Alloc_script.peak_bytes` in intermediate mode; per-kind exact lower bounds
from `Arena_problem.Kind_problem.script`; matching
`Storage_script.peak_bytes ~where` predicates in role mode.
`Arena_problem.combined_bound_bytes` uses padded sizes and cannot validate an
exact combined payload peak.

## Search, planning and execution

Beam width, child-expansion budget, graph size and estimated state-array bytes
are explicitly bounded. No clock belongs in the core library. Stream children
into bounded survivors; budget exhaustion returns the best completed incumbent.
State/work arithmetic is checked before JS array narrowing. Increasing budget
at fixed width must not worsen the engine's payload incumbent; increasing
width need not improve it. A compulsory-node footprint is an order-independent
payload lower bound, extended with fixed-prefix/result bounds for role mode.
Matching that bound or exhaustive tiny enumeration proves only payload
optimality. Width-pruned search proves neither infeasibility nor pool optimality.

A separate orchestration layer plans a bounded complete-candidate portfolio:
original, both constructive orders and best complete beam result, deduplicated
by order. Use the existing witnessed planners with identical alignment,
allocator budget/seed, limits and lifetime policies. Among admitted candidates
whose target payload peak is no greater than baseline, minimize actual pool
bytes, then target peak, then outside peak; keep original on an exact tie.
Intermediate pool size is `Arena_plan.stats.pool_bytes`; role pool size is
`Storage_plan.footprint.execution`. Return chosen graph and matching plan
together and reuse the winning plan. Never schedule inside an arena-only API
whose caller still executes the original graph.

If a feasible baseline cannot be improved, keep it. Candidates may rescue a
baseline planning/admission refusal. Intermediate mode preserves `Required`
errors and `Best_effort` original-order release fallback when all are refused;
role mode retains its existing errors, without a new admission/fallback API.
Compare metadata only; prepare pools/constants only for the winner. Share the
existing `Arena_run` plan checks and acquire from the chosen plan to avoid
replanning. `Required` uses `Arena.footprint`, including out-of-arena bytes.
Validation, script, arithmetic and witness failures remain correctness errors. Keep the existing
`Arena_script_mismatch` / `Storage_script_mismatch` entry checks.

Execution remains opt-in. Successful runs must be bitwise identical; hook
order and the first error among independent failures may change. Custom
executors must opt into this ordering contract; future effectful nodes require
ordering edges. Provenance wrappers and hooks use the selected graph. Result
leases and constant versions retain their existing lifetime guarantees.

## Evidence required before closeout

Implement in stages: validation/contracts; exact replay; constructive policies
and independent tiny oracle; role adapters; bounded beam; pool acceptance;
explicit native/JS integration; paired corpus measurement. Validate malformed
graphs, duplicate operands, fan-out, sinks, skipped indices, retained outputs,
quantization, mixed kinds, overflow, search limits and original-order fallback.
Numerically compare original release, scheduled release and scheduled arena
runs. Reject new order with old plan before an executor call.

Use a separate deterministic scheduling corpus test so allocator-only goldens
keep their fixed-order interpretation. Report before/after exact payload,
per-pool peaks, padded diagnostics, witnessed pool sizes, refusals and work;
timing belongs in host artifacts. Capture source/corpus revisions and policies.
Zero accepted pool growth is required; actual reductions remain unmeasured.
Default-on execution, prepared-run caching, Native4D adaptation and RSS claims
require further evidence outside this initial plan.
