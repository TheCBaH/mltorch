# Machine IR — implemented design

Status: **in progress**. The project-owned representation below the CFG form of
the structured SSA IR (see the SSA IR design in this directory). Three
executable forms share one container: generic (primitive computation, byte
addressing, virtual SSA values), selected (one target's legal forms over virtual
values) and allocated (physical register views, spills, resolved transfers).
Each boundary has a verifier, a deterministic printer and an independent
interpreter; native execution and SIMD performance are later gates.

Loop transformations, numerical planning and algorithm choice stay in
structured SSA. Machine IR consumes the resolved computation: it never chooses
another reduction tree, enables FMA on its own, or reconstructs matmul from
instruction patterns. CompCert remains a separate C backend.

## Libraries and dependency direction

| Library | Role | Depends on |
|---|---|---|
| `lib/machine_ir` | Schema, types, containers, builder, verifier, printer, comparison protocol | `core`, `err_trace`, `expr`, `fmt` |
| `lib/machine_interp` | Byte memory, numeric primitives, generic interpreter, helper models | `machine_ir` |
| `lib/machine_lower` | Census; verified CFG SSA -> generic Machine IR | `machine_ir`, `ssa_ir` |
| `test/machine_ir`, `test/machine_lower` | Fixtures, malformed-IR and mutation evidence, census matrix | the above |
| `test/machine_source` | Source differential harness: reference, structured SSA, CFG, generic | the above, `native`, `ssa_lower`, the Loop/SSA test fixtures |

`expr` supplies only neutral language identities (axes, sources, local
variables) used by failure static identity. `machine_ir` never depends on
`ssa_ir`, `native`, Loop emission, Unix, Bigarray, CompCert or rivet. Target
interfaces will live in `machine_ir`; target implementations depend on it,
never the reverse. Both libraries are in the integer-signature audit's scope.

## Types, identities and data model

Ids are `Core.Tagged_int` domains (`Mir_id`: block, function, helper,
instruction, region, revision, site, slot, register unit, value, view); a
builder allocates them through bounded counters. Value types (`Mir_type`):
`f32`, `f64`, `i8/i16/i32/i64` (width without signedness — operations choose
it), `pred`, `ptr64` (not an integer), fixed vectors/masks, and `order`, the
compile-time sequencing state with no storage. Constants (`Mir_const`) are typed
raw bits: binary32 in the low 32, integers canonical zero-extended, predicates
0/1. Sizes, offsets and alignments are `int64` with checked arithmetic
(`Mir_layout`); host indices are narrowed only after a fit check.

The one data model is 64-bit pointers, little-endian, Linux/ELF; a generic
program records it without naming an ISA.

A source `Index` is an `i32` holding its checked signed 32-bit domain; byte
offsets exist only in Machine IR (SSA's unused `Offset` scalar was removed when
this IR took byte addressing). `Local` becomes an object pointer and an SSA
effect becomes order state.

## Program, blocks and order

A program holds functions, regions (byte objects: caller-bound, constant or
program-owned uninitialized), views (a window of a region with a permission,
role and optional source identity — overlapping views are allowed and distinct
ids prove nothing about disjointness), helper descriptors, the data model, the
planning summary and a revision id that never enters normalized text.

A block has typed parameters, one order parameter, straight-line instructions
and one terminator. Every edge binds its target's parameters and order
parameter simultaneously. Generic terminators: `branch`, `jump`, `return`, and
`fail`, which commits an already-computed failure kind and payload and ends the
invocation without reevaluating anything.

An instruction's uses, results and effect class are derived from its opcode
(`Mir_op`, `Mir_typing`); there is no cached operand or clobber list to
override. Ordered operations (call, event, load, store) consume the block's
latest order state and produce the next; terminators consume the latest. A
stale or missing order state is a verifier rejection. `Partial` operations
(`sdiv`/`srem`, `fcvt.f64.i64`, `narrow`) are pure but domain-restricted and are
never speculated.

## Generic operations

Address of a view; bitcast (i32<->f32, i64<->f64 only); call (function or
helper); const; copy; event (a logical mark with multiplicity); float binary
(`add`, `div`, `max` — IEEE maximum, NaN propagates and +0 > -0 — `mul`,
`sub`), compare (ordered `eq`/`le`/`lt`, `uno`), conversions (each one rounding:
`fext`, `fround`, `scvt.i64.f32`, `scvt.i64.f64`), `ffma` (always one
rounding), `fcvt.f64.i64` (defined only on finite values in `[-2^63, 2^63)`),
unary (`neg`, `sqrt`, `trunc`); modular integer arithmetic and shifts (count a
constant below the width), comparisons, signed division/remainder (defined only
for a nonzero divisor and not `min / -1`), sign/zero extension, truncation,
`narrow` (signed narrowing defined only when the value fits: the realization of
an SSA in-domain proof, a defect in interpretation otherwise); raw
little-endian load/store of a width with an alignment; predicate logic;
`ptr.add` (pointer plus signed i64 bytes); select.

## Verification (generic)

Shared structure (`Mir_check`, reused by later stages): unique ids and single
definitions; every block reachable; the entry never a branch target; every use
dominated (same block earlier, or a dominating block); each use carries its
definition's type; edge arity and types; the order chain; regions (size,
power-of-two alignment up to 4096, constant bytes), views (inside their region,
never writable over constant bytes), helpers (no order types). The generic stage
adds: shift counts; static access permissions for addresses derived from a
view; failure payloads against their kind's schema; returns against the
function's results; and **guard evidence** for `sdiv`/`srem` and `fcvt.f64.i64`.
Evidence is a predicate fact established by a dominating single-predecessor
branch edge (closed under `and`/`or`/`not`): `y != 0` and `not (x == min && y ==
-1)` for division, `-2^63 <= x` and `x < 2^63` (which a NaN fails) for the
conversion. Without it the operation is refused, never assumed.

Diagnostics (`Mir_diagnostic`) name the stage, function, block, instruction
and a closed problem variant. A verified program exists only as
`Mir_verify.Generic.t`, the verifier's result.

## Printing

`Mir_pp` names blocks in reverse postorder and values in definition order, so
two builder histories of one program print the same text; the revision never
appears. Stages print their own opcodes through the same naming.

## Census and support matrix

`Mir_census` matches every SSA operation, decode, encode, conversion and type
exhaustively, giving its generic expansion or named helper, the plan slice
that admits it, its observable, and each failing condition classified as a
**language failure** (the guard ends a generic block; the operation's effect
runs only after all its guards) or a **compiler invariant** (a verified proof
or interpreter defect check, never a failure row). The classification follows
the SSA interpreter, not `Ssa_effects.may_fail`: `Index_of_i64` outside the
index domain, an invalid in-domain add/scale, a flat or proven access outside
its buffer, a local write outside its object and an anonymous-local or
unwritten read are defects; a named local read outside its object is an
`unbound_local` failure. The test prints the matrix.

## Planning provenance

The SSA plan does not retain the target it consumed and a bare SSA/CFG program
carries no FMA provenance. `Mir_planning.t` is the resolved, immutable,
line-serialized summary: subject (a digest of the planned program's text),
policy and schedule identity, working precision, logical lane width, FMA mode
(`exact`, `forbidden`, `relaxed_madd`) and required capabilities — data, never
the closure-bearing target. `admit` refuses a missing summary, one bound to a
different subject, and a contraction under `forbidden`. Permissions are never
inferred from opcodes.

## Failure records and the comparison protocol

A failure (`Mir_failure`) is the SSA interpreter's row: kind, static identity
(source and axis, overflow operator, meter limit kind, scan axis and variable,
local variable) and typed payload. The model record keeps each kind's existing
schema — `kind:int32` at 0, `invocation:int32` at 4, twelve `int64` words from
8 — and only `scan_projection` (word 5) and `unbound_local` (word 0) hold a site
word. A site binds to the first compatible entry of the bundle's own site table
(the SSA C emitter's rule); a reachable site-bearing failure with no compatible
entry is a refusal, and decoding the one-past-table sentinel is a defect.

`Mir_observation` normalizes a route's run: status (success, failure row,
defect, unsupported capability, exhausted fuel), defined cells of each public
output and logical event counts. `Mir_compare.observations` agrees only on two
successes with equal outputs and events, or two failures with equal rows and
equal failure-prefix events. Defects, unsupported capabilities and fuel never
agree, even with themselves. Integers and signed zeros compare exactly; a NaN
matches any NaN. Rows compare decoded static identity, payload and invocation;
raw site numbers only under an explicit shared numbering. Machine IR's `ffma`
is always compared against the fused (`fused:true`) oracle; only
`relaxed_external`, for an external engine whose summary says `relaxed_madd`,
accepts the unfused oracle's cell instead.

## Generic interpreter

`lib/machine_interp` executes a verified generic program by its own dispatch;
it never calls the SSA interpreter, `Ssa_scalar` or a generated kernel.

- **Memory** (`Mir_memory`): one instance per region at a deterministic
  synthetic base address (page aligned, with an unmapped gap after it), stored
  as sparse 4 KiB pages with a per-byte initialization map, so a region wider
  than 4 GiB costs only the pages touched. A pointer is (instance, view window,
  permission, region offset); `ptr.add` defects on wraparound; every access
  checks the whole byte range against the view window and the region, the
  synthetic address's alignment, the permission and each byte's
  initialization. One-past pointers exist; dereferencing one is a defect.
  Nothing is a host pointer.
- **Numerics** (`Mir_numeric`): exact bits throughout. Binary32 `+ - * / sqrt`
  are the binary64 result rounded once more (exact for these operations);
  binary32 FMA forms the binary64 sum rounded-to-odd (TwoSum error, sticky
  bit) before the final binary32 rounding; i64 to binary32 shortens the
  magnitude to 53 bits with round-to-odd first; maximum, the checked
  float-to-i64 domain and integer division domains are written out. Shared
  primitives with the SSA oracle: binary64 arithmetic, `Float.fma` and the
  host's binary64-to-binary32 rounding. Tests cross-check FMA and i64-to-f32
  against the SSA library's separate derivations (random and constructed
  double-rounding cases).
- **Control**: an iterative block dispatcher; edge arguments are all read
  before any parameter is rebound; calls nest to a bounded depth; a step
  budget bounds the run. A `fail` stops at once with its row; a callee's or
  helper's failure propagates unchanged.
- **Outcomes**: success with results; a failure row; a defect (`bad_access`,
  `domain`, `invalid_program`, `uninitialized`) with its location; unsupported
  capability; exhausted fuel (never a scan-budget failure). Events are counted
  per kind and survive a failure (the failure prefix).
- **Helpers**: a helper id binds to a deterministic `Mir_helper_model`; a
  declared helper with no model makes the run `unsupported` before anything
  executes, and a model raising a failure its descriptor does not declare is a
  defect.

## CFG to generic lowering

`Mir_lower.program` verifies the structured program, admits the planning
summary against its subject (refusing a missing or foreign summary, a
contraction under `forbidden`, and binary32 arithmetic under a binary64
summary — a binary32 value produced by a rounding conversion is not binary32
arithmetic), lowers to the CFG, translates it and verifies the result; a
generic program the verifier rejects is reported as a lowering defect, never
returned. Anything outside the admitted slice is a typed refusal naming the
census slice that will admit it.

- **Layout** (`Mir_layout_map`): each SSA buffer becomes one region and one
  view sized `elements × element bytes` with checked arithmetic; inputs are
  read-only bound regions, outputs read-write bound regions, scratch
  uninitialized. The kernel function takes no parameters and addresses
  buffers through `addr view`.
- **Values**: an SSA effect has no machine value — the order chain follows
  control flow and original instruction order, which is the effect order —
  so effect parameters and arguments are dropped. `Index` is `i32`.
- **Guards**: each source-failing check ends the current generic block: the
  true edge continues in a fresh block, the false edge reaches a block that
  computes the payload and fails. Coordinate checks run per axis in N T D H W
  C order (the first axis outside wins, payload all six coordinates
  sign-extended); `Index_add`/`Index_scale` compute in i64, guard the index
  domain and truncate, with payload `(lhs, rhs)` (`(k, x)` for a scale).
  In-domain variants use `narrow`. A checked load's guards precede its
  address formation and read.
- **Access**: the row-major element offset in i64 (sign-extended
  coordinates, extents as constants), scaled by the element bytes, added to
  the view's address. Decodes and encodes are explicit (`load.i32; bitcast;
  fext` for binary32, `fround; bitcast; store.i32` back). Unchecked accesses
  keep only the interpreter's byte-range defect check.
- **Origins**: every instruction records its CFG block and position, the SSA
  output origin and its expansion role.

- **Integers and conversions** (the M4.1 slice): `i64.div` guards a zero
  divisor, then `min / -1`, before `sdiv`; `float.to_i64` guards NaN, then an
  infinity, then `[-2^63, 2^63)` (a NaN or an infinity also fails the range
  test, so the order is the row) before `fcvt.f64.i64`; `i64` to binary32 and
  binary64 are single `scvt` conversions; floor and ceiling division by a
  positive literal adjust the truncated quotient by the remainder's sign and
  `narrow`; `Index_of_i64` is `narrow`; maximum is IEEE `fmax`; the pool
  predicate is `olt(best, value) or uno(value)`; a gather check guards
  `[-extent, extent)`; `Float_fma` is `ffma`, admitted only when the summary
  permits contraction.

`Mir_ssa_rows` maps an SSA interpreter failure to the Machine IR row it must
equal; it is the oracle adapter, used by tests and model hosts.

`Mir_lower.Mutation` is fault injection for the evidence suite only
(conversion guard order, double rounding through binary64, eager load, guard
order, operand order, byte scaling, sequential transfer, zero extension);
every mutation is detected by the source harness.

## Source differential harness

`test/machine_source` lowers fresh source kernels (pointwise, noncommutative
arithmetic, shifted and competing coordinate failures, index overflow, matmul
of odd shapes, empty and reversed reductions) and directly built recurrences,
and runs each through the reference, the structured SSA interpreter, the CFG
interpreter and the generic interpreter over bytes packed by the harness (not
the SSA memory; outputs start zeroed as in the existing hosts). The generic
route must agree with the SSA routes on outputs as exact binary32 cells, on
failure rows and on logical marks. The reference's own verdict against SSA is
shown alongside: on a 63-bit host it does not report the index overflow the
SSA IR checks for, which is the existing documented difference.

## Host numerics

OCaml's arm64 backend contracts `x *. y +. z` into one fused instruction. Any
host computation that must round a product before a sum (an unfused oracle, a
two-step reference) hides the product behind `Sys.opaque_identity`;
`Ssa_scalar`'s unfused `Float_fma` path does.
