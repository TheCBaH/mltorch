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
| `lib/machine_target_aarch64` | AArch64 views and AAPCS64, admitted forms and semantics, selection, allocation registers | `machine_ir`, `machine_interp` |
| `lib/machine_target_x86_64` | x86-64 views and System V, admitted SSE2/GPR forms (SSE4.1, FMA3 under features), selection, allocation registers | `machine_ir`, `machine_interp` |
| `lib/machine_alloc` | Liveness and intervals, physical parallel copies, the reference allocator, frame realization | `machine_ir` |
| `lib/machine_check` | The independent symbolic allocation checker | `machine_ir` only — never the allocator |
| `test/machine_ir`, `test/machine_lower` | Fixtures, malformed-IR and mutation evidence, census matrix | the above |
| `test/machine_source` | Source differential harness: reference, structured SSA, CFG, generic | the above, `native`, `ssa_lower`, the Loop/SSA test fixtures |
| `test/machine_aarch64` | Selected forms, malformed rejections, generic vs selected (C3), selection mutations | the above |
| `test/machine_a64_native` | Native per-form conformance executable (C3n), `make machine.a64.conformance` | `machine_target_aarch64`, `unix`, gcc on an AArch64 host |
| `test/machine_alloc` | Allocation, checking, physical interpretation (C4), frames (C5), mutations, liveness | the above |
| `test/machine_x86_64` | x86-64 forms, negatives, C3/C4/C5 interpreted on any host, mutations | the above |

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
(the SSA C emitter's rule). What a table's silence means is stated, not
guessed (`Mir_failure.Unlisted`): by default a site-bearing failure with no
compatible entry is a refusal; a table declared complete — a bundle's Loop
table, which lists every site its lowering could not prove unreachable — gives
it the one-past-table sentinel, as `Ssa_c` does, and decoding the sentinel is
a defect, so a failure the declaration wrongly ruled out still cannot pass as
a row.

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
- **Freshness**: `undef` clears the initialization of exactly its view's
  window; the bytes stay, so a stale read is a defect rather than a value.
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
- **Storage and quantization** (the M4.2 slice): binary16 decodes through
  explicit bit manipulation to binary64; bfloat16 is the high half of a
  binary32 (`zext; shl 16; bitcast`); i32 sign-extends then `scvt`. Bool is
  computed in i32 — a decode zero-extends the byte and compares with zero, an
  encode selects `0`/`1` (NaN stores 1) and `itrunc`s to i8 before the byte
  store — because neither target keeps i8 arithmetic: on both, an i8 or i16
  value lives in a 32-bit register with only its low bits meaningful, which
  is all a byte store and an extension read. The `itrunc` is AArch64
  `uxtb`/`uxth` and x86-64 `movzx`, natively conformance-tested on AArch64.
  Dequantization is `scale × (q − zero_point)` with `q` sign-extended to
  i64: per-tensor parameters are constants; per-channel ones are an i64-word
  table per buffer (`Mir_layout_map.Params`: scale bits at `8c`, zero point
  after the scales), a read-only bound region indexed by the access's C
  coordinate. A flat access names no channel, and the SSA verifier already
  rejects a per-channel buffer through one, so the lowering's own refusal is
  only a backstop.
- **Locals and the meter** (the M4.3 slice): each `local.alloc` site owns a
  scratch region (`Mir_layout_map.Scratch`, eight bytes per binary64 cell,
  checked arithmetic, all sites together bounded by `local_limit`), and every
  run of the site begins with `undef view` — the generic op that makes a
  view's bytes undefined again — before `addr view` gives the local's
  pointer. Reusing one region for every run of a site is sound only because
  no earlier instance stays reachable: a local reaching a use through
  anything but its own allocation's result (a loop or join parameter) is
  refused, so no CFG lifetime or escape metadata is needed, and SSA handle
  uniqueness is never taken as a stack-reuse proof. A read of a cell this run
  never wrote is then the interpreter's `uninitialized` defect, as it is the
  oracle's. A read outside a named local's object guards `[0, slots)` (in
  i64) and fails `unbound_local`, its site bound to the first compatible
  table entry; outside an anonymous one, and any write outside, is the
  byte-range defect. `check_local` is the same guard; `check_scan` guards the
  row, then the lane, with payload `(row, lane, extent)`. The scan meter is a
  runtime region of two i64 words (updates remaining, live state), reset at
  entry when the program touches it and by `meter.reset`: a charge loads the
  remaining count, fails `scan_meter(updates_exhausted)` unless it is
  positive — before the body it guards — and stores it less one; a reserve
  fails `scan_meter(state_over_limit)` when the live state plus `2 × width`
  passes the limit; a release subtracts. Both are invocation storage
  re-established on entry or at the site, so a failure needs no cleanup.
  `Local` lowers to `ptr`.
- **Math functions** (the M4.4 slice): `cos`, `exp`, `log` and `sin` are
  calls to `Mir_math` helpers — versioned, pure, binary64, infallible — each
  bound to the C library symbol of its name, which a native realization
  declares and links. Their interpreter models (`Mir_math_model`) are the
  host's binary64 libm, which is the primitive the oracle and the C and Wasm
  backends use: agreement holds by construction, and another platform's libm
  is a reported disagreement, never a tolerance. A binary32 operand widens,
  calls, and rounds once, as the precision rewrite defines. `erf` is owned:
  no helper, but the reference's Abramowitz-Stegun formula expanded into
  primitives (an exact absolute value by masking the sign bit, `fdiv`, the
  Horner chain in the reference's association) with one `exp` call; at
  binary32 every step is a binary32 operation and `exp` runs between a
  widening and one rounding, matching `Ssa_numerics.erf32`. A program
  declares exactly the helpers it calls. A flat `check_access` lowers to
  nothing (it has no source failure). Vectors are the only refused family.

`Mir_ssa_rows` maps an SSA interpreter failure to the Machine IR row it must
equal; it is the oracle adapter, used by tests and model hosts.

`Mir_lower.Mutation` is fault injection for the evidence suite only
(channel zero, charge after the body, conversion guard order, double
rounding through binary64, eager load, the error function's last step
distributed or its binary32 form rounded once, guard order, operand order,
byte scaling, sequential transfer, stale local bytes, zero extension);
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

## Target interface and the selected stage

`Mir_target` (pure, in `machine_ir`) names features (closed across targets),
register banks, physical views (`bits` of a unit from bit `lo`: W0/X0 share a
unit, as do S0/D0/Q0), destination write rules (`merge` or `zero_upper`),
allocation constraints (fixed use/result, tied, early clobber), classes, the
ABI (argument/result views, preserved views with their preserved bit range,
reserved views, stack alignment) and the cited specification. Hardware,
assembler and interpreter capability stay separate properties.

`Mir_sel.TARGET` is what a target supplies: its closed opcode and branch-test
families, per-form features, uses, typing, order, constraints, result write
rule, implicit clobbers, whether it changes condition state, the condition bits
it defines and reads. `Mir_sel.Make` instantiates the selected container (the
generic `Mir_block`/`Mir_func`/`Mir_program` over `Op.t = Event | Machine of
target op | Undef` — the two target-neutral instructions emit no code; `undef`
stays through allocation, where it has no locations, so every stage's
interpreter still checks a local's freshness — and `Terminator.t = Branch of test | Jump | Return` — no `fail`
exists here), its verifier (the shared structure plus: features within the
program's set, constraint shapes, branch-test typing, and **local condition
state** — a `flags` value is used only in its own block with no
condition-changing instruction in between, reads only bits its producer
defines, and is never a block parameter) and printer. A verified selected
program exists only as `Verified.t`. `Mir_sel_interp.Make` runs one through
the target's semantic function (`exec`, `test`) with the generic dispatcher's
control and simultaneous edges; a tie constrains allocation only, so the
virtual input keeps its value. Reading an undefined condition bit is a defect.

## AArch64

Forms (DDI 0487 K.a): add/sub (register, imm12), mul, sdiv, msub, logical
(register, bitmask immediate — encodability checked), shifts by immediate,
mov/movz/movn/movk (movk tied), sxtw/uxtw/W truncation, cmp (register, imm12),
csel/cset, fp add/sub/mul/div/max, fneg/frintz/fsqrt, fcmp, fcsel, fcvt,
fcvtzs, scvtf, fmadd, fmov (register, to/from GPR), ldr/str (B, H, W, X, S, D
at an unsigned scaled offset), adrp + add :lo12: for a view's address, and
b.cond/cbz/cbnz tests. W arithmetic takes i32 only (a predicate or byte would
leave its canonical range); a predicate lives in a W register as 0/1.
Every form zeroes the rest of its destination unit. SUBS and FCMP define all of
NZCV (FCMP: 0011 unordered, 0110 equal, 1000 less, 0010 greater); nothing else
in the slice touches NZCV. Loads and stores permit unaligned normal-memory
access.

**Selection** (`A64_select`) keeps every generic value's id and type, and
blocks, parameters and order states unchanged; temporaries are fresh.
Constants are MOVZ/MOVN/MOVK sequences (floats via a GPR and FMOV); an address
is ADRP + ADD :lo12:; a predicate result is CMP/FCMP then CSET (ordered
less-than is `mi`, less-or-equal `ls`, unordered `vs`, unsigned `lo`/`ls`); a
select is CMP #0 then CSEL/FCSEL `ne`; remainder is SDIV then MSUB; a
predicate branch is CBNZ. `fail` becomes the model failure record written
into a runtime view (kind, invocation -1, the kind's words from
`Mir_failure.layout`, the site word bound through the supplied table — a
missing compatible entry is a refusal) and a return of status 1; a return gets
status 0. i8/i16 arithmetic is refused in this slice; calls are described
with allocation below. `A64_select.Mutation` is fault injection for evidence
(contraction, `lt` for ordered less-than, a missing record word, signed for
unsigned compare).

**Generic vs selected (C3)**: the selected run's status 0 means success with
outputs in memory; otherwise the stored record is decoded through the site
table and compared as a row. Source kernels, failure kernels, matmul and the
integer/conversion programs agree with both the generic route and the SSA
oracle.

**Native per-form conformance (C3n)**: `make machine.a64.conformance` builds a
generated inline-assembly harness with gcc and runs every admitted form
instance (immediates, conditions and sizes enumerated) on the host CPU: NZCV is
seeded before and read right after the instruction, the destination is seeded
with a pattern so the write rule is visible, memory forms use a canaried
buffer, FPCR is fixed to round-to-nearest-even with no flush-to-zero, no
default NaN and no traps, then restored, and FP/ASIMD support is probed with
`getauxval`. Operands are every pair of boundary values plus seeded random
ones. Results compare exactly (any NaN for a NaN — payloads are not modelled),
flags exactly, memory exactly; ADRP/ADD :lo12: are checked against a real
symbol. Each deliberately wrong semantic entry (carry, unfused fmadd, fmax of
signed zeros, unordered flags, W-write merge) must make the run fail. A
non-AArch64 host reports "unavailable", never a pass.

## x86-64

Selection runs through the shared builder (`Mir_select`: value identities,
temporaries, order remapping across splits, status propagation, record stores
from `Mir_failure.layout`, the record's runtime view), which AArch64 now uses
too. Forms (Intel SDM 325462-084): ALU ops, IMUL, NEG and shifts by immediate
tied to their first operand; CMP/TEST/BT/UCOMISD producing condition state
(TEST leaves AF undefined, BT defines CF only); CMOV tied to its else operand;
`setcc`+`movzbl` as one probed pair; CQO+IDIV with the dividend and both
results in rax/rdx and rdx early-clobbered against the divisor (IDIV's #DE is a
domain defect); MOV/MOVABS immediates; LEA (RIP-relative for views); loads and
stores with `[base + index*scale + disp32]` (selection folds a pointer plus an
index times 1, 2, 4 or 8); legacy-SSE scalar arithmetic, conversions and SQRT
merging the destination's upper bits; whole-register ANDP/ANDNP/ORP/XORP and
MOVAP leaving them untracked (`undefined_upper`); CMPSD/CMPSS equal and
unordered masks; ROUNDSD/SS only under SSE4.1 and VFMADD231SD/SS only under
FMA, else a typed refusal — never a rounded multiply and add. Compares:
ordered equality is `e` and `np` combined; ordered less-than and
less-or-equal compare reversed with `a`/`ae`; unordered is `p`. MAXSD returns
its source operand for a NaN or two zeros, so IEEE maximum is a sequence:
equal operands give their AND, a NaN operand their sum. A float select masks
through CMOV-chosen all-ones/zero moved to the XMM bank. System V calls take
rdi, rsi, rdx, rcx, r8, r9 and xmm0-7, return in rax/rdx and xmm0/1, and
clobber rax, rcx, rdx, rsi, rdi, r8-r11, every XMM register and the flags;
rbx, rbp and r12-r15 are preserved; rsp, rbp, r10 and r11 are reserved. A
call pushes an 8-byte return address (`call_push`): the frame plus the push
keeps rsp 16-byte aligned at calls; the physical interpreter pushes a token
and requires it intact at the callee's exit. MXCSR moves only through memory,
which the late-form contract does not admit, so x86-64 frames leave it to the
native entry wrapper. Native per-form conformance needs an x86-64 runner.

## Allocated stage, checking and reference allocation

**Form** (`Mir_phys`): SSA is gone. A location is a register view or a frame
slot (read and written whole, before layout). A block has no parameters: its
origin is the selected block it realizes or the selected edge a split block
carries, and its entry contract lists where each selected parameter arrives.
An executed instruction keeps its selected form — whose virtual operands are
correspondence metadata only — plus one location per use and per result;
moves carry the value they transfer as a witness. Terminators branch on
located operands, jump, or return values already in convention registers.

**Physical verifier** (`Mir_phys_verify`): each location fits its value (bank
and width; a condition only in the condition register; a slot exactly the
value's bytes), no reserved register is touched, operand counts match, fixed
operands are in their register, a tied result shares its use's unit, an
early-clobber result overlaps no use, instruction operands are registers.

**Physical interpreter** (`Mir_phys_interp`): register units with a validity
mask per bit; every register starts undefined; a result is written under its
form's write rule (zero-upper or merge); a declared clobber leaves its units
undefined; slots hold a value of the size written. It never consults virtual
names: operands resolve by position to their listed locations.

**Checker** (`Mir_checker`, in its own library with no allocator import): from
the selected program it takes what each use expects; from the allocated one
what each location holds, by symbolic dataflow over units and slots (a holder
is a value and the width it was written at). Moves copy a holder after
verifying the witness; an instruction checks its uses, forgets clobbered
units, the condition register when it changes condition state without a
condition result, and every older holder of the values it defines (a stale
loop-iteration copy), then records its results. Entering a block's
realization renames the carrying selected edge's arguments to the block's
parameters at the claimed locations. States meet by intersection to a
fixpoint, so a value unknown on any path is unknown at the join. Structure is
checked too: each selected block realized once, its instructions present in
order, terminators reaching the selected targets directly or through one
split block of that edge. A rejection is a compiler error; the artifact is
never run.

**Reference allocator** (`Mir_ref_alloc`): every value in its own slot;
operands reloaded into per-position scratch registers (or their fixed
register), the result written to a result scratch (or its fixed or tied
register, or the condition register) and spilled. Parameters live in slots, so
an edge is a simultaneous slot-to-slot transfer resolved by
`Mir_parallel_copy` — a move whose destination nothing pending still reads
goes first; a remaining cycle saves one destination to a cycle scratch; overlap
decides "still read" — placed at the end of a single-successor block or in a
split block. AArch64 draws scratch from x9-x14 and v16-v21 (`A64_regs`).
`Mir_ref_alloc.Mutation` injects allocation defects for evidence.

**Calls.** The fallible convention is a status return: every selected
function returns its results then an `i32` status (0 success); on failure the
callee has already stored its record and no result is defined. AArch64 `bl`
takes its arguments and results in fixed registers (AAPCS64 sequences:
x0-x7 with W views for 32-bit values, v0-v7 as S or D; several results in
the same sequence, an extension of the base convention) and clobbers every
bit AAPCS64 does not preserve: x0-x18, x30, v0-v7 and v16-v31 whole, the
upper 64 bits of v8-v15, NZCV. Selection follows a call that may fail
(a helper declaring failures, or a function that can reach `fail` or such a
call) with a branch on its status whose taken side returns that status, the
record untouched; the block splits there and later order states are remapped
to the continuation's. A call that may fail with results, or from a function
with results, is refused in this slice (neither result could be left
undefined). The selected interpreter runs a callee function in its own frame
and a helper by its model, storing a failing helper's record exactly as an
owned native helper must; the physical interpreter does the same over shared
registers with a frame of slots per activation, invalidates exactly the
clobbered bits after the call, and on every function exit compares each
callee-saved range that was defined on entry (a change is a
`preserved_state` defect). The checker kills a holder only when the clobbered
bits overlap the bits it was written with, so a D value in v8 survives a call
and the same value in v16 does not.

**Frame realization** (`Mir_frame`, target hooks in `Mir_sel.TARGET`,
`A64_frame` and `X64_frame`): slots become memory based on the stack pointer
(`Mem {base; offset; bytes}`). Layout gives each slot and save word an aligned
offset (an optional pad forces large offsets in tests); the frame size plus
any pushed return address keeps the stack 16-byte aligned, and each target
decides which stack-pointer steps encode and keep its own alignment (AArch64
multiples of 16, as an access through a misaligned SP faults; x86-64 whole
words). A function's frame size is `None` until realization. The prologue moves the stack pointer in encodable
steps (a 4 KiB-multiple part and the rest; a frame needing more is refused as
beyond the code model), saves the link register when the function calls,
every callee-saved register it writes, and — in the entry function — the
caller's FPCR, which it then zeroes (round to nearest even, no flush to zero,
no default NaN, no traps); an epilogue before every return, failure exits
included, restores all of it. An offset no access encodes is reached through
reserved x16: a late `add` computes the address and the access is based on
x16; FPCR travels through reserved x17. Late forms touch only reserved
scratch, the stack pointer and control registers; saves move preserved, link
or reserved registers whole. The physical verifier checks frame accesses
(based on SP or reserved scratch, inside the frame, encodable), step
encodability and alignment, and save widths. The checker resolves frame memory
through address holders (`entry SP + k`, tracked through `Sp` steps and late
address arithmetic), kills overlapping frame bytes on writes, and otherwise
treats realized programs like abstract ones. The physical interpreter, with
`~realized:true`, runs on a stack region: an activation may touch only its
own frame, a spilled pointer keeps its provenance while its bytes are
untouched, a call needs an aligned stack, an exit needs the entry stack
pointer back and the link register's return address intact, preserved state
and FPCR are compared on every exit, and an FP form under any FPCR but the
modelled zero is unsupported. Bounding the window per activation stands in
for canaries: any access outside it is a defect at once.

**Liveness** (`Mir_liveness`): live-in/out to a fixpoint (block parameters
define, edge arguments use at the source's end, order excluded) and linear-scan
intervals over reverse postorder with two positions per instruction, lifetime
holes, and loop-header extension. A value's range covers each read; coverage
where a value is dead is accepted imprecision from the loop rule.

## Host numerics

OCaml's arm64 backend contracts `x *. y +. z` into one fused instruction. Any
host computation that must round a product before a sum (an unfused oracle, a
two-step reference) hides the product behind `Sys.opaque_identity`;
`Ssa_scalar`'s unfused `Float_fma` path does, and so does the reference error
function (`Expr`'s Abramowitz-Stegun form), which on AArch64 had compiled to
five fused instructions and differed from the as-written formula — the one
the C (`-ffp-contract=off`), Wasm and JavaScript evaluations compute — by an
ulp at 0.5. Still fused on AArch64 and outside this design's scope: the Loop
reference interpreter's unfused binary64 `Fma` branch and the `arange`
factory's `start + i * step`.
