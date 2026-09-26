# Loop IR JavaScript backend: an AST, typed builders, a printer, and a JS-only executor

Status: **implemented**. Builds on the finished Phase 2 of the kernel-DSL design
(the Kernel DSL design doc, sections "Loop IR", "JavaScript and C semantics",
"Phase 2"); the JS-side value representations chosen there are unchanged. JS
backend layout and mirrors: `js_backends_design.md`.

## Outcome

Everything below shipped, with these decisions and deviations:

- **D1 to D5** as proposed: `lib/js_ast` (D1), `new Function` memoised by printed
  source (D2), `%.17g` literals with no backend difference found (D4), and no
  `loop_ir_js` mirror until an executable needs one (D5). **D3** was decided by
  evidence: the signed-zero disagreement was live (see "Signed zero"), so
  `Num.of_idx` prints `i + 0`.
- The library is `(wrapped false)` with `js_` module prefixes, and the checker is
  `Js_check` (`Fault.t`); see "As built" under `Js_ast`.
- The executor lives at `js/loop_js_exec`, beside `js/jsoo` rather than under it
  (its closure holds the ordinary `native`).
- `Idx.scale k` of a constant `0` is folded to `0`, beyond the two folds first proposed:
  the W-only offset test needs it.
- The failure table gained `Site` and per-node index overflow records, because the
  interpreter's row could not otherwise be rebuilt from JavaScript.
- The Melange probe found a real difference on its first run (`Float.sign_bit (-0.)`
  is false there); the printer reads the sign of zero from `1 / x`.
- The op sweep and the `Loop_check` fixtures never reach a NaN in a pool comparison,
  so a dedicated test guards `pool_better` in-process.
- Emitted size fell 9% over the 588 op-sweep programs (826,402 to 750,916 bytes).

## Why

`Loop_js.emit` works, and the node gate proves it against the Loop interpreter.
But it goes straight from `Loop_program.t` to text with `Fmt.str`. Every
rule that matters to correctness is enforced only by the author's care at each
format string:

- **Mixing `Number` and `BigInt`.** `1n + 1` throws a `TypeError` at run time.
  Nothing in the emitter stops an int64 subexpression from landing in a
  `Number` context. The int64 carrier is the one place that needs two numeric
  representations, and it is exactly where text concatenation gives no help.
- **Precedence.** Safety comes from putting parentheses around everything.
  That is why the output reads `b1[(((((0 * 1 + 0) * 1 + 0) ...`. Any format
  string that forgets the parentheses changes meaning silently.
- **Helper selection by substring.** The prelude includes a helper when the
  body text contains `name ^ "("`. `"bf16_to_float("` contains
  `"f16_to_float("`, so any program that decodes bf16 also carries the f16
  decoder (`loop_js.ml`, `contains`). That is harmless today. It shows the
  mechanism cannot tell a call from a substring.
- **Undeclared names.** A temporary missed by `declarations` becomes an
  implicit global in sloppy mode. Nothing checks that every identifier the
  body uses is bound.
- **Runtime helpers are opaque strings.** Nothing structural knows which
  helper depends on which (`bf16_to_float` needs `bf16_bits`/`bf16_view`).

Moving to an AST puts each of these rules in one place: a constructor, a type,
or a checker. The printer is then the only code that produces JavaScript text.
It is also the prerequisite for a C printer (Phase 5 of the kernel-DSL design):
that one prints its own AST, but the split is the same.

## The split: what is neutral, what is JS-only

| Layer | Library | Backends | Depends on |
|---|---|---|---|
| JS AST, identifiers, printer, scope checker | `js_ast` (`lib/js_ast`) | all (native, jsoo, melange via `js_ast_mel`) | `fmt` |
| Typed builders over the AST | `js_ast` (`Js_build`) | all | — |
| Loop IR → JS AST lowering, runtime helpers as AST | `loop_ir` (`Loop_js`, `Loop_js_runtime`) | native + jsoo | `js_ast`, `native`, `expr` |
| **Executing generated code** in-process | `loop_js_exec` (`js/loop_js_exec`) | **jsoo only** | `js_of_ocaml`, `loop_ir`, `native` |
| Generated-code gate (file + `node`) | `test/loop_ir` (unchanged) | native generator, node runs output | — |

Only execution is tied to a backend. Building, printing, checking and goldening
the AST all run natively under `make runtest`, and under
`make jsoo.inline-runtest` as `loop_ir` already does. `js_ast` needs no
Bigarray and no submodule, so it is inside Melange's reach as well. That costs
nothing, and it keeps the AST honest about staying a plain data library.

Why not put the AST in `loop_ir`? The AST is not about loops. A second
producer, such as a C-free WebGPU/WGSL host shim or a model-explorer
visualiser, should not have to link `native` to get a JS printer. `lib/js_ast`
joins CLAUDE.md's JS-reachable list from its first commit, because its Melange
mirror links it into a JS build right away (see "Melange"). It therefore sits
under the integer rule: see "Literals".

## `Js_ast`: a closed subset of ECMAScript

The AST covers what the emitter and the helpers need, and nothing more. A
construct not in the AST cannot be emitted. That is the point of the AST.

```ocaml
module Ident : sig
  type t = private string
  val v : string -> t
  (* [A-Za-z_$][A-Za-z0-9_$]*, not a reserved word, not a known global
     (Math, BigInt, Number, NaN, Infinity, the typed-array constructors):
     raises Invalid_argument otherwise. A name is a programming constant,
     never data, so a bad one is a defect, not an Err. *)
end

type unop = Neg | Not | Plus
type binop =   (* alphabetical; precedence lives in the printer's table *)
  | Add | And | Div | Eq_strict | Ge | Gt | Le | Lt | Mul | Ne_strict | Or | Sub

type expr =
  | Array of expr list
  | Bigint of int64               (* printed [123n]; negative as [-123n] via Neg *)
  | Binary of binop * expr * expr
  | Bool of bool
  | Call of expr * expr list
  | Cond of expr * expr * expr
  | Global of Global.t            (* Math, BigInt, Number, typed-array ctors: closed variant *)
  | Index of expr * expr          (* a[i] *)
  | Member of expr * Ident.t      (* Math.floor *)
  | New of expr * expr list
  | Null
  | Number of float               (* round-trip literal; NaN/Infinity spelled by the printer *)
  | Object of (Ident.t * expr) list
  | String of string              (* escaped by the printer *)
  | Unary of unop * expr
  | Var of Ident.t

type lvalue = Lvar of Ident.t | Lindex of expr * expr

type stmt =
  | Assign of lvalue * assign_op * expr     (* =, +=, -= ; assignment is never an expression *)
  | Const of Ident.t * expr
  | Expr of expr
  | For of { var : Ident.t; init : expr; test : expr; body : stmt list }  (* update is [var++] *)
  | Function of Func.t
  | If of expr * stmt list * stmt list
  | Let of Ident.t * expr
  | Return of expr option

module Func : sig type t = { name : Ident.t; params : Ident.t list; body : stmt list } end
module Program : sig type t = { prelude : stmt list; entry : Func.t } end
```

Deliberate omissions: no `var`, no `while`, no closures and no arrow functions
(the kernel-DSL design already says: no closure per element), no `this`, no
`eval`, no `==`, no bitwise operators, no `**`, no `++` outside a `For`
header, no labels and no `break`. Bitwise operators are left out of the AST on
purpose. `x | 0` on an index is the silent-wrap class the integer rule forbids,
and taking the constructor away does more than any review. The f16/bf16
helpers do need `>>`, `&`, `|` and `<<` on 16-bit patterns. They get
`Bit_and | Bit_or | Shl | Shr`, and only `Js_build.Bits` (below) produces
them, typed over a `bits16` kind that no index can be passed as.

`For` is the counted `for (let v = init; test; v++)` form, not general
ECMAScript `for`. That form is the only one the Loop IR produces, and keeping
it narrow keeps the scope rule simple: `v` is bound in `test` and `body`.

**As built.** `lib/js_ast` is `(wrapped false)` with a `js_` prefix on every module
(`Js_ast`, `Js_ident`, `Js_global`, `Js_print`, `Js_check`, later `Js_build`), so the
Melange mirror refers to `Js_ast` unqualified like the other mirrored libraries.
`Ident` and `Global` are re-exported from `Js_ast`. `Stmt` and `Func` are one
`module rec` (a function is a statement and holds statements), so a statement is
`Js_ast.Stmt.t`, and `Js_ast.stmt` is an alias. The checker reports
`Js_check.Fault.t` (`Duplicate | Unbound`).

### Printer (`Js_print`)

- **Precedence-driven parentheses.** There is one table from `binop`/`unop` to
  ECMAScript precedence and associativity, and a child gets parentheses only
  when the table requires them. Special cases:
  - a negative `Number` or `Bigint` literal is a `Unary Neg`. It is
    parenthesised as the left operand of `**`, which is absent anyway, and on
    the left of `.` or `[`.
  - `a - -b` and `a + +b` get a space (`- -`), never `--`.
  - `Member`/`Index` on a numeric literal parenthesises the literal
    (`(1).toString()`).
  - an `Object` at the start of an expression statement is parenthesised.
    `return {…}` needs nothing.
  - mixing `||` with `??` is not reachable, because `??` is not in the AST.
- **Sign of zero.** `-0` is detected from `1. /. x`, not `Float.sign_bit`: the
  Melange probe diff caught `Float.sign_bit (-0.)` reading false there.
- **Literals.** `Number x` uses the round-trip rule `Loop_js.float_literal`
  has today (`%.17g`, `NaN`, `Infinity`, `-0` as `Unary (Neg, Number 0.)`).
  A `Number` built from an OCaml `int` goes through `float_of_int`, which is
  exact for every value a 32-bit jsoo `int` can hold. The printer never
  formats an OCaml `int` directly, so whether it runs natively or under jsoo
  cannot change the text. `Bigint n` prints `Int64.to_string` plus `n`.
- **Layout.** Two-space indentation, one statement per line, no line
  wrapping. `Format` boxes are not used: they make the text depend on a
  margin setting, and emitting twice must be byte-identical. The printer
  writes to a `Buffer.t`.
- **Directive.** `Program.t` prints with `"use strict";` first. An assignment
  to an undeclared name is then a `ReferenceError`, not a new global.
- **Two renderings of one `Program.t`:**
  `script` produces the prelude, then `function entry(...) {…}`, which is what
  the node gate writes to `run.js`. `factory_body` produces the prelude, the
  function, then `return entry;`, which is what `new Function` receives (see
  the executor).

### Scope checker (`Js_ast.Check`)

`closed : Program.t -> (unit, Unbound.t list) result` walks the program with
lexical scopes (function parameters, `let`/`const` at their block, the `For`
variable, function names at the program level). It reports every `Var` that
is not bound and every duplicate declaration in one block. `Global.t` is
closed, so a global cannot be misspelled into a free variable. `Loop_js.to_ast`
runs `closed` before it returns, and a failure there is a defect in the
lowering (`invalid_arg`). This replaces the hope that `declarations` found
every temporary.

`free : stmt list -> Ident.Set.t` is the same walk. The prelude is chosen from
it (below).

## `Js_build`: typed helpers over the untyped AST

The AST itself is untyped so the printer can stay simple. The helpers the
lowering actually calls carry a phantom kind, so mixing representations is an
OCaml type error:

```ocaml
type num  (* JS Number: binary64 working value *)
type idx  (* JS Number holding an index proven in [-2^31, 2^31) *)
type big  (* JS BigInt: the int64 carrier *)
type bool_
type bits16
type 'k arr  (* a typed array whose element reads as 'k *)

type 'k t = private Js_ast.expr

module Num : sig  (* float working values *)
  val const : float -> num t
  val ( + ) : num t -> num t -> num t   (* ... *)
  val fround : num t -> num t
  val of_idx : idx t -> num t           (* see "Signed zero" below *)
  val of_big : big t -> num t           (* Number(b): rounds to nearest, as Int64.to_float *)
  val lt : num t -> num t -> bool_ t
end
module Idx : sig
  val const : int -> idx t
  val add : idx t -> idx t -> idx t
  val scale : int -> idx t -> idx t
  val floor_div_pos : idx t -> int -> idx t   (* Math.floor(a / d) *)
  val ceil_div_pos : idx t -> int -> idx t
  val of_big_bounded : big t -> idx t         (* Number(b): only after a range check; see below *)
end
module Big : sig
  val const : int64 -> big t
  val add_wrap : big t -> big t -> big t      (* BigInt.asIntN(64, a + b) *)
  val sub_wrap : big t -> big t -> big t
  val mul_wrap : big t -> big t -> big t
  val div_unchecked : big t -> big t -> big t (* the caller emits the zero / min/-1 guards first *)
  val of_idx : idx t -> big t
  val of_num_trunc : num t -> big t           (* BigInt(Math.trunc(x)): after its range check *)
end
val load : 'k arr t -> idx t -> 'k t
val store : 'k arr t -> idx t -> 'k t -> Js_ast.stmt
```

What the types buy:

- `Big.add_wrap` is the only way to add two `big` values, so a missing
  `asIntN` is not a possible state. The Phase 2 mutation "no wrap on +" could
  not be written.
- Only `Idx` values can go in an array subscript, and no `Idx` operation
  prints a bitwise operator.
- `Idx.of_big_bounded` is the one crossing from `BigInt` to index. It is
  named for its precondition, and its only caller is the gather lowering,
  right after the range check. This is "narrow only after bounding" in the
  generated code's own vocabulary.

**Folding at the builder, exactness-preserving only.** `Idx.add` drops an
`Idx.const 0` and `Idx.scale 1 a` is `a`, and `Idx.scale k` of a constant `0` is `0` (a constant
carries no guard). `Idx.scale 0 a` is *not* folded to
`0`: `a` may carry an overflow guard's subexpression the program still
checks. That turns `(((((0 * 1 + 0) * 1 + 0) ...` into `b1[i0]` without
changing a value, because index `Number`s are exact integers in the proven
domain. `Num` does no folding at all. `x + 0` and `x * 1` are not identities
on binary64 (`-0 + 0 = +0`), and the interpreter is the oracle.

### Signed zero at the index-to-float crossing

In JavaScript an index `Number` can be `-0`, where OCaml's `int` cannot:
`Math.ceil(-3 / 4)` is `-0`, and `-1 * 0` is `-0`. As an array subscript, `-0` is
harmless. But `Value_of_index` was emitted as the index expression itself, so a
`-0` index that became a float value was `-0.` where the interpreter produces
`+0.`, and the harness compares bitwise.

**Verdict: live defect, fixed.** Two hand-built gate cases (`Value_of_index` of
`Ceil_div_pos (Add (Var, Const -3), 4)` and of `Scale (-1, Var)`, at `Var = 0`)
disagreed with the interpreter on the string emitter. No lowered native kernel
was found that builds either shape (`Value_of_index` producers are `arange`,
pooling window areas, resize remainders and LSTM lane numbers), but the Expr
language accepts both, so the fix sits at the crossing: `Num.of_idx` prints
`i + 0`, which maps `-0` to `+0` and leaves every other integer unchanged. It is
the one function every index-to-float crossing goes through, and it is built with
`Num.add`, so the "`Num` folds nothing" rule is what keeps it.

## Runtime helpers as AST (`Loop_js_runtime`)

Each helper becomes a `Js_ast.stmt list` built with `Js_build`. The comment
naming its OCaml counterpart stays in the OCaml source. Each helper carries a
`Helper.t` record:

```ocaml
module Helper : sig
  type t = { name : Name.t; defines : Ident.t list; body : Js_ast.stmt list }
end
type Name.t = Bf16_to_float | Coord_failure | Erf | F16_to_float | Float_max
            | I64_from_float_failure | Pool_better   (* closed; alphabetical *)
```

As built, `Loop_js_runtime` also exposes a typed call per helper (`float_max :
num t -> num t -> num t`, ...), the one place each helper's kinds are stated, and
`Helper.defines` lists every top-level name the body declares (`bf16_to_float`
defines `bf16_bits` and `bf16_view` too).

Prelude selection is a fixpoint over `Check.free`. Start from the entry
function's free identifiers, add every helper that `defines` one of them, and
repeat with the added helpers' own free identifiers. Emit the chosen helpers
in `Name.t` order, the same as `Loop_js_runtime.helpers` order today. This
fixes the bf16/f16 over-inclusion, and a `bf16_bits` shared array comes in
because the function reads it, not because a string happened to match.

The helpers are still tested one by one against their OCaml counterparts,
exactly as today (the 65536-pattern f16/bf16 tests, the D4 tolerance runs). A
transcription moving from a `{js|…|js}` string into builder calls is a
rewrite, so every one of those tests must run red-then-green once. The
mutation for each is: the bf16 shift, the f16
subnormal scale, `pool_better` made strict, `float_max` as `>`.

## Loop IR → AST (`Loop_js`)

`Loop_js.to_ast : Loop_program.t -> Js_ast.Program.t` replaces the string
functions one for one: `index` goes to `Idx`, `expr` to `Num`/`Big` by the
GADT's carrier (`'a Loop_expr.t`'s type index picks the builder module), and
`pred` to `bool_ t`. Naming stays as it is now (first appearance, `i`/`x`/`a`/
`b` prefixes), and names are now `Ident.t`. The helper names and these
prefixes are disjoint by construction, and `Ident.v` enforces it.

Every multi-operand node names its operands left to right with explicit `let`s:
OCaml evaluates call arguments in an unspecified order, and a name must not
depend on the compiler's pick.

`emit p = Js_print.script (to_ast p)` keeps its signature, so `loop_js_gen`
and the node gate are unchanged.

**Failure records get one encoding table.** `Loop_js_failure` holds a closed
`Kind.t` (the `kind:` strings, alphabetical), a closed `Field.t` (the record
keys), and `fields : Kind.t -> Field.t list`, the keys a record of that kind
writes. `record` refuses a field list that is not exactly the table's, so a
field cannot be forgotten or invented at one call site. The emitter and the
runtime helpers build every failure `Object` through it, and the executor
decodes with the same table. The closed string-valued fields are variants too
(`Overflow_op`, `Projection`, `Meter`).

Two failures carry more than the interpreter's row could be rebuilt from a
`kind` alone, so the table records how each is decoded:

- **`Site`.** A `Fail_if`'s ordinal in program order (`Loop_js_failure.sites`
  walks statements in order, a `For` body, then an `If`'s branches). It names the
  static part of a failure that has no printed form: which `Expr.Local_var.t` an
  `unbound_local` or a `scan_projection` names. The emitter numbers the sites it
  writes in the same walk and checks each against `sites` by physical equality,
  so the two cannot drift. Decoding reads the `Loop_failure.t` at the site.
- **`Index_overflow`.** The interpreter reports the first node, in post-order,
  whose value leaves the index domain, with its operator and operands. The
  emitter therefore writes one `if` per `Add`/`Scale` node in the same
  post-order, each returning `{ kind, op, lhs, rhs }`, instead of one disjunction.
  The disjunction still exists as the predicate `Index_overflows` when it is not
  the direct guard of an `Index_overflow` failure (a lowering defect otherwise).

A scan meter's limit is an int64 the JavaScript meter counts down in a `Number`,
so `to_ast` refuses one that is not exact there (beyond 2^53) rather than round
it.

## The executor: running generated code in-process (jsoo only)

`js/loop_js_exec`, library `loop_js_exec`, depending on `js_of_ocaml`. It sits beside `js/jsoo`, not under it: its closure holds the ordinary `native` (through `loop_ir`), and `js/jsoo`'s documented closure check expects only the mirrored `_js` names.
It is only linked by `(modes js)` executables and by an
`(inline_tests (modes js))` suite. It links natively as a library but not
into a native executable (the `caml_js_*` primitives do not exist there), so
the build graph enforces the restriction.

```ocaml
type compiled
val compile : Loop_program.t -> (compiled, [> `Js_compile of string ]) Err.t
val run :
  compiled -> bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, [> Loop_interp.error | `Js_exception of string ]) Err.t
```

`run` has `Loop_interp.run`'s shape on purpose. `Loop_check` gains a third
executor under jsoo, and the harness's existing verdict (bitwise values, kind
and payload on failures, NaN compared as NaN) applies unchanged. That is the
in-process alternative Phase 2 considered for running generated code and did not pick,
and it now complements the file-based gate rather than replacing it:

| | node gate (`make loop.js.runtest`) | in-process executor |
|---|---|---|
| Runs | hand-built and lowered cases written to `run.js` | any `Loop_check` fixture, the op sweep |
| Oracle | interpreter's text rendering | interpreter's `Tensor.packed` results, via the harness |
| Failures compared | as printed | decoded to `Expr.Eval` rows, kind **and** payload |
| Needs | node on `PATH` | a jsoo build (`make jsoo.inline-runtest`) |

**As built.** `exec = compile` then `run`, with the same shape as
`Loop_interp.run`. Beside the two above, the interface exposes `compile_source`
and `compile_as` (compile a supplied factory body, so a test can hand it source
the emitter would never write: a `SyntaxError`, a kernel that throws) and
`int64_view` (the alias, so its round trip is tested against the Bigarray
directly). `Loop_check.run` takes an optional `Executor.t`, defaulting to the one
`Loop_check.install` last set; native code never installs one.

The jsoo test library beside it copies the `Loop_check` fixtures and the op sweep
from `test/loop_ir` (copied, never forked) and installs the executor once, so every
`Loop_check.run` in them is also checked against generated code. Inline tests run
as their module initialises, in link order, so each copied suite begins by naming
the installer; without that a suite could run before the install and pass without
ever having met generated code. A kernel that leaves `[loop_js_exec_test]` red
under a deliberately broken emitter (`Mul` printed as `+`) is what proves the wiring.

Mechanics:

- **Compile.** `new Function(Js_print.factory_body ast)()` returns the kernel
  function. A `SyntaxError` is imported as `` `Js_compile ``
  (`Err.import`: the engine detected it, not us), and it means the printer is
  wrong. Browsers whose CSP forbids `unsafe-eval` refuse here with the same
  row. That is a deployment property, recorded rather than worked around. A
  Blob-URL module would need `script-src blob:` and an async API. `compiled`
  memoises by the printed source, which is deterministic, so identical
  programs share one function.
- **Binding without copies.** jsoo backs a Bigarray with a typed array. The
  external `caml_ba_to_typed_array` returns that array for any kind, so an
  input tensor's `Payload.data` is passed as is, and an output tensor is
  allocated in OCaml and written in place by the kernel. The executor checks
  each argument against `Loop_js.typed_array` (`instanceof` the named
  constructor), so a layout disagreement becomes a typed
  `` `Binding_mismatch ``, not a wrong read.
- **int64 buffers.** jsoo stores an `int64` Bigarray as an `Int32Array` of
  `[lo, hi]` pairs (`bigarray.js`, kind 7). On a little-endian host,
  `new BigInt64Array(a.buffer, a.byteOffset, a.length / 2)` aliases it
  exactly. Endianness is checked once at load
  (`new Uint8Array(new Uint16Array([1]).buffer)[0] === 1`), and a big-endian
  host is refused, not byte-swapped. This relies on a jsoo runtime detail.
  The executor's first test writes `min_int`, `-1`, `2^53+1` through
  `Bigarray` and reads them back through the alias, so a jsoo change turns it
  red.
- **Failures.** A non-`null` return is decoded with `Loop_js_failure` into the
  interpreter's row, e.g. `coord_out_of_range` into `Expr.Eval`'s
  out-of-range row with buffer id, axis, index and coordinate. A host
  exception escaping the kernel (`TypeError` from a `BigInt`/`Number` mix,
  `RangeError`) is `` `Js_exception ``. The emitter's contract is that this
  never happens, so the harness treats it as a defect verdict, not as a
  failure kind that can match.
- **Mirrors.** `loop_ir` depends on ordinary `native`, which is correct for
  the inline suite. A jsoo *executable* that also links the `native_js`
  mirror (the tail-call evaluator) needs a `loop_ir_js` mirror by the
  existing `copy_files` + `expr.ml` shim pattern. That mirror is added only
  when such an executable exists, not speculatively.
  **Landed**: the whole-model verification work below is that executable.
  `js/jsoo/loop_ir_js` follows the pattern exactly, with one addition the
  original note didn't anticipate: `lib/loop_ir` is a WRAPPED library, but
  `js/loop_js_exec/loop_js_exec.ml` (mirrored alongside it as
  `js/jsoo/loop_js_exec_js`) has a literal `open Loop_ir` that must keep
  resolving unmodified, and a dune wrapped library's alias module name has
  no override independent of its own `name` — so the mirror is UNWRAPPED
  and a checked-in `loop_ir.ml` recreates the `Loop_ir.*` namespace by hand
  as plain module aliases (mechanical, same spirit as `expr.ml`'s one-line
  alias, not a fork of any real logic).

### Whole-model verification (2026-09-22)

A real downloaded model's Region-authored nodes now run through
Loop_js_exec-compiled JavaScript, in-process, under node, checked against
the release's own top-5 rankings — not just the op sweep's synthetic
kernels this backend's own gate already covered. `fastvit_sa12` (SDPA,
1218 raw / 332 native-graph nodes) is the target: at the time this was
written, believed to be the only model in `PT2_MODELS_CRAM` with a
Region-authored op at all (**corrected below, "Every node..." section**:
`test_convnext2` has 9, almost certainly LayerNorm) — a pure-CNN model
(`mobilenetv2_050`, this backend's own `JS_PT2_MODEL`) has zero
Region-authored nodes and would exercise nothing new. Its two SDPA nodes
(`[H=16 W=49 C=32]`, a real transformer head count/sequence length/head
dim, not a toy extent) both take the generated-JS path.

Design: `lib/native/eval_direct.ml`'s `region_result`/`region_group_result`
gained a pluggable `Region_executor.t` seam (`lib/native/region_executor.mli`),
defaulted to `Region_execution.materialize`/`materialize_group` themselves
(behaviorally, not syntactically, identical — see below).
`lib/loop_ir/loop_region_program.ml` wraps the SAME `Region_program.t` the
reference path already built in a minimal sibling `Kernel.t` (no
re-derivation from the originating op) and runs it through the unmodified
`Loop_lower`. `js/jsoo/loop_js_exec_js/loop_region_executor.ml` compiles
and runs that through `Loop_js_exec`, falling back to
`Region_executor.default` (logged, not silently swallowed) on any refusal,
with a `Coverage.t` counter distinguishing generated-JS calls from
fallback ones and a `Coverage.check` assertion that fails loudly — with a
coverage message, not a ranking mismatch — if a run silently took only the
fallback path.

One correction made along the way, worth recording since it revises the
seam's own first-committed shape: `Region_execution.lowered` (opaque) and
`Expr.Eval.Env.t` (a deliberately narrow per-coordinate scalar loader,
`lib/expr` has no `Tensor`/`Tensor_sig` dependency at all) together carry
no per-source `Tensor_sig.t`, which `Loop_region_program.lower` needs to
build a `Kernel.Input.t`. `Region_executor.t` gained `~bindings:
Tensor.packed Tensor_id.Map.t` (the reference path's own merged operand +
synthetic-default map, real tensors) to supply it — `Region_executor.default`
is therefore a thin ignore-`~bindings` wrapper around `materialize`
rather than being it directly, though still behaviorally identical
(reconfirmed by a mutation test both before and after the widening).

Cost, measured directly rather than estimated: one `fastvit_sa12` sample
through the pure evaluator is ~103s native, and `jsoo.pt2.run`'s own
~4.9x node multiplier puts one sample at roughly 8 minutes under node —
`results.json` holds ten samples, so even the single-sample verification
this landed with is already past what `jsoo.pt2.run` itself (ten samples
of the much smaller `mobilenetv2_050`, ~7.5 min total) treats as too
expensive for anything but a MANUAL target. `make loop_js.pt2.run`
(Makefile) occupies that same tier — not a step in the `jsoo` GitHub job.
Full run against the real archive: PASSED, exit 0, ~7m46s, `--strict`
ranking match confirmed, coverage confirmed no fallback taken.

Pointer: kernel-DSL design doc's Phase 2 section — this is downstream of
that work, not a revision to it.

### Every node through its own generated-JS kernel (2026-09-22)

The whole-model verification above covers only the Region-authored arm
(RmsNorm/LayerNorm/Softmax/Sdpa/Lstm). Every other computing op kind still
ran through `Eval_direct`'s own OCaml formula even under jsoo — 0 of
`mobilenetv2_050`'s nodes and 2 of `fastvit_sa12`'s 712 took the
generated-JS path. This closes that gap: **parity with the interpreter**
— every node `Eval_direct` evaluates also runs through its own
`Loop_js_exec`-compiled JavaScript kernel, bitwise equal to the direct
result, with `pending`/`fallback` asserted zero at closure. It is **not
fusion** — each kernel covers exactly one node's output, no cross-node
scheduling or whole-graph `Kernel.t`.

**The seam.** `Node_executor.t` (`lib/native/node_executor.mli`) mirrors
`Region_executor.t`'s own convention exactly: a pluggable record field
`Eval_direct`'s non-Region compute arms call instead of computing
directly, universally quantified over the error row (`Eval_direct` sits
above this seam and cannot name its own error type here), defaulting to
`direct ()` so an absent `?node_executor` reproduces the exact prior
behavior. `lib/native/eval_direct.ml` was split (`admit`, the Bool/mixed-
dtype checks, staying; `compute`, every op's formula, moving to new
`eval_direct_compute.ml`) to make room for the seam without the file
crossing its line cap.

**Per-node compilation.** `lib/loop_ir/loop_node_program.ml` adapts ONE
output ordinal of ONE node to a `Loop_program.t`: `Eval_symbolic
.node_program` for that node, `Kernel_adapt.of_stage_program
~select:{oid} ~outputs:[oid]` (both narrowed to `oid` alone — a multi-
output node's OTHER outputs are not this lowering's concern, and
`~select` absent would pull every sibling into the "required" list,
failing the single-element `~outputs` check), `Fusion_plan.default`
(nothing to fuse for one value), `Loop_lower.lower`. `js/jsoo
/loop_js_exec_js/loop_node_executor.ml` builds the JS-only `Node_executor
.t`: a compile table keyed by output `Tensor_id.t` (unique only within
one executor instance — sharing a table across graphs answered for an
unrelated node whose id collided, a real bug T4.5's own sweep found and
fixed), a milestone predicate `scope` deciding which nodes route today,
`precompile` (shape-only, before any weight loads), `Coverage.t`
(`generated_js`/`fallback`/`pending`, per op kind) and `check_parity
~allow`, and shadow mode (run both paths, compare bitwise, report and
keep the trusted result on disagreement).

**`scope` widens in place, never gains a case per milestone.** It started
as "the default float-pixel arm only" and grew, by format and by op, as
each family landed: the two I64→F32 promotion arms (`Mul_scalar`/
`To_copy Float` reading an I64 operand through an explicit checked cast),
the Bool-storage arms (`Nonzero_bool`, already a Pixel-stage boundary
conversion), `Zeros`/`Eye` at their walked format, and — the one
structural extension — genuine I64-declared outputs (I64 `Reshape`/
`Permute`/`Add`/`Sub`/`Mul`, the index output of `Max_dim`/
`Max_pool2d_with_indices`, `To_copy Long`, `Arange`'s exact form,
`Unbind`/`Split_with_sizes`'s own I64 arm). By closure `scope` excludes
nothing by op kind at all — only `Unbind`/`Split_with_sizes` (a config-
dependent output count) and `Zeros`/`Eye` outside their walked format stay
format-gated, and even those route once walked.

**A `Kernel.t` could not originally name an int64 output at all**
(`Kernel.create`'s `Output.t`-resolving fold only searched `values`,
never `values_i64`) — a real, load-bearing gap discovered mid-session, not
a documented restriction: `Kernel_eval`/`Loop_lower` already treated
every `values_i64` entry as an unconditional materialization root/output
buffer, so only the PUBLIC naming path was blind to it. The fix is one
fallback branch in that fold. It immediately unblocked every I64-output
op kind's actual ROUTING (walk coverage for these had already landed
separately) and surfaced a second, narrower bug: `Kernel_adapt` copies
`values_i64` unconditionally of `~select` (by design — an int64 value is
always materialized), so a value-ordinal-only lowering of a two-output
node dragged an unrelated sibling's own i64 stage along as a second
output buffer. Fixed by pruning `Stage_program.stages_i64`/`outputs` to
what the target `oid` transitively needs before adapting
(`Loop_node_program.reachable_i64`).

**`Unbind`/`Split_with_sizes` needed their own I64 arm in `Eval_symbolic`**,
not a `Loop_node_program` change: neither op had one, so an I64 source
silently round-tripped through the default arm's `S.load` — lossy above
2^53, a real violation of `Graph_builder`'s own "slices retain the input
format" contract. `Split.Unbind.Compute_i64`/`Split.Split_with_sizes
.Compute_i64` (same shape as `Reshape.Compute_i64`: identical coordinate
math, `T.i64_load` instead of `S.load`) plus matching `process_node`
dispatch arms — folds over every output, since both ops are multi-output
— closed it. `check_parity ~allow:[]` then passed for every walked
subject: Definition of Done #3.

**The Lstm group executor** was the one remaining Region-authored gap:
`Region_executor.t`/`.group`'s TYPES and native `default`/`default_group`
already existed and `Eval_direct` already threaded `?region_group_executor`
end to end, but no JS-backed `group` implementation did. Added
`Region_execution.group` (a `lowered_group -> Region_group.t` accessor,
the group twin of `program`/`output_shape`), `Loop_region_program
.lower_group` (one multi-value `Kernel.t` from several sibling `Grouped`
refs, each ordinal's shape read off its own `Region_group.Emitter.t`
rather than a caller-supplied `~out_shape` — a group projects several
differently-shaped outputs off one shared recurrence), and
`Loop_region_executor.make_group` (the group twin of `make`).

**D7** (put the region executor on the node executor's own compile table
and report) was decided as a **scoped yes**: unify the report, not the
compile table. A literal merge would need `Region_executor.t`/`.group`'s
own type to carry the originating op for per-op-kind labeling (it never
receives one today, unlike `Node_executor.t`), which is real surface-area
change to a landed, public signature for a cosmetic benefit — the actual
risk the "changes a landed component" caution was about. `js/jsoo
/loop_js_pt2/loop_js_pt2.ml` instead prints and gates on all three
coverage sources (node, solo-region, region-group) together, with neither
executor's own signature touched.

**Numbers.** `--nodes` alone (no shadow, whose own cost the first timing
pass mistakenly attributed to the generated path) is ~50x faster than the
reference path on `mobilenetv2_050` (2.2s vs 109.8s, one sample, cold,
under node) and ~39x on `fastvit_sa12` (12.2s vs ~480s) — the design's own
hypothesis (§6, ~25x at kernel scale) holds strongly at whole-model scale
too. `mobilenetv2_050 --nodes` is a tier-2 CI step
(`loop_js.node.pt2.runtest`); `fastvit_sa12` stays MANUAL (too slow for
CI at either tier). Both are confirmed at full parity with `--shadow
--strict`: `mobilenetv2_050` exit 0, 415/415 nodes, zero fallback/pending,
ranking match. `fastvit_sa12` exit 0, every node kind (including
`Unbind=6`, the one gap this work closed) plus its 2 SDPA nodes via the
Region executor all `fallback=0`, ranking match.

**Every model in `PT2_MODELS_CRAM` confirmed, not only the two named
above (follow-up, 2026-09-22).** `test_convnext2`, `mobilenetv3_small_050`,
`regnetx_002` and `efficientnet_b0` also pass `--nodes --shadow --strict`
at full parity — every op kind `fallback=0 pending=0`, ranking match. All
six were already downloaded locally when this was checked; none needed
new work, since the seam is genuinely per-node and does not special-case
which model it runs on. Node counts differ per model.

**Canonical: generated JS over `Pipeline.canonical`'s output, not the raw
graph, is now the DEFAULT (2026-09-22, made default 2026-09-23).**
`js/jsoo/loop_js_pt2/loop_js_pt2.ml` runs the same `node_executor`/
`region_executor`/`region_group_executor` values, unmodified, through
`Native_interp.transform`/`evaluate` instead of `Native_interp.run` —
`Node_executor`/`Loop_node_program` are indifferent to how a
`Graph_ir.graph` was produced, so this needed no change to either. Every
model still passes at full parity on the canonicalized graph (including
its new node shapes — `Batch_norm` folds into `Conv2d`'s weights and
disappears; `Permute` drops from the hundreds to single digits). A
`--direct` flag opts back into the raw, untransformed graph; the Makefile
keeps that path exercised on the smaller `fastvit_sa12` subset
(`loop_js.pt2.run`) rather than paying for both forms on every model —
`loop_js.node.pt2.runtest` (tier-2 CI, `mobilenetv2_050`) runs canonical.
The native OCaml side made the matching choice the same day: `make
native-infer-verify` now checks the canonical (`transform --fold`) graph
against ATen for every `PT2_NATIVE_VERIFY_MODELS` entry, and
`native-infer-verify-direct` keeps the raw-graph-vs-ATen check for the
smaller `mobilenetv2_050` subset (see the Makefile's own comments there).

The speedup canonicalization itself buys is real but not uniform:
permute/batchnorm-heavy CNNs see a large additional win on top of the
generated-JS one (`efficientnet_b0` 327s → 133s, `mobilenetv2_050` 89s →
40s, both roughly 2.2-2.5x, for 3-5s of canonicalization cost), while a
model whose wall time is dominated by genuinely compute-heavy nodes sees
little (`fastvit_sa12`'s SDPA/Conv2d/Gelu-heavy graph: ~510s either way).
Node count dropping by half (`fastvit_sa12`: 712 → 366) does not imply
wall-clock time dropping by half when the removed nodes (redundant
permutes) were never the bottleneck.

## Melange

Where Melange stands today (`js_backends_design.md`, "Melange" and "Not done"):
it builds `walk_core`, `core` and `Expr` (`expr_internal_mel`/`expr_mel`, used
by the plain-value Expr probe). It does **not** build `lib/native`, because
Melange has no Bigarray. `js/melange/shim/bigarray.ml` is deliberately shaped
for jsont and must not grow into `Payload`'s kind semantics. `native_mel` waits
for melange-re/melange#1807, which is unmerged and in no release. Moving to
melange 7 does not change that. Melange stays gated behind
`--profile melange`.

What that means for each layer:

| Layer | Melange | Why |
|---|---|---|
| `js_ast` (AST, `Js_build`, printer, `Check`) | **yes, now**: mirror `js_ast_mel` | depends on `fmt` only, and `fmt_mel` already exists |
| `Loop_js` (Loop IR → AST) | no | `loop_ir` → `native` → `Payload` → Bigarray; waits for #1807 with `native_mel` |
| Executor | no, not planned | it binds `Tensor.packed` storage, which is Bigarray again. A Melange twin would also need `Melange.Js`/`[%mel.raw]` instead of `Js_of_ocaml`: a second binding layer to keep in step, for a backend that cannot yet produce a `Loop_program.t` |

**Mirror `js_ast` into Melange anyway.** That is not coverage for its own
sake. The printer's promise is the same bytes whatever backend runs it, and
Melange is the backend most likely to break it:

- `Number` literals go through `Printf "%.17g"`. Under Melange, `Printf` float
  formatting is Melange's own runtime (`caml_format_float` over JS
  `toPrecision`/`toExponential`), not glibc's, and neither jsoo nor native
  shares it.
- `Bigint` literals go through `Int64.to_string`. Melange represents `Int64`
  as a pair of 32-bit words, so `min_int`, `-1` and `2^53+1` take a
  conversion path that no other backend exercises.
- `int` is 32 bits there too, so any printer arithmetic on an OCaml `int`
  (indentation depth, `float_of_int` of an index constant) is under the same
  rule as the jsoo-reachable libraries.

The check reuses the probe pattern the JS-backends doc already relies on.
A section in `probes_pure` prints a fixed corpus of ASTs: every literal edge
above, every precedence pair, and one helper from the runtime rebuilt in the
test from `Js_build`, since `Loop_js_runtime` itself is in `loop_ir` and out
of reach. `make melange.runtest` then diffs `subset_probe.exe` against
the emitted JS, and `make jsoo.runtest` diffs its own twin. Three backends
print the same text or a diff shows up. That is the "do the backends agree"
question, and it is the right one here. Whether the printed JS is *correct*
is answered by the node gate and the executor.

When #1807 lands and `native_mel` exists, `loop_ir_mel` follows by the same
mirror pattern, and `Loop_js.to_ast` then emits under Melange with no code
change. That is the payoff of keeping lowering and printing
backend-neutral. The executor is reconsidered then, and not before: it would
be the first Melange code to bind JS objects, which the scope split in the
JS-backends doc keeps out of Melange's pure closure today.

## Testing

- **Printer goldens** (native and jsoo): every precedence pair once, negative
  literals in each position, `-0`, `NaN`, `±Infinity`, `Bigint` extremes,
  string escapes.
- **Precedence differential**: random expression trees over `Number`
  (seeded, `Walk_core`'s PCG), printed twice (once minimal, once fully
  parenthesised), evaluated by the executor under jsoo, and compared bitwise.
  A wrong table entry changes a value. Proof it bites: swap `Mul`/`Add`
  precedence and watch it go red.
- **Scope checker**: an undeclared temporary, a duplicate `let`, and a `For`
  variable used after its loop are each rejected. Proof it bites: drop one
  temporary from `declarations`, and `to_ast` fails on the existing fixtures.
- **Prelude selection**: a bf16-only program's script contains no
  `f16_to_float`. That is a golden, and today's emitter fails it.
- **Behavioural equivalence**: the node gate's `expected.txt` is produced by
  the interpreter, so it does not change. The gate stays green across the
  switch, which is the proof the AST path means the same thing. The
  `loop_js_test` text goldens do change (fewer parentheses, folded offsets).
  Promote them only after the gate is green.
- **Executor** (jsoo inline suite only): the int64 alias round-trip, one
  `Load_out_of_range` decoded with its full payload, then every
  `Loop_check` fixture and the op sweep through the third executor.

## Stages

1. **`js_ast`**: AST, `Ident`, printer, `Check`, goldens and the
   printer-only tests. Nothing uses it yet. Add `js_ast_mel` and the
   printer probe section, so `melange.runtest` and `jsoo.runtest` diff the
   printed corpus from the first commit.
2. **`Js_build`**, plus the signed-zero reachability case (settle it before
   the lowering depends on `Num.of_idx`).
3. **`Loop_js` on the AST**: helpers as AST, prelude by `free`,
   `Loop_js_failure` table. Node gate green with its `expected.txt`
   untouched. Retire the string emitter in the same change, with no
   parallel path.
4. **`loop_js_exec`**: compile/bind/decode, the int64 alias test, the
   precedence differential, and `Loop_check`'s third executor in the jsoo
   inline suite.
5. **Record**: kernel-DSL design Phase 2 notes, CLAUDE.md library table
   (`js_ast`, and `loop_js_exec` as jsoo-only), CLAUDE.md's note on backend
   scopes (Melange now reaches `js_ast` as well), and the JS-backends design
   doc's layout table and mirror list.

## Open questions

- Whether `Num` should get a real simplifier (common subexpressions, hoisting
  loop-invariant offsets). It belongs to Phase 3's CSE/LICM on the Loop IR,
  not to the JS backend. A JS-only optimisation would make the two text
  backends disagree about what they run.
- A `Number` fast path for int64 values proven within `±2^53` (already open in
  the kernel-DSL design). With `Js_build` it becomes a new builder module and
  one crossing function, not a change to every call site.
- Whether the webapp should run kernels through `loop_js_exec`. That needs the
  `loop_ir_js` mirror, and it needs the CSP decision above made per deployment.
- Decoding a `Loop_node_executor`/`Loop_region_executor` JS failure row
  directly (`Loop_js_failure`'s own decoded shape) instead of always
  re-running `direct ()` on any refusal to get an error value the caller's
  row can carry. Fine at today's cost (a refusal is rare enough that
  re-running once is cheap), but a per-node executor with heavier fallback
  traffic would pay for it twice.
- Fusing multiple nodes into one generated kernel. Every node through its
  own generated-JS kernel deliberately stopped short of this (design
  intro) — each kernel covers exactly one node's output, no cross-node
  `Fusion_plan.plan`. The speedup already measured (~40-50x over the
  reference path, whole-model) came from generated JS alone; fusion is a
  separate optimisation this parity work does not need and did not
  attempt.
