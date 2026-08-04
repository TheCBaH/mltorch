# Native `Expr` language refactoring

Status: **implemented**, August 2026. This document defines the `Expr` language:
its library boundary, syntax, semantics, construction, interpretation,
rewriting, migration, and validation. Written as a proposal; the sections marked
**Resolved** record where building it changed the answer, and those decisions
are the source of truth over the surrounding prose.

The four that a reader should know up front:

- the index domain is checked host `int`, not `int64`;
- the public modules own their recursive types in one `module rec` group in a
  main module, because every layout with one file per module fails to compile;
- `Check` verifies only what composition can violate, since the rest is
  unconstructable through the public API;
- reducer display names are numbered from 1, matching the representation this
  replaced, so the migration moved no goldens at all.

The current behavior is described by `native_compute_design.md` and
`native_symbolic_language.md`. Those remain the source of truth for operation
semantics while this document defines the new expression representation.

## Decision summary

Move the expression language out of `lib/native/expr.ml` into its own library
and directory. The public namespace is:

```text
Expr.Axis
Expr.Role
Expr.Coord
Expr.Source
Expr.Reduce_var
Expr.Index
Expr.Bool
Expr.Value
Expr.Reduction
Expr.Intrinsic
Expr.Builder
Expr.Rewrite
Expr.Fold
Expr.Eval
Expr.Pp
```

The source files are correspondingly short and unprefixed: `axis.ml`,
`index.ml`, `value.ml`, and so on. There are no `expr_index.ml` or
`expr_value.ml` units.

The language is functional:

- expressions are immutable values;
- there are no assignments, mutable variables, statement sequences, stores, or
  imperative loops in the AST;
- `Reduce` denotes a pure ordered fold, not a statement loop;
- construction and rewriting can be implemented with an explicitly threaded
  immutable name supply;
- interpretation is defined by structural recursion and pure functions.

An adapter may use encapsulated mutation as an implementation convenience when
conforming to an existing stateless interface, but neither the public language
nor its semantics depends on that choice.

## Library namespace and directory layout

### Required public namespace

In Dune, an ordinary `(wrapped false)` library exposes each unit globally. A
library containing `index.ml` and `value.ml` would expose `Index` and `Value`,
not `Expr.Index` and `Expr.Value`.

Public modules must be referenced as `Expr.Value`, `Expr.Index`,
`Expr.Rewrite`, and so on. The simplest layout that produces that form is a
separate library using Dune's default wrapping:

**Resolved by compiling it.** One file per public module does not work, and the
three smaller layouts that look like they should were each tried and rejected:

1. Separate `bool.mli` / `value.mli` units are a dune **cycle**
   (`Bool.intf -> Value.intf -> Bool.intf`): `Bool`'s guard payload is a
   `Value.t` and `Value.select` takes a `Bool.t`.
2. A `module rec` whose members *alias* a private AST module's mutually
   recursive types fails ascription — mid-check OCaml has no
   `Bool.t = Syntax.bool_expr` equation yet, and reordering only moves the
   error to the symmetric case.
3. Sequential façades over such a module fail too: whichever is written first
   names the other's type, giving `Unbound module Value`.

What compiles is the **public modules owning their types directly**, in one
`module rec` group in a main module:

```text
lib/expr/
  dune
  expr.ml / expr.mli    # the main module (see below)
  axis.ml/.mli
  role.ml/.mli
  coord.ml/.mli
  source.ml/.mli
  max_op.ml/.mli
  checked.ml            # private
```

```lisp
(library
 (name expr)
 (private_modules checked)
 (libraries core fmt))
```

`expr.ml` is named after the library, so dune emits no alias module and that
file *is* `Expr`. Its sections, **in this order** — the order is required, not
stylistic, because OCaml resolves modules in source order even for types outside
the cycle, and `Bool.Index_eq`, `Reduction.lo`/`hi`, `Value.Value_of_index` /
`Load` and `Value.Intrinsic` all reference the acyclic modules:

```text
leaf aliases (Axis, Role, Coord, Source, Max_op)
Reduce_var
Index
Intrinsic
module rec Bool / Reduction / Value
Builder / Rewrite / Fold / Check / Eval / Pp
```

Two consequences worth stating, because both are easy to get wrong:

- **An explicit main module means dune synthesises no aliases.** `Expr.Axis`
  does not exist until `module Axis = Axis` appears in *both* `expr.ml` and
  `expr.mli`. `Checked` is deliberately not aliased.
- **Dune forbids a wrapped library's subordinate modules from naming its main
  module.** So `Builder`, `Rewrite`, `Fold`, `Check`, `Eval` and `Pp` cannot
  live in their own files — they all depend on the recursive types. That is not
  purely a cost: it keeps the raw constructors reachable for `Rewrite`, which
  must rebuild structure *without* the smart constructors, since those fold and
  a structure-preserving rewrite must not silently change syntax.

`expr.mli` repeats the same sequence with the variants and the reduction record
marked `private`, so external code pattern-matches but cannot construct. The
trade is one large sectioned file instead of short per-domain ones; `expr.mli`
keeps the public surface readable in one place, and `test/expr/namespace_safety.t`
asserts both properties with paired must-compile controls.

### Dependency direction

The new library must not depend on `native`; otherwise making `native` consume
`Expr.Value` would create a library cycle. The dependency direction is:

```text
core, fmt
    │
    ▼
  expr
    │
    ▼
  native
```

Consequently the expression language owns small language-level concepts such as
its axes, coordinate record, source symbols, and position/delta role markers.
It does not store `Tensor.t`, `Tensor_sig.t`, `Vec6.t`, `Dim.t`, payload formats,
or graph objects. Native adapters translate those values at the boundary.

This makes `Expr` a real independent library rather than a directory containing
modules that still secretly form part of the `native` compilation unit.

## Motivation

The current `Expr` successfully represents per-coordinate computations, but its
minimal representation makes several correctness properties implicit.

### Current language

`Expr.t` currently contains:

- `Const`;
- binary add, subtract, multiply, and divide;
- unary exponential and square root;
- comparisons and conditional selection;
- index-to-value conversion;
- tensor loads with six index expressions;
- ordered sum and maximum reductions;
- compact max-pool value/index operations.

`Expr.index_expr` contains output-axis variables, reduction variables, integer
constants, addition, integer scaling, positive floor/ceiling division, min, and
max. `Expr.eval` interprets the resulting tree against concrete coordinates and
a tensor binding.

### Problems to solve

| Current property | Refactoring motivation |
|---|---|
| values, booleans, and indices share one source module | their different invariants and consumers are difficult to see |
| position/delta roles disappear in `index_expr` | a signed expression can reach a load without retaining where nonnegativity was established |
| reducer variables are plain integers | independently constructed expressions can collide when combined |
| reducer allocation is tied to `Symbolic.Make` | the AST has no self-contained composition protocol |
| a load embeds a Native tensor signature | the language cannot become an independent library |
| index evaluation uses OCaml `int` | large intermediate arithmetic can wrap before a later bound check |
| constants are stored and compared as ordinary floats | signed zero and NaN bit distinctions can be lost accidentally |
| recursive traversals are open-coded | evaluator, printer, rewriting, and inspection can disagree about binders or new constructors |
| the AST cannot express an f32 round | a scalar conversion that changes values has no representation |
| compact operations lack a common capability boundary | it is unclear which supporting functions must accompany a new intrinsic |

The refactoring makes these contracts explicit without changing the computations
described by existing operation definitions.

## Goals

- Provide a self-contained library whose public modules are all under `Expr`,
  including `Expr.Index`, `Expr.Value`, and `Expr.Eval`.
- Keep the AST immutable and its semantics purely functional.
- Separate axes, roles, indices, values, booleans, reductions, and sources.
- Preserve position-versus-delta typing in constructed index expressions.
- Make reducer scope and capture avoidance explicit.
- Use checked-width integer arithmetic in symbolic index interpretation.
- Define constants, equality, hashing, and printing consistently.
- Represent f32 rounding as an explicit value expression.
- Centralize structural traversal and rewriting.
- Provide a reference interpreter independent of Native tensor storage.
- Preserve the meaning and printed readability of all current expressions.
- Permit gradual migration from the current monolithic `Expr` module.

## Non-goals

- Changing existing operation mathematics.
- Adding assignments, statement blocks, mutation, or general recursion.
- Adding output iteration or tensor storage operations to the language.
- Defining graph, tensor allocation, scheduling, or transformation types.
- Making the pretty-printer a parser or stable serialized format.
- Adding general mixed-dtype arithmetic.
- Performing floating-point algebraic simplification.
- Replacing the existing direct evaluator for operations.
- Turning every compact intrinsic into generic expression nodes immediately.
- Introducing a universal extensible-effects or visitor framework.

## Functional versus imperative constructs

### AST semantics are functional

Every constructor denotes a value. Evaluation depends only on:

- the expression;
- the six output-coordinate components;
- reducer bindings introduced lexically by enclosing reductions;
- the supplied source-load function.

There is no observable evaluation state. The same expression and environment
produce the same result.

`Select` is a pure conditional expression. Only the selected branch is
evaluated. `Reduce` is a pure left fold over a half-open integer domain. A host
implementation may compile a fold into a loop, but mutation is not part of the
language or its denotation.

### Name generation is not language mutation

Reducer identities need to be fresh during construction and rewriting. The
public design uses a pure state-passing builder:

```ocaml
module Builder : sig
  type state
  type 'a t

  val return : 'a -> 'a t
  val bind : 'a t -> ('a -> 'b t) -> 'b t
  val run : 'a t -> 'a
  val fresh_reduce : Reduce_var.t t
end
```

The reference implementation is equivalent to
`state -> value * state`; the representation remains private. Nested builders
compose with `bind`, so callers never mutate or manually synchronize a counter.
`Rewrite.freshen` returns a builder computation for the same reason. This makes
construction deterministic and referentially transparent.

An adapter for an existing module interface may encapsulate a `ref` and call the
same constructors. That is an implementation detail outside the expression
model. Expressions produced by such an adapter are indistinguishable from those
built with explicit supply threading.

### Interpreter implementation

The reference interpreter is specified recursively. An implementation may use
tail recursion or a private local loop for stack safety and efficiency. This
does not add an imperative construct to the language because no state is exposed
and the result remains the defined ordered fold.

## Semantic domains

### Axes and coordinates

The language retains the fixed ordered frame:

```ocaml
module Axis : sig
  type t = N | T | D | H | W | C

  val all : t list
  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit
end
```

`Coord.t` is the expression library's six-field product, parameterized over its
component type:

```ocaml
module Coord : sig
  type 'a t = {
    n : 'a;
    t : 'a;
    d : 'a;
    h : 'a;
    w : 'a;
    c : 'a;
  }

  val get : 'a t -> Axis.t -> 'a
  val map : ('a -> 'b) -> 'a t -> 'b t
end
```

The record follows the repository convention by living in `Coord` and being
named `t`. Native conversion to and from `Vec6` is external to this library.

### Position and delta roles

```ocaml
module Role : sig
  module Position : sig
    type t
  end

  module Delta : sig
    type t
  end
end
```

The types are markers, not runtime values:

- `Role.Position.t Index.t` may be used by `Value.load`;
- `Role.Delta.t Index.t` may be negative and supports arithmetic;
- converting a position to a delta is always safe;
- converting a delta to a position requires a constructor that records why it
  is considered nonnegative.

The Native adapter must align its existing position/delta roles with these
markers using manifest type aliases. In particular, `Dim.index` aliases
`Expr.Role.Position.t` and `Dim.delta` aliases `Expr.Role.Delta.t`; the existing
`Semantics.position`/`delta` aliases then continue through `Dim`. A runtime
conversion is insufficient because the symbolic implementation's associated
`'role index` type must type-check at the module boundary.

### Source symbols

`Expr.Source.t` is an opaque symbol naming an external scalar-addressable source:

```ocaml
module Source : sig
  type t

  val create : int -> t
  val to_int : t -> int
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end
```

The language deliberately knows nothing about tensor ownership, storage format,
quantization, or graph identity. Its evaluator asks the caller to provide the
working floating-point value for a source and coordinate. This is the boundary
that keeps the library independent.

Source creation should be deterministic. The initial representation may be a
private `int`; no semantics may depend on allocation order beyond symbol
equality.

### Working values

Value arithmetic follows the current OCaml working domain. `Const`, primitive
arithmetic, selection, loads, and reductions produce working floating-point
values.

`Round_f32 e` means:

```text
convert e to IEEE binary32 using the specified rounding mode,
then widen the binary32 result back to the working value domain
```

It remains an expression value and can participate in ordinary arithmetic. It
does not imply a store or mutation.

## Proposed syntax

The following signatures communicate the intended invariants. Exact OCaml
syntax for mutually recursive modules may differ.

### Reducer variables

```ocaml
module Reduce_var : sig
  type t

  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end
```

Construction is restricted to `Builder`. Reducer identity is meaningful only
within an expression namespace; display names are assigned lexically by the
printer rather than exposing the internal ID.

### Index expressions

```ocaml
module Index : sig
  type _ t

  val output : Axis.t -> Role.Position.t t
  val reduce : Reduce_var.t -> Role.Position.t t
  val zero : Role.Position.t t
  val const : int -> Role.Delta.t t
  val of_position : Role.Position.t t -> Role.Delta.t t
  val add : Role.Delta.t t -> Role.Delta.t t -> Role.Delta.t t
  val scale : int -> Role.Delta.t t -> Role.Delta.t t
  (* result-returning: the divisor is a raw int and must be validated *)
  val floor_div_pos : Role.Delta.t t -> int -> (Role.Delta.t t, error) Core.result
  val ceil_div_pos : Role.Delta.t t -> int -> (Role.Delta.t t, error) Core.result
  val min : Role.Delta.t t -> Role.Delta.t t -> Role.Delta.t t
  val max : Role.Delta.t t -> Role.Delta.t t -> Role.Delta.t t
  val clamp_low : Role.Delta.t t -> Role.Position.t t
  val assume_position : Role.Delta.t t -> Role.Position.t t
end
```

The implementation is expected to be a GADT or equivalent private
representation. Smart constructors are public; raw constructors may be exposed
through an internal view used by sibling modules.

`assume_position` remains a distinct node. It does not prove nonnegativity; it
records a claim made by the expression author. Inspection can locate every such
claim. `clamp_low` establishes nonnegativity by its own semantics.

`floor_div_pos` and `ceil_div_pos` require a strictly positive divisor. The
smart constructor returns a result only from a validated positive value, or the
operation returns a `Core.result`. A raw zero/negative divisor must not enter the
AST.

### Boolean expressions

```ocaml
module rec Bool : sig
  type t = private
    | Value_lt of Value.t * Value.t
    | Index_eq of
        Role.Delta.t Index.t * Role.Delta.t Index.t
end
```

`Bool.t` is not a general standalone logic. It exists to guard `Value.select`.
Additional predicates should be added only when required by an existing
operation.

### Reductions

```ocaml
and Reduction : sig
  type kind = Sum | Max

  type t = {
    kind : kind;
    var : Reduce_var.t;
    lo : Role.Position.t Index.t;
    hi : Role.Delta.t Index.t;
    body : Value.t;
  }
end
```

The bound variable is in scope only in `body`. Bounds may refer to enclosing
reducers, but not to `var` itself unless recursive domains are deliberately
introduced later.

The denotation is the current ordered half-open left fold:

```text
Sum: acc starts at 0
Max: acc starts at negative infinity
for r = lo, lo + 1, ..., hi - 1:
  acc becomes combine(acc, body with var = r)
```

Operand order and update association are semantic. Rewriting must not treat a
reduction as an unordered collection.

### Value expressions

```ocaml
and Value : sig
  type binary_op = Add | Sub | Mul | Div
  type unary_op = Exp | Sqrt

  type t = private
    | Const of float
    | Binary of binary_op * t * t
    | Unary of unary_op * t
    | Select of Bool.t * t * t
    | Value_of_index of Role.Delta.t Index.t
    | Load of Source.t * Role.Position.t Index.t Coord.t
    | Round_f32 of t
    | Reduce of Reduction.t
    | Intrinsic of Intrinsic.t
end
```

The target has no general `Let`. Values are immutable trees. If a current
operation demonstrates that scalar sharing must be semantic rather than an
evaluation optimization, `Let` can be considered later together with its binder
and substitution rules.

### Intrinsics

The current compact max-pool operations remain one closed intrinsic family. The
descriptor stores only expression-library values:

- source symbol;
- input height and width;
- positive window height and width;
- positive stride height and width;
- nonnegative padding height and width;
- symbolic output coordinate;
- result choice: value or flattened index.

No Native configuration or tensor type appears in the descriptor. The Native
adapter converts already-validated parameters to the expression descriptor.

Geometry derived from a descriptor — window bounds, the flattened index — is
**public and result-returning**, and goes through the same checked arithmetic as
`Index`. There are two interpreters: Native's `Ground_eval` expands the same
stencil and, as a consumer, cannot reach the library-private checked helpers, so
routing only the reference evaluator through them would leave the verifier's
copy unchecked and let the two drift. Validating the descriptor's fields is not
sufficient either — `out_h * stride` and `ih * in_w` are *aggregates* of
individually in-range factors and need their own bounds.

An intrinsic definition must provide:

- structural comparison and hashing;
- source and index traversal;
- interpretation;
- deterministic printing;
- size accounting;
- reducer-variable freshening where relevant.

This list is the language-level capability contract. An intrinsic missing one
of these operations is not part of a complete `Expr` implementation.

## Construction

### Smart constructors

Public construction should go through `Value`, `Bool`, `Index`, `Reduction`, and
`Builder` functions. Private constructors prevent malformed reducer scope and
invalid division parameters from being created casually.

Smart constructors may perform only small, unconditional folds such as:

- scale by one;
- addition of an index constant zero;
- division by one;
- min/max of structurally identical indices;
- folding checked integer-only constants.

They must not apply floating-point algebraic identities. In particular, do not:

- reorder add or multiply operands;
- reassociate arithmetic;
- remove addition by floating zero;
- reorder reduction updates;
- erase `Round_f32` based only on its argument's syntax.

Signed zero, NaN behavior, and rounding make apparently ordinary identities
observable.

### Pure builder

```ocaml
module Builder : sig
  type state
  type 'a t

  val return : 'a -> 'a t
  val bind : 'a t -> ('a -> 'b t) -> 'b t
  val run : 'a t -> 'a
  val fresh_reduce : Reduce_var.t t

  val reduction :
    kind:Reduction.kind ->
    lo:Role.Position.t Index.t ->
    hi:Role.Delta.t Index.t ->
    (Role.Position.t Index.t -> Value.t t) ->
    Value.t t
end
```

`reduction` allocates a variable, passes its index expression to the callback,
and threads the updated immutable state through the returned computation. Nested
construction composes with `bind`. Convenience syntax may make this read like
direct construction, but its meaning remains pure state passing and requires no
global counter.

### Compatibility adapter

The existing symbolic operation interface does not thread builder state. Its
adapter can remain a generative module and privately retain the current local
counter or supply reference. This is acceptable because:

- the mutation is encapsulated;
- it affects only opaque identity allocation;
- the resulting AST is immutable;
- construction order is already fixed by operation evaluation;
- combining results from independent adapters still goes through freshening.

This is the only anticipated imperative implementation convenience. It is not
an AST construct or a semantic assumption.

## Scope and rewriting

### Central rule

No consumer may recursively rewrite `Value.t` while ignoring reducer scope.
`Expr.Rewrite` owns all transformations that can cross a binder.

### Required operations

```ocaml
module Rewrite : sig
  val freshen : Value.t -> Value.t Builder.t

  val substitute_output :
    Role.Position.t Index.t Coord.t -> Value.t -> Value.t

  val alpha_normalize : Value.t -> Value.t

  val map_sources : (Source.t -> Source.t) -> Value.t -> Value.t
end
```

`freshen` replaces every bound reducer identity consistently and preserves free
variables. A top-level well-formed expression should have no free reducer.

**Freshen the fragment being inserted, before composing it — not the result.**
Once a nominal collision exists, the damage is already done: if a fragment's own
binder shares an ordinal with a variable the fragment references, the two
references are indistinguishable in the tree, and renaming the binder
necessarily drags the outer reference along. There is no record of which binder
it meant. The sound alternative is not to create the collision — draw the
fragment's variables from the destination's supply via `Builder.run_from`, or
freshen before the reference is placed.

Note also what composing two *well-formed* fragments actually produces:
shadowing, not capture. Neither references the other's variables, so evaluation
is unaffected and the damage is structural — alpha-normalisation, comparison and
every binder-counting fold stop meaning anything once one variable binds twice
on a path. That is why `Check` rejects it even though no number changes — and
why the symptom has to be *measured* rather than read off the printed form: the
printer's naming environment is scoped, so it renders the shadowing exactly as
evaluation resolves it and shows nothing wrong.

Note that `freshen` must skip identities occurring **free** in the expression.
Drawing a replacement onto one binds a reference whose binder lives outside, so
an ill-scoped fragment silently becomes well-scoped-looking with a different
denotation — capture inflicted by the very function that exists to prevent it.

`substitute_output coord body` replaces only output-axis variables. It does not
replace reducer variables. If substitution places `body` beneath another
reduction scope, the caller freshens it first.

`alpha_normalize` deterministically renames reducers by lexical traversal. It
does not reorder operations or reductions.

Its numbering is *the lowest ordinals not occurring free in the expression*, in
lexical order — `0, 1, ...` for a closed expression, and skipping the free ones
otherwise. It cannot be "always from zero": `freshen` must not draw a
replacement that lands on a free identity, because doing so binds a reference to
a binder outside the expression, which turns an ill-scoped term into a
well-scoped-looking one with a different denotation. `alpha_normalize` makes
that maximally likely, since it always starts from `Builder.initial` and so
walks straight into any low free ordinal. The result is still canonical:
alpha-equivalent expressions have the same free set and therefore skip the same
ordinals.

### Substitution law

For a well-formed body `e`, coordinate expression `s`, concrete coordinate `o`,
and source environment `env`:

```text
eval(env, o, substitute_output(s, e))
=
eval(env, evaluate_indices(o, s), e)
```

This law should be property-tested, including nested reductions whose internal
IDs were originally allocated by independent builders.

### Read-only folds

`Expr.Fold` provides reviewed, scope-aware queries:

- node count and depth;
- sources and source use counts;
- output axes used;
- free reducer variables;
- index expressions and `assume_position` sites;
- intrinsic occurrences;
- structural hash.

These functions operate on the stored tree. Structural hash is an optimization,
never a replacement for structural equality.

Avoid a maximally generic visitor abstraction at first. Concrete folds and
rewrites make binder behavior visible in signatures and error reports.

## Interpretation

### Evaluation environment

```ocaml
module Eval : sig
  module Env : sig
    type t = {
      load : Source.t -> int Coord.t -> (float, error) Core.result;
    }
  end

  val index :
    output:int Coord.t ->
    reducers:(Reduce_var.t -> int option) ->
    'role Index.t ->
    (int, error) Core.result

  val value :
    Env.t ->
    output:int Coord.t ->
    Value.t ->
    (float, error) Core.result
end
```

The environment's `load` function owns all source-specific behavior. `Expr`
supplies evaluated coordinates and consumes the resulting working float.

This preserves library independence and makes the evaluator easy to test with a
pure map from `(Source.t, Coord.t)` to values.

### Checked integer behavior

**Resolved: the index domain is host `int`, checked — not `int64`.** The
motivation stated above is that *unchecked* `int` arithmetic can wrap before a
later bound check; that is answered by checking, not by widening. `int64` would
not remove the narrowing, only relocate it to every host that must produce a
Bigarray offset — the exact defect class CLAUDE.md logs six instances of — and
`Int64` is an allocating three-field emulation under js_of_ocaml, on a path
evaluated once per output pixel.

The predicates are width-agnostic and therefore self-adapting: under
js_of_ocaml, where `Sys.int_size` is 32, the same comparisons fire at 2^31. That
is a *stronger* guarantee than `int64` gives, because it rejects at the
operation that would wrap rather than at a boundary narrow. The consequence — an
index reaching 2^40 is accepted natively and rejected under jsoo — is honest:
the engine cannot index such a tensor there regardless.

Evaluation checks every operation before producing its result:

- addition and subtraction overflow;
- scale/multiplication overflow;
- positive division parameters;
- conversion of an index to a working float where precision matters.

**Every check is a pre-operation bound.** A check on a wrapped result is not a
bound. In particular `add` does not compute `a + b` and inspect operand signs
against the result sign — sound under two's-complement wrapping, but not a
bound — and `mul` does not use `r / a = b`, which at `a = -1, b = min_int` passes
on a genuine overflow because the product wraps to `min_int` and `min_int / -1`
is `min_int` too. Every arithmetic expression *inside* a check must itself be
safe.

There is no checked negation: once division stops negating its dividend (below),
nothing in the language negates anything — `scale (-1) x` goes through `mul`.

Checking only the final result is insufficient because an intermediate can wrap
back into range. The evaluator returns a structured error before calling
`Env.load` with a wrapped coordinate.

### Floor and ceiling division

For positive `d`:

```text
floor_div(n, d) = greatest integer q such that q*d <= n
ceil_div(n, d)  = least integer q such that q*d >= n
```

**Implemented by remainder adjustment, not by negating the dividend.** With host
division truncating toward zero, `q = n / d`, `r = n mod d`, then floor is
`if r < 0 then q - 1 else q` and ceiling is `if r > 0 then q + 1 else q`.

`ceil_div(n, d) = -floor_div(-n, d)` is the same function on paper but a worse
implementation: for a positive divisor every floor and ceiling quotient of every
representable dividend is itself representable (`ceil_div(min_int, 1) =
min_int`), so negating the dividend manufactures an overflow the operation does
not have. The remainder form has no error case beyond the divisor, a positive
divisor rules out the trapping `min_int / -1`, and the adjustment cannot
overflow either — the only dividend at a representation boundary is reached when
`d = 1`, where the remainder is zero and no adjustment happens. It is therefore
strictly *more* defined than the form it replaces.

Tests must cover values immediately around negative and positive multiples of
`d`, as host-language integer division truncates toward zero instead, and must
assert the *mathematical* result at `min_int`/`max_int` rather than an overflow.

### Index-to-float conversion

Exactness is a **round trip**, not a magnitude threshold: `float_of_index i`
succeeds iff `Int64.of_float (float_of_int i) = Int64.of_int i`. "Reject above
2^53" would be wrong in both directions — binary64 stops representing
*consecutive* integers there, but `2^53 + 2`, `2^54` and `min_int` are exact
while `2^53 + 1`, `2^54 + 2` and `max_int` are not.

The conversion short-circuits when `Sys.int_size <= 53`: every representable
js_of_ocaml `int` is already exact, so running the round trip there would put
two emulated `Int64` conversions per `Value_of_index` back on the per-pixel
path — the cost the `int` domain was chosen to avoid — for a check that can
never fire.

It is a **shared helper**, not an evaluator detail. Native's `Ground_eval` is a
second interpreter and converts independently in two places; a geometry helper
returning `int` does not make the conversion that follows it exact.

### Floating behavior

- Binary and unary operations retain the current OCaml order.
- `Select` evaluates only its selected branch.
- `Round_f32` performs the specified binary32 round trip. A likely reference
  implementation is the corresponding float-to-bits and bits-to-float pair,
  checked against an f32 storage round trip.
- Sum and maximum use the current seeds and left-fold order.
- Max-pool uses the shared selection rule: ordinary ties retain the incumbent,
  and value/index updates remain paired under NaN handling.

### Errors

Errors should distinguish:

- unknown source;
- unbound reducer variable;
- invalid positive parameter;
- integer overflow;
- invalid coordinate supplied to a load callback;
- malformed reducer scope;
- unsupported intrinsic descriptor.

Use `Core.result` and repository error-composition conventions. Do not rebuild an
error by extracting its kind and thereby discard its detection backtrace.

## Equality, ordering, hashing, and printing

### Constants are bitwise

`Value` owns `compare`/`equal`/`hash`. Constant comparison and hashing use
`Int64.bits_of_float` — and the hash must mix **both halves** of those bits:
`Int64.to_int` drops the top bit on a 63-bit `int` and the top 32 under
js_of_ocaml, so hashing the raw conversion silently collides `-0.` with `0.`,
which is the very pair this rule exists to separate. Ordinary floating
comparison is not adequate because it can conflate:

- negative and positive zero;
- NaN and non-NaN behavior;
- distinct bit representations relevant to exact structural identity.

The constructor can continue storing `float` in memory. If a serialized form is
introduced later, it should store the bits explicitly.

### Reducer alpha-equivalence

Structural comparison first alpha-normalizes or compares under a lexical binder
mapping. Expressions that differ only in opaque reducer identities compare
equal. Expressions with different nesting, bounds, kind, update order, or body
do not.

### Deterministic printer

`Expr.Pp` assigns reducer display names `r1`, `r2`, ... in lexical order. Output
must not depend on the hidden numeric IDs or allocation history. (Numbered from
one, matching the allocation counter of the representation this replaced, which
already incremented before building the body and so was already lexical —
that is what let the cutover move no goldens.)

The naming environment is **scoped**, threaded down the tree beside a counter
that runs across it — not one map keyed by reducer identity. Two independently
built fragments can bind the same ordinal in *sibling* scopes; that is well
scoped, and `Check` accepts it, because neither shadows the other. A global
identity-to-name map keeps only the last such occurrence, so it prints both
binders under one name and renders a free reference in one sibling as bound
because its twin binds that identity. `Fold.binders` remains identity-keyed and
is for inspection only, for the same reason.

The printer should make these constructs visible:

- output axes;
- assumed positions versus clamped positions where diagnostically useful;
- source symbols;
- half-open reduction bounds;
- `Round_f32`;
- intrinsic geometry and result choice.

Pretty output is for diagnostics and expect tests. Parsing it is not a supported
construction path.

## Well-formedness

The library provides a structural validation pass independent of concrete source
data, as `Expr.Check.value` — its own section rather than a member of `Value`,
because it needs the same scope-aware traversal as `Fold` and putting it in the
façade would invert the layering:

```ocaml
val Check.value : ?max_size:int -> ?max_depth:int -> Value.t -> (unit, error) Core.result
```

**Narrowed, deliberately.** It verifies only what *composition* can still
violate:

- every reducer reference is in lexical scope;
- a reducer does not bind the same opaque variable as an enclosing reducer;
- the expression remains within configured size and nesting limits if supplied.

**The limits are metered, are checked first, and share one traversal.**
Measuring the whole tree and comparing afterwards is not a resource guard: both
scope traversals recurse over the entire expression, so an oversized input
exhausts the stack before the limit is ever consulted — the check fails on
exactly the input it exists to reject. So a supplied limit runs ahead of the
scope rules and stops the traversal at the first node past it, which bounds the
recursion by the limit rather than by the input.

**One walk per limit is not enough**, and this is the part that is easy to get
wrong twice. Whichever walk runs first still descends the full input whenever
its *own* bound is loose, so a size limit too loose to reject anything happily
walks a million levels before a tight depth limit is ever consulted — and
reversing the order simply breaks the dual case, a huge shallow tree under a
loose depth limit. Both budgets therefore ride a **single** traversal, which
bounds the recursion by the tighter of the two. An absent limit is `max_int`
(unreachable, not a skipped pass); with both absent there is nothing to bound
and the walk is skipped. `Fold.size` and `Fold.depth` are that same traversal
with both budgets unreachable, so there is exactly one description of what
counts as a node and what counts as a level.

Two consequences are part of the contract: an expression that is both oversized
and ill-scoped is reported as oversized, and `` `Too_large ``/`` `Too_deep ``
carry the **limit**, not the measure — naming the actual size would mean
measuring the whole tree, which is the thing being avoided.

The other rules originally listed here are **not** checked, and must not be
added back: division parameters, load roles, intrinsic dimensions and the
private representation invariants are each made unconstructable through the
public API by a result-returning smart constructor or by the `Index` GADT. A
rule for them could never be turned red, and CLAUDE.md is explicit that a check
which has never failed is not evidence. Consumers that need the guarantee get it
from the constructor, not from a later pass — in particular the Native adapter's
safety rests on its own typed inputs (`Op_config.Pos.t`, `Nonneg.t`,
`Dim.extent Dim.t`), not on `Check`.

`assume_position` is structurally valid but remains visible in the inspection
result. Proving that its value is actually nonnegative depends on the caller's
coordinate domain and is outside this language-only check.

## Relationship to the current Native modules

This section describes migration boundaries only; the `Expr` library itself has
no dependency on these modules.

### Role and coordinate adapters

The Native adapter converts:

- `Axis.t` to `Expr.Axis.t`;
- `Vec6` coordinates to `Expr.Coord`;
- `Tensor_id.t` to deterministic `Expr.Source.t` values.

Native's position/delta markers become manifest aliases of
`Expr.Role.Position.t` and `Expr.Role.Delta.t`. Unlike axes and coordinates,
these cannot be ordinary adapter conversions because they appear as phantom
parameters in the symbolic module's associated index type.

### Source binding

Current `Tensor_sig.t` metadata stays in Native. A Native evaluation environment
maps each `Expr.Source.t` back to its tensor signature and concrete tensor, then
implements `Expr.Eval.Env.load` with the existing raw-read semantics.

The expression AST therefore stays stable if Native later changes how signatures
or tensor storage are represented.

### Existing symbolic adapter

The current symbolic module changes its associated types to:

```text
value = Expr.Value.t
index = role Expr.Index.t
input = adapter-owned source descriptor
```

Each `load` extracts the `Expr.Source.t` and builds `Expr.Value.Load`. Arithmetic
and reductions delegate to Expr smart constructors. Operation definitions remain
unchanged.

### Existing expression consumers

Migration must update these consumers before deleting `lib/native/expr.ml`:

- symbolic expression construction;
- schedule grounding;
- whole-graph symbolic stage construction and grounding;
- concrete-coordinate ground conversion;
- expression printers in tests;
- any coefficient/normalization adapter that names old operation variants.

During migration, `lib/native/expr.ml` may temporarily re-export or adapt the new
types. The compatibility module must be removed once every consumer uses the
new modules under `Expr`; leaving two authoritative representations would
defeat the refactor.

## Migration sequence

This is dependency order for a later implementation plan, not a prescribed
commit list.

### Stage 1: create the independent library

- Add `lib/expr/dune` with default wrapping.
- Add `Axis`, `Role`, `Coord`, `Source`, and `Reduce_var`.
- Add direct unit tests for comparison, printing, and coordinate access.
- Confirm external test code sees `Expr.Axis`, not a leaked top-level `Axis`.

Exit criterion: the library builds independently with only its declared small
dependencies and exposes the intended namespace.

### Stage 2: add typed indices

- Implement the private/GADT index representation.
- Add smart constructors and checked `int` interpretation.
- Pin negative floor/ceiling division behavior.
- Add structural equality, hashing, and printing.

Exit criterion: every current index-expression form has an equivalent typed form
and a reference evaluation test.

### Stage 3: add values, booleans, reductions, and intrinsics

- Implement mutually recursive syntax modules.
- Implement pure builder supply and reducer scoping.
- Implement structural check, comparison, hashing, size, and printing.
- Port max-pool descriptors to library-owned primitive data.

Exit criterion: every current expression can be represented and printed without
depending on a Native type.

### Stage 4: add rewriting

- Implement reducer freshening.
- Implement output-coordinate substitution.
- Implement alpha-normalization and source mapping.
- Add property and mutation tests for capture avoidance.

Exit criterion: independently built nested reductions compose without capture,
and substitution satisfies its interpretation law.

### Stage 5: add the reference interpreter

- Implement checked index and value evaluation.
- Implement the load callback boundary.
- Implement `Round_f32` and intrinsic semantics.
- Differential-test the new interpreter against the current one over the shared
  old language.

Exit criterion: old and new interpreters agree bitwise on the characterization
suite, and new overflow/error cases fail explicitly.

### Stage 6: migrate Native construction and consumers

- Adapt symbolic construction to `Expr.Value`, `Expr.Index`, and the other new
  modules.
- Add source, coordinate, and role conversions.
- Migrate each current expression consumer.
- Keep operation definitions unchanged.
- Delete the compatibility representation and old monolithic file.

Exit criterion: there is one expression representation, all existing symbolic
tests pass, and no `Expr_*` modules or globally leaked `Index`/`Value` units are
required.

## Validation strategy

### Characterization before migration

Before changing representation, pin:

- pretty output for every constructor;
- constant bit comparisons, including both zero signs and NaN;
- nested reducer construction and evaluation;
- negative index division boundaries;
- selection branch laziness;
- empty and one-element reductions where legal;
- max-pool ties and NaN selection;
- checked overflow expectations for the new evaluator.

A test intended to protect a formerly implicit invariant must be demonstrated to
fail under a deliberate defect.

### Expression unit tests

- Role-preserving constructors admit only valid index combinations.
- `Value.check` rejects free and duplicate reducer bindings.
- Freshening preserves interpretation.
- Alpha-normalization is idempotent.
- Alpha-equivalent reductions compare and hash equally.
- Source mapping changes only source symbols.
- Constant equality and hashes follow bits.
- Printers are deterministic across different internal reducer IDs.
- `Round_f32` matches the chosen reference round trip.
- Intrinsic traversal visits every contained source and index.

### Property tests

For generated well-formed small expressions:

```text
eval(freshen(e)) = eval(e)
alpha_normalize(alpha_normalize(e)) = alpha_normalize(e)
eval(substitute_output(s, e), o) = eval(e, eval_indices(s, o))
compare(e1, e2) = 0 implies hash(e1) = hash(e2)
```

Generation must respect reducer scope. Separate malformed generators should
exercise `Value.check` and error reporting.

### Differential tests

During coexistence, construct equivalent old and new expressions from the same
operation description and compare:

- evaluated value bits;
- evaluated load coordinates;
- reduction iteration order;
- selected intrinsic value/index result;
- printed structure, allowing only intentional namespace/notation changes.

### Mutation expectations

Tests must reject at least:

- replacing one output axis with another;
- changing a reduction upper bound by one;
- capturing a reducer from an independently built expression;
- treating a signed delta as a position silently;
- using truncating division for a negative numerator;
- comparing constants with ordinary float equality;
- evaluating both `Select` branches;
- removing `Round_f32`;
- wrapping an overflowing index intermediate;
- updating max-pool value without the matching index.

## Performance constraints

The language is a symbolic/reference path, but basic operations should remain
linear and predictable:

- construction is linear in the produced tree;
- freshening and alpha-normalization are linear in tree size;
- folds visit each node once;
- printing does not repeatedly recompute free-variable sets;
- interpretation checks arithmetic without rebuilding the expression;
- structural hashing may be cached only if immutability makes the cache
  observationally irrelevant.

No design decision should force mutable nodes or pointer identity. Sharing may
be added internally later, but semantic equality remains structural.

The direct operation evaluator remains outside this library and is not changed
by these constraints.

## Error and diagnostic requirements

Errors should report enough context to locate the malformed expression:

- constructor being evaluated or checked;
- output axis or reducer display name;
- source symbol and coordinate;
- offending integer operands;
- whether the failure is scope, division, overflow, coordinate, or intrinsic
  configuration;
- a bounded pretty-printed expression fragment.

Use the repository's `Core.result` conventions and preserve error backtraces.
Printing reducer names lexically rather than exposing internal IDs makes errors
stable and comparable across runs.

## Alternatives considered

### Keep `Expr` inside the unwrapped Native library

This requires prefixed units such as `Expr_index` to avoid leaking generic names
globally. It also makes the expression language depend directly on Native tensor
and shape representations. A separate namespace is clearer and enforces the
intended dependency boundary.

### Use `(wrapped false)` for the new library

Plain wrapped-false produces top-level `Index` and `Value`, which is exactly the
naming collision the move is intended to avoid. A qualified nested directory
can compensate, but default wrapping is simpler and gives the desired API
directly.

### One universal typed expression

A single `type _ t` could contain values, booleans, and indices. It makes simple
value traversal and mutually recursive reduction descriptors harder to read.
The selected design uses strong typing where it carries a specific invariant—the
position/delta index distinction—and separate modules for the other domains.

### De Bruijn reducer indices

De Bruijn indices make alpha-equivalence structural and need no fresh names, but
substitution and diagnostics become less accessible. Abstract nominal variables
with a pure supply and explicit freshening are preferred. The internal
representation can be revisited if freshening cost becomes material.

### Expose raw constructors

Public variants ease pattern matching but let every consumer build malformed
scope or invalid division parameters. Private constructors plus sibling views
and centralized folds keep the public contract enforceable.

### Add general `Let` and statements

`Let`, assignment, and statement sequencing expand scope and evaluation rules
without being required by the current operation language. The target remains a
pure expression tree. Add a new binding form only in response to a concrete
semantic requirement and with corresponding substitution laws.

## Resolved decisions

- `Expr` becomes its own independent library and directory.
- Public modules are under `Expr`; source files have short names such as
  `index.ml` and `value.ml`.
- Use Dune's default wrapping to obtain that namespace.
- The library depends only on small shared support libraries, not Native.
- The AST is immutable and purely functional.
- No imperative construct is present in the syntax.
- Pure supply threading is the reference binder-allocation model.
- Encapsulated adapter mutation is permitted but not semantically observable.
- Axes, coordinates, roles, and source symbols are owned by `Expr`.
- Native position/delta markers become manifest aliases of the Expr roles.
- Sources are opaque; source-specific loading is an evaluator callback.
- Index expressions retain position/delta roles and use checked `int` values.
- Reducer variables are abstract, scoped, freshenable, and alpha-normalizable.
- Values, booleans, reductions, indices, and intrinsics have separate modules.
- A private `Syntax` unit owns mutually recursive types; public units expose one
  path such as `Expr.Value` or `Expr.Bool` for each language domain.
- `Round_f32` is a pure value expression.
- Constants compare and hash by binary64 bits.
- Max-pool remains a closed intrinsic with complete language support.
- Rewriting is centralized and scope-aware.
- The pretty-printer is deterministic but not a serialized language.
- The old monolithic representation is removed after migration.

## Open questions, as resolved

Each was settled by building it; the details are in the sections above.

1. **Private variants versus an internal view.** Neither, quite: the public
   modules own their types directly and `expr.mli` re-declares them `private`.
   An internal AST module cannot work — a `module rec` aliasing its recursive
   types fails ascription mid-check.
2. **Supply representation.** A flat `int` ordinal behind an abstract
   `Builder.state`, with `initial`/`run_from` so a non-monadic adapter can
   resume rather than restart, and `Supply_exhausted` checked before it wraps.
3. **`Source.t` allocation.** A stateless bijection with `Tensor_id.t` through
   the integer, so nothing depends on allocation order.
4. **Private wrapper types inside `Intrinsic`.** No — Native already owns
   `Op_config.Pos`/`Nonneg`, and the result-returning constructor validates.
5. **Printer stability.** All of it. Every form renders as before, which is what
   let the cutover land with zero golden movement.
6. **Size/depth limits.** Optional arguments on `Check.value`, not a separate
   `Fold` query.
7. **Structured error variants.** Fixed in the checked-arithmetic and
   interpretation sections, with one detection point and owner each.

## Original open questions

1. Whether private variants alone or additional internal view types provide the
   cleanest pattern-matching access for `Eval`, `Rewrite`, `Fold`, and `Pp`.
2. Exact immutable supply representation and whether internal reducer IDs use
   `int` or a structured namespace/ordinal pair. **Resolved:** a flat `int`
   ordinal behind an abstract `Builder.state`, with `initial`/`run_from` so an
   adapter can resume rather than restart the supply, and an
   `exception Supply_exhausted` checked before the ordinal could wrap.
3. How deterministic `Expr.Source.t` allocation maps existing tensor IDs while
   preserving library independence.
4. Whether positive/nonnegative window dimensions deserve small private wrapper
   types inside `Intrinsic`.
5. Which current printer forms remain byte-for-byte stable and which become
   clearer after the module split.
6. Whether `Value.check` accepts optional size/depth limits or those remain a
   separate `Fold` query.
7. Exact structured error variants for checked arithmetic and source loading.

## Definition of done

The refactoring is complete when:

- `lib/expr` builds independently and exposes modules such as `Expr.Index` and
  `Expr.Value` without globally leaking generic module names;
- no source unit requires an `Expr_` filename prefix;
- the public AST contains only immutable functional expressions;
- every current expression form is representable without a Native type;
- all loads use opaque `Expr.Source.t` and a caller-provided load function;
- position/delta roles survive construction into the index AST;
- reducer construction, freshening, and substitution are capture-safe;
- index evaluation detects intermediate overflow before wrapping;
- constants compare, hash, and print consistently with their bits;
- `Round_f32` and compact intrinsic behavior are interpreted and tested;
- structural folds and rewrites share the same scoped representation;
- the reference interpreter agrees bitwise with the current behavior over the
  characterization suite;
- all current expression consumers use the new modules under `Expr`;
- `lib/native/expr.ml` and any temporary compatibility representation are gone;
- the remaining open choices are implementation details rather than unresolved
  language semantics.
