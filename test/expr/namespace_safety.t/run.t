What the type checker must reject about [Expr]'s public surface.

Each negative case is paired with a control that must still compile. Without the
controls a broken harness — a wrong toplevel invocation, say — would reject
everything and look like a pass. See test/native/version_safety.t for the same
discipline applied to version-indexed ids.

  $ R=../../..
  $ check() { sh $R/test/expr/namespace_safety.sh $R/test/expr/expr_probe.exe "$1" "$2"; }

The namespace does not leak. [lib/expr] is wrapped, so its units are reachable
only under [Expr]. This is the whole reason the language moved out of
[lib/native], which is (wrapped false) and would publish a global [Axis] and
[Value] into every consumer.

  $ check "bare Axis" "ignore (Axis.N)"
  bare Axis: rejected
  $ check "bare Coord" "ignore (Coord.of_fn (fun _ -> 0))"
  bare Coord: rejected
  $ check "Axis under Expr" "ignore (Expr.Axis.N)"
  Axis under Expr: COMPILES
  $ check "Coord under Expr" "ignore (Expr.Coord.of_fn (fun _ -> 0))"
  Coord under Expr: COMPILES

An explicit main module means dune synthesises no aliases, so each leaf has to
be re-exported by hand in expr.ml AND expr.mli. Dropping one of those lines is
invisible inside the library and only shows up here.

  $ check "Role is re-exported" "ignore (Expr.Index.zero : Expr.Role.Position.t Expr.Index.t)"
  Role is re-exported: COMPILES
  $ check "Source is re-exported" "ignore (Expr.Source.create 0)"
  Source is re-exported: COMPILES

The AST is private: external code MATCHES the variants — grounding and the
transform verifier both need to — but cannot CONSTRUCT them. Construction goes
through the smart constructors, which is what stops a malformed reducer scope or
an invalid divisor being built casually.

  $ check "construct a Value" "ignore (Expr.Value.Const 1.)"
  construct a Value: rejected
  $ check "construct a Bool" "ignore (Expr.Bool.Value_lt (Expr.Value.const 1., Expr.Value.const 2.))"
  construct a Bool: rejected
  $ check "construct an Index" "ignore (Expr.Index.Zero)"
  construct an Index: rejected
  $ check "smart constructor instead" "ignore (Expr.Value.const 1.)"
  smart constructor instead: COMPILES
  $ check "match a Value" "ignore (match Expr.Value.const 1. with Expr.Value.Const x -> x | _ -> 0.)"
  match a Value: COMPILES

A reducer variable has no public constructor at all: only [Builder] mints one,
which is what makes the scope discipline enforceable rather than advisory.

  $ check "mint a Reduce_var directly" "ignore (Expr.Index.reduce 0)"
  mint a Reduce_var directly: rejected
  $ check "mint one through Builder" "ignore (Expr.Builder.run Expr.Builder.fresh_reduce)"
  mint one through Builder: COMPILES

The reduction record is private too, so its bounds and body cannot be rebuilt
around a foreign binder.

  $ check "build a Reduction record" "ignore { Expr.Reduction.kind = Expr.Reduction.Sum; var = Expr.Builder.run Expr.Builder.fresh_reduce; lo = Expr.Index.zero; hi = Expr.Index.const 1; body = Expr.Value.const 0. }"
  build a Reduction record: rejected
  $ check "read a Reduction field" "ignore (fun (r : Expr.Reduction.t) -> r.Expr.Reduction.body)"
  read a Reduction field: COMPILES
