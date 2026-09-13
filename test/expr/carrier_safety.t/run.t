What the type checker must reject about [Expr]'s carrier-indexed dtype GADT.

Each negative case is paired with a control that must still compile. Without
the controls a broken harness -- a wrong toplevel invocation, say -- would
reject everything and look like a pass. See test/expr/namespace_safety.t for
the same discipline applied to a different property (namespace leakage, AST
privacy) of this same library, and .ai/ for the dtype design record this
proves against.

  $ R=../../..
  $ check() { sh $R/test/expr/namespace_safety.sh $R/test/expr/expr_probe.exe "$1" "$2"; }

[Value.add] is pinned to [float t]: an [int64 t] operand -- built with
[i64_const], never a bare literal -- must not typecheck where a [float t] is
expected, and vice versa for [i64_add].

  $ check "add rejects an int64 operand" "ignore (Expr.Value.add (Expr.Value.const 1.) (Expr.Value.i64_const 1L))"
  add rejects an int64 operand: rejected
  $ check "add of two floats" "ignore (Expr.Value.add (Expr.Value.const 1.) (Expr.Value.const 2.))"
  add of two floats: COMPILES
  $ check "i64_add rejects a float operand" "ignore (Expr.Value.i64_add (Expr.Value.i64_const 1L) (Expr.Value.const 2.))"
  i64_add rejects a float operand: rejected
  $ check "i64_add of two int64s" "ignore (Expr.Value.i64_add (Expr.Value.i64_const 1L) (Expr.Value.i64_const 2L))"
  i64_add of two int64s: COMPILES

[Select]'s two branches share one carrier ([Select : bool_expr * 'a t * 'a t
-> 'a t]): mismatched branches must not typecheck, but the SAME carrier must
work generically at both float and int64, since [Select] is the one
constructor the design deliberately keeps carrier-polymorphic.

  $ check "select branches disagree on carrier" "ignore (Expr.Value.select (Expr.Bool.value_lt (Expr.Value.const 1.) (Expr.Value.const 2.)) (Expr.Value.const 1.) (Expr.Value.i64_const 1L))"
  select branches disagree on carrier: rejected
  $ check "select at float" "ignore (Expr.Value.select (Expr.Bool.value_lt (Expr.Value.const 1.) (Expr.Value.const 2.)) (Expr.Value.const 1.) (Expr.Value.const 2.))"
  select at float: COMPILES
  $ check "select at int64" "ignore (Expr.Value.select (Expr.Bool.i64_eq (Expr.Value.i64_const 1L) (Expr.Value.i64_const 1L)) (Expr.Value.i64_const 1L) (Expr.Value.i64_const 2L))"
  select at int64: COMPILES

[Bool.i64_eq]/[i64_lt] are pinned to [int64 t] operands; [Bool.value_lt] to
[float t] -- each rejects the other carrier.

  $ check "i64_eq rejects float operands" "ignore (Expr.Bool.i64_eq (Expr.Value.const 1.) (Expr.Value.const 2.))"
  i64_eq rejects float operands: rejected
  $ check "i64_eq of two int64s" "ignore (Expr.Bool.i64_eq (Expr.Value.i64_const 1L) (Expr.Value.i64_const 2L))"
  i64_eq of two int64s: COMPILES
  $ check "value_lt rejects int64 operands" "ignore (Expr.Bool.value_lt (Expr.Value.i64_const 1L) (Expr.Value.i64_const 2L))"
  value_lt rejects int64 operands: rejected
  $ check "value_lt of two floats" "ignore (Expr.Bool.value_lt (Expr.Value.const 1.) (Expr.Value.const 2.))"
  value_lt of two floats: COMPILES

The two casts, [i64_to_float : int64 t -> float t] and [float_to_i64 : float t
-> int64 t], go in opposite directions -- annotating a cast's result at its
OWN operand's carrier (instead of the carrier it actually produces) must not
typecheck.

  $ check "i64_to_float does not return int64" "let (_ : int64 Expr.Value.t) = Expr.Value.i64_to_float (Expr.Value.i64_const 1L) in ()"
  i64_to_float does not return int64: rejected
  $ check "i64_to_float returns float" "let (_ : float Expr.Value.t) = Expr.Value.i64_to_float (Expr.Value.i64_const 1L) in ()"
  i64_to_float returns float: COMPILES
  $ check "float_to_i64 does not return float" "let (_ : float Expr.Value.t) = Expr.Value.float_to_i64 (Expr.Value.const 1.) in ()"
  float_to_i64 does not return float: rejected
  $ check "float_to_i64 returns int64" "let (_ : int64 Expr.Value.t) = Expr.Value.float_to_i64 (Expr.Value.const 1.) in ()"
  float_to_i64 returns int64: COMPILES
