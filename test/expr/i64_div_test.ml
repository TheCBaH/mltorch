(* [I64_div]: explicit truncating division (toward zero, not floor). A zero
   divisor and [-2^63 / -1] are structured errors, never a trap or a wrapped
   value, and operands evaluate left to right so the first error reported is the
   left one's. [Eval.value] returns floats, so an exact result is read back
   through a subtraction that leaves a small remainder. Runs under both
   backends (see test/expr/dune), including a nesting deep enough to hand over
   to the JS machine. *)

open Expr

let env =
  {
    Eval.Env.load = (fun _ _ -> assert false);
    load_index = (fun _ _ -> assert false);
  }

let output = Coord.of_fn (fun _ -> 0)
let pp_res = Core.Pretty.err_result ~ok:(Fmt.fmt "%.1f") ~error:Eval.pp_error
let eval e = Eval.value env ~output (Value.i64_to_float e)
let c = Value.i64_const
let div a b = Value.i64_div (c a) (c b)

let%expect_test "truncates toward zero for every sign combination" =
  List.iter
    (fun (a, b) -> Fmt.pr "%Ld/%Ld = %a@." a b pp_res (eval (div a b)))
    [ (7L, 2L); (-7L, 2L); (7L, -2L); (-7L, -2L); (1L, 2L); (-1L, 2L) ];
  [%expect
    {|
    7/2 = 3.0
    -7/2 = -3.0
    7/-2 = -3.0
    -7/-2 = 3.0
    1/2 = 0.0
    -1/2 = 0.0 |}]

let%expect_test "the quotient is exact past 2^53" =
  (* (2^53 + 1) / 1 - 2^53 = 1; a float quotient would round to 2^53 first. *)
  let big = 9_007_199_254_740_992L in
  Fmt.pr "%a@." pp_res
    (eval (Value.i64_sub (div (Int64.add big 1L) 1L) (c big)));
  (* 3 * 2^53 + 3 over 3 is 2^53 + 1, again a value a float cannot hold. *)
  Fmt.pr "%a@." pp_res
    (eval (Value.i64_sub (div (Int64.add (Int64.mul 3L big) 3L) 3L) (c big)));
  (* [min_int / 1] keeps [min_int]. *)
  Fmt.pr "%a@." pp_res
    (eval (Value.i64_sub (div Int64.min_int 1L) (c Int64.min_int)));
  [%expect {|
    1.0
    1.0
    0.0 |}]

let%expect_test "a zero divisor and min_int / -1 are structured errors" =
  Fmt.pr "%a@." pp_res (eval (div 5L 0L));
  Fmt.pr "%a@." pp_res (eval (div 0L 0L));
  Fmt.pr "%a@." pp_res (eval (div Int64.min_int (-1L)));
  (* [max_int / -1] fits, and [min_int / 2] is fine. *)
  Fmt.pr "%a@." pp_res
    (eval (Value.i64_add (div Int64.max_int (-1L)) (c Int64.max_int)));
  [%expect
    {|
    I64 division by zero
    I64 division by zero
    I64 division overflow: -2^63 / -1 does not fit
    0.0 |}]

let%expect_test "operands evaluate left to right, so the left error is first" =
  let nan_cast = Value.float_to_i64 (Value.const Float.nan) in
  Fmt.pr "%a@." pp_res (eval (Value.i64_div nan_cast (c 0L)));
  Fmt.pr "%a@." pp_res
    (eval (Value.i64_div (c 1L) (Value.i64_div (c 1L) (c 0L))));
  Fmt.pr "%a@." pp_res (eval (Value.i64_div (div 1L 0L) nan_cast));
  [%expect
    {|
    Float-to-I64 cast of NaN
    I64 division by zero
    I64 division by zero |}]

let%expect_test "a deep chain reports the error at its root, in both backends" =
  let rec chain n acc =
    if n = 0 then acc else chain (n - 1) (Value.i64_div acc (c 1L))
  in
  let deep = chain 1_000 (c 5L) in
  Fmt.pr "%a@." pp_res (eval deep);
  Fmt.pr "%a@." pp_res (eval (Value.i64_div deep (c 0L)));
  [%expect {|
    5.0
    I64 division by zero |}]
