(* [I64_sum]: the typed int64 reduction. The accumulator is an exact int64
   (modular two's-complement), never a float, and the binder is scoped like a
   float [Reduce]'s. [Eval.value] returns floats, so each result is read back
   through a subtraction that leaves a small exact remainder: a float
   accumulator would print the wrong number, not just a rounded one. Runs under
   both backends (see test/expr/dune), including a nesting deep enough to hand
   over to the JS machine. *)

open Expr

let env =
  {
    Eval.Env.load = (fun _ _ -> assert false);
    load_index = (fun _ _ -> assert false);
  }

let output = Coord.of_fn (fun _ -> 0)
let pp_res = Core.Pretty.err_result ~ok:(Fmt.fmt "%.1f") ~error:Eval.pp_error
let eval e = Eval.value env ~output (Value.i64_to_float e)
let pos_delta r = Index.of_position r
let big = 9_007_199_254_740_992L (* 2^53 *)

let sum ~lo ~hi body =
  Builder.run
    (Builder.i64_sum
       ~lo:(Index.assume_position (Index.const lo))
       ~hi:(Index.const hi)
       (fun r -> Builder.return (body r)))

let%expect_test "the accumulator is exact int64, past 2^53" =
  (* 2^53 + 0, 2^53 + 1, 2^53 + 2 sum to 3 * 2^53 + 3; a float accumulator
     rounds 2^53 + 1 back to 2^53 and would leave 0 or 2 here, not 3. *)
  let s =
    sum ~lo:0 ~hi:3 (fun r ->
        Value.i64_add (Value.i64_const big) (Value.i64_of_index (pos_delta r)))
  in
  Fmt.pr "%a@." Pp.value_i64 s;
  [%expect {| i64_sum(r1=0..3: (9007199254740992 + i64_of_index(r1))) |}];
  Fmt.pr "%a@." pp_res
    (eval (Value.i64_sub s (Value.i64_const (Int64.mul 3L big))));
  [%expect {| 3.0 |}]

let%expect_test "an empty or reversed range is 0" =
  let s lo hi = sum ~lo ~hi (fun _ -> Value.i64_const 7L) in
  Fmt.pr "%a %a@." pp_res (eval (s 0 0)) pp_res (eval (s 5 2));
  [%expect {| 0.0 0.0 |}]

let%expect_test "the sum wraps modulo 2^64 like I64_binary" =
  let two_max = sum ~lo:0 ~hi:2 (fun _ -> Value.i64_const Int64.max_int) in
  Fmt.pr "%a@." pp_res (eval two_max);
  [%expect {| -2.0 |}]

let%expect_test "nested binders each see their own index" =
  (* sum over i<3 of sum over j<4 of i*j = (0+1+2) * (0+1+2+3) = 18. *)
  let s =
    Builder.run
      (Builder.i64_sum ~lo:Index.zero ~hi:(Index.const 3) (fun i ->
           Builder.i64_sum ~lo:Index.zero ~hi:(Index.const 4) (fun j ->
               Builder.return
                 (Value.i64_mul
                    (Value.i64_of_index (pos_delta i))
                    (Value.i64_of_index (pos_delta j))))))
  in
  Fmt.pr "%a@." Pp.value_i64 s;
  [%expect
    {| i64_sum(r1=0..3: i64_sum(r2=0..4: (i64_of_index(r1) * i64_of_index(r2)))) |}];
  Fmt.pr "%a@." pp_res (eval s);
  [%expect {| 18.0 |}]

let%expect_test "structural identity ignores binder names, not bounds or body" =
  let mk ~hi ~step =
    sum ~lo:0 ~hi (fun r ->
        Value.i64_add (Value.i64_of_index (pos_delta r)) (Value.i64_const step))
  in
  let f = Value.i64_to_float in
  let same a b =
    Value.compare (f a) (f b) = 0 && Value.hash (f a) = Value.hash (f b)
  in
  Fmt.pr "same: %b, other bound: %b, other body: %b@."
    (same (mk ~hi:3 ~step:1L) (mk ~hi:3 ~step:1L))
    (same (mk ~hi:3 ~step:1L) (mk ~hi:4 ~step:1L))
    (same (mk ~hi:3 ~step:1L) (mk ~hi:3 ~step:2L));
  [%expect {| same: true, other bound: false, other body: false |}]

let%expect_test "the reducer is a binder, reported by Fold.binders" =
  let s = sum ~lo:0 ~hi:3 (fun _ -> Value.i64_const 1L) in
  Fmt.pr "binders: %d, free reducers: %d@."
    (List.length (Fold.binders (Value.i64_to_float s)))
    (Reduce_var.Set.cardinal (Fold.free_reducers (Value.i64_to_float s)));
  [%expect {| binders: 1, free reducers: 0 |}]

(* 1000 nested additions around one sum: past the JS evaluator's cutoff (50),
   so it hands the remaining subtree to its machine mid-tree and the machine's
   own int64 sum frame produces the answer there. (Plain [I64_binary] nesting
   overflows the JS stack a few thousand deep, with or without a sum: the
   pre-existing, documented limit, so this stays at 1000.) *)
let%expect_test "a sum under a very deep expression, on both backends" =
  let core =
    sum ~lo:0 ~hi:4 (fun r -> Value.i64_of_index (pos_delta r))
    (* 0+1+2+3 *)
  in
  let rec wrap n e =
    if n = 0 then e else wrap (n - 1) (Value.i64_add (Value.i64_const 1L) e)
  in
  Fmt.pr "%a@." pp_res (eval (wrap 1_000 core));
  [%expect {| 1006.0 |}]
