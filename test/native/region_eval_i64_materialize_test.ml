(* [Region_eval.materialize_i64] (P3.4's own no-locals slice: the exact
   int64 pixel-shaped materialize -- one evaluation of the output expression
   per output coordinate through the real environment, reusing
   [Tensor.materialize_i64] rather than a per-key Region loop, since a pixel
   program has no locals to amortize). Proves the whole path end to end:
   real environment binding, coordinate-dependent output, exact values past
   float's 2^53 mantissa, and that the result is a genuinely distinct,
   independently-owned tensor per call (Gate 3's own exit criterion, "a
   subsequent invocation cannot mutate a returned tensor"). *)

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:1

let read t w =
  Err.or_raise
    ~pp_error:(fun fmt (`Wrong_format (Payload.Fmt f)) ->
      Fmt.pf fmt "wrong format %a" Payload.pp_fmt f)
    (Tensor.read_i64_at6 t (function
      | Axis.W -> w
      | Axis.N | Axis.T | Axis.D | Axis.H | Axis.C -> 0))

let%expect_test "Region_eval.materialize_i64: exact per-coordinate pixel output"
    =
  let env = Expr_bridge.env ~binding:(fun _ -> None) in
  (* [9_007_199_254_740_993L] (above float's exact-mantissa range) plus the
     coordinate's own W position, cast from the float index the same way
     [region_slots_i64_test.ml]'s vector case does. *)
  let output =
    Expr.Value.i64_add
      (Expr.Value.i64_const 9_007_199_254_740_993L)
      (Expr.Value.float_to_i64
         (Expr.Value.value_of_index
            (Expr.Index.of_position (Expr.Index.output Expr.Axis.W))))
  in
  let result =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_eval.materialize_i64 ~output_shape:shape ~env output)
  in
  Fmt.pr "%Ld,%Ld,%Ld@." (read result 0) (read result 1) (read result 2);
  [%expect {| 9007199254740993,9007199254740994,9007199254740995 |}];
  (* A second materialization from the same expression/env is a fresh
     tensor, not an alias: mutating it independently must not affect the
     first result. *)
  let result2 =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_eval.materialize_i64 ~output_shape:shape ~env output)
  in
  (match result2 with
  | Tensor.Tensor t -> (
      match t.Tensor.payload.Payload.fmt with
      | Payload.I64 -> t.Tensor.payload.Payload.data.{0} <- 0L
      | _ -> assert false));
  Fmt.pr "first still %Ld, second now %Ld@." (read result 0) (read result2 0);
  [%expect {| first still 9007199254740993, second now 0 |}]

let%expect_test
    "Region_eval.materialize_i64: an unbound load is a structured error, not a \
     silent default" =
  let env = Expr_bridge.env ~binding:(fun _ -> None) in
  let missing = Tensor_id.of_int 0 in
  let output =
    Expr.Value.i64_load
      (Expr_bridge.source_of_id missing)
      (Expr_bridge.coord_of_vec6 (Vec6.of_fn (fun _ -> Expr.Index.zero)))
  in
  (match Region_eval.materialize_i64 ~output_shape:shape ~env output with
  | Ok (_ : Tensor.packed) -> print_string "unexpectedly succeeded"
  | Error _ -> print_string "failed as expected");
  [%expect {| failed as expected |}]
