(* [Eval_direct]'s explicit Float-to-I64 cast for [Pointwise.To_copy]'s
   [Long] target on an F32 operand (the reverse direction of
   [to_copy_i64_test.ml]'s [Float]-target fix): [Graph_builder.to_copy]
   now threads an I64 output edge for [Long] regardless of the operand's own
   format, and [eval_direct.ml] routes an F32 operand through
   [Pointwise.To_copy.Compute_to_long], which applies
   [Direct.float_to_i64] -- [Value.i64_of_float]'s checked policy (truncate
   toward zero in [-2^63, 2^63), reject NaN/infinities/out-of-range) --
   rather than [Compute]'s bare [S.trunc], whose result would be read back
   as an ordinary float, never a genuine int64 payload cell. *)

open Graph_ir
open Graph_direct_fixtures

let build_to_copy_long x =
  let open Err.Syntax in
  let* g =
    lift_build
      Graph_builder.(
        build ~name:"to_copy_long" ~outputs:(fun r -> [ r ])
        @@
        let* x_ref = input ~shape:(s1c (Array.length x)) ~name:"x" () in
        to_copy ~name:"out" Pointwise.To_copy.Long x_ref)
  in
  let x_t =
    Tensor.materialize
      (s1c (Array.length x))
      (fun c -> x.(Dim.to_int (Vec6.get c Axis.C)))
  in
  let* env =
    lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x_t ]))
  in
  tensor_of_name g env "out"

(* Small, exactly-F32-representable values, matching [to_copy_i64_test.ml]'s
   own rationale for the reverse direction: the OPERAND here is F32, whose
   24-bit mantissa already loses precision far below 2^53, so "exact past
   2^53" is not a meaningful claim for this cast's input -- a mistake in the
   new arm (wrong operand, or a forgotten checked cast) shows up as a wrong
   small value instead. *)
let%expect_test
    "Direct graph: To_copy(Long) truncates an F32 operand toward zero" =
  let result = build_to_copy_long [| 3.7; -3.7; 1000000. |] in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {|
    out = tensor i64 [C=3] {3, -3, 1000000}
    |}]

(* Both values are exact powers of two, so F32's coarse mantissa at this
   magnitude does not round them before the checked cast runs -- a genuine
   test of [Value.i64_of_float]'s inclusive lower bound and a large in-range
   value, not merely small integers. *)
let%expect_test
    "Direct graph: To_copy(Long) accepts the exact -2^63 lower bound and a \
     large in-range value" =
  let result =
    build_to_copy_long [| -9223372036854775808.; 4611686018427387904. |]
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    out = tensor i64 [C=2] {-9223372036854775808, 4611686018427387904}
    |}]

(* An already-I64 operand needs no cast at all -- a plain identity copy.
   Also closes a real hazard, not just an "unsurveyed" gap: since
   [Graph_builder.to_copy] declares this node's output I64 unconditionally
   for the [Long] target, leaving an I64 operand to [Eval_direct]'s generic
   fallback (which always writes via [Schedule.evaluate] at F32) would
   silently produce an F32 payload under a declared I64 [Tensor_sig]. Past
   2^53 to prove this is a genuine typed read, not [Payload.get_float]'s own
   incidental [Int64.to_float] promotion round-tripped back losslessly by
   coincidence. *)
let%expect_test "Direct graph: To_copy(Long) on an I64 operand is an identity" =
  let open Err.Syntax in
  let result =
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"to_copy_long_i64" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 3) ~name:"x" ~fmt:Payload.(Fmt I64) () in
          to_copy ~name:"out" Pointwise.To_copy.Long x)
    in
    let x =
      Tensor.materialize_i64 (s1c 3) (fun c ->
          Int64.add 9007199254740993L
            (Int64.of_int (Dim.to_int (Vec6.get c Axis.C))))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    out = tensor i64 [C=3] {9007199254740993, 9007199254740994, 9007199254740995}
    |}]

(* [Err.Exn.E], the same convention [Direct.load_index]/[i64_load] already
   establish for a value-dependent runtime error -- not a returned [Error]
   and not a silently wrapped/truncated result. *)
let catch f =
  try
    ignore (f ());
    "no exception"
  with Err.Exn.E e -> Format.asprintf "raised: %a" Err.Exn.pp_kind e

let%expect_test
    "Direct graph: To_copy(Long) rejects NaN, infinities and out-of-range \
     magnitudes" =
  Fmt.pr "%s@." (catch (fun () -> build_to_copy_long [| Float.nan |]));
  Fmt.pr "%s@." (catch (fun () -> build_to_copy_long [| Float.infinity |]));
  Fmt.pr "%s@." (catch (fun () -> build_to_copy_long [| Float.neg_infinity |]));
  (* Exact power of two -- F32's coarse mantissa at this magnitude does not
     round it before the checked cast runs, so this genuinely exercises the
     policy's exclusive upper bound, not an artifact of F32 rounding. *)
  Fmt.pr "%s@."
    (catch (fun () -> build_to_copy_long [| 9223372036854775808. |]));
  (* Far enough past the boundary that F32's rounding (coarse as it is at
     this magnitude) cannot bring it back in range -- unlike a value one ULP
     past 2^63, which F32 would round back down to a representable in-range
     magnitude before this cast ever sees it. *)
  Fmt.pr "%s@." (catch (fun () -> build_to_copy_long [| 1e30 |]));
  [%expect
    {|
    raised: Float-to-I64 cast of NaN
    raised: Float-to-I64 cast of an infinite value
    raised: Float-to-I64 cast of an infinite value
    raised: Float-to-I64 cast of 0x1p+63, outside [-2^63, 2^63)
    raised: Float-to-I64 cast of 0x1.93e594p+99, outside [-2^63, 2^63)
    |}]
