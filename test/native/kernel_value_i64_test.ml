(* [Kernel.Value_i64.t]: a standalone exact int64 Kernel value (no
   [Region_group.Ref], no cross-value dependency yet -- see the doc on
   [Kernel.Value_i64.t] and the implementation tracker's D09/P4.1 notes).
   Every admission rule gets a test that PRODUCES it, matching
   [kernel_test.ml]'s own discipline: a rule whose test cannot go red is not
   evidence. *)

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n
let tid = Tensor_id.of_int
let i64 = Payload.Fmt Payload.I64
let f32 = Payload.Fmt Payload.F32

let sg ?(fmt = i64) id shape =
  Tensor_sig.create ~id:(tid id) ~name:"" ~shape ~fmt ()

let pp_kernel = Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Kernel.pp_error

(* [start + i*step] over the C-axis output coordinate -- the closed-form
   exact Arange shape, matching [test/expr/check_i64_test.ml]'s own fixture. *)
let arange_shaped ~start ~step =
  Expr.Value.i64_add
    (Expr.Value.i64_const start)
    (Expr.Value.i64_mul
       (Expr.Value.i64_const step)
       (Expr.Value.float_to_i64
          (Expr.Value.value_of_index
             (Expr.Index.of_position (Expr.Index.output Expr.Axis.C)))))

let value_i64 ?(fmt = i64) id shape pixel =
  { Kernel.Value_i64.id = tid id; sg = sg ~fmt id shape; pixel }

let%expect_test "Kernel: a standalone int64 value is admitted" =
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~values_i64:[ value_i64 0 (s1c 3) (arange_shaped ~start:2L ~step:3L) ]
       ~inputs:[] ~values:[] ~outputs:[] ());
  [%expect {| ok |}]

let%expect_test "Kernel: an int64 value must actually be I64-formatted" =
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~values_i64:
         [ value_i64 ~fmt:f32 0 (s1c 3) (arange_shaped ~start:2L ~step:3L) ]
       ~inputs:[] ~values:[] ~outputs:[] ());
  [%expect {| t0: an int64 value must be i64 and unquantized, got f32 |}]

let%expect_test "Kernel: an int64 value may not read another value" =
  let source = Expr_bridge.source_of_id (tid 1) in
  let coord =
    Expr_bridge.coord_of_vec6 (Vec6.of_fn (fun _ -> Expr.Index.zero))
  in
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~values_i64:[ value_i64 0 (s1c 1) (Expr.Value.i64_load source coord) ]
       ~inputs:[] ~values:[] ~outputs:[] ());
  [%expect
    {| t0: an int64 value may not read another value yet, only a closed expression |}]

let%expect_test "Kernel: an int64 value's own body must be closed" =
  let stray, _ =
    Expr.Builder.run_from Expr.Builder.initial Expr.Builder.fresh_reduce
  in
  let unbound =
    Expr.Value.float_to_i64
      (Expr.Value.value_of_index
         (Expr.Index.of_position (Expr.Index.reduce stray)))
  in
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~values_i64:[ value_i64 0 (s1c 1) unbound ]
       ~inputs:[] ~values:[] ~outputs:[] ());
  [%expect {| t0: free reducer #0 |}]

let%expect_test "Kernel: int64 and float values share one id namespace" =
  Format.printf "%a@." pp_kernel
    (Kernel.create
       ~values_i64:[ value_i64 0 (s1c 1) (arange_shaped ~start:0L ~step:1L) ]
       ~inputs:
         [
           {
             Kernel.Input.id = tid 0;
             sg = sg ~fmt:f32 0 (s1c 1);
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:[] ~outputs:[] ());
  [%expect {| duplicate id t0 |}]

(* End to end: admission plus real materialization through [Kernel_eval.run],
   read back exactly -- past float's 2^53 mantissa, proving the int64 value
   never round-trips through the engine's f32 compute domain. *)
let%expect_test "Kernel_eval.run materializes a standalone int64 value exactly"
    =
  let shape = s 1 1 1 1 1 3 in
  let k =
    Err.or_raise ~pp_error:Kernel.pp_error
      (Kernel.create
         ~values_i64:
           [
             value_i64 0 shape
               (Expr.Value.i64_add
                  (Expr.Value.i64_const 9_007_199_254_740_993L)
                  (Expr.Value.float_to_i64
                     (Expr.Value.value_of_index
                        (Expr.Index.of_position (Expr.Index.output Expr.Axis.C)))));
           ]
         ~inputs:[] ~values:[] ~outputs:[] ())
  in
  let result =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run k ~bind:(fun _ -> None))
  in
  let tensor = Tensor_id.Map.find (tid 0) result in
  let read c =
    Err.or_raise
      ~pp_error:(fun fmt (`Wrong_format (Payload.Fmt f)) ->
        Fmt.pf fmt "wrong format %a" Payload.pp_fmt f)
      (Tensor.read_i64_at6 tensor (function
        | Axis.C -> c
        | Axis.N | Axis.T | Axis.D | Axis.H | Axis.W -> 0))
  in
  Fmt.pr "%Ld,%Ld,%Ld@." (read 0) (read 1) (read 2);
  [%expect {| 9007199254740993,9007199254740994,9007199254740995 |}]
