(* The benchmark's own description of each vector-vector operation, and what it
   refuses: a combination with no kernel is a benchmark error, reported before
   any assembler sees it. *)

module K = Peak_kernels

let%expect_test "every operation and what its accumulator holds" =
  List.iter
    (fun op ->
      Fmt.pr "%-15s widening=%b float=%b one=0x%x sum=%d" (K.Vv_op.mnemonic op)
        (K.Vv_op.widening op) (K.Vv_op.floating op) (K.Vv_op.one op)
        (K.Vv_op.sum op);
      List.iter
        (fun sew ->
          match K.Vv_op.accumulator op ~sew with
          | Some e -> Fmt.pr " e%d->%s" sew (K.Elem.name e)
          | None -> ())
        [ 8; 16; 32 ];
      Fmt.pr "@.")
    K.Vv_op.all;
  [%expect
    {|
    vadd.vv         widening=false float=false one=0x1 sum=2 e8->int8 e16->int16 e32->int32
    vfadd.vv        widening=false float=true one=0x3f800000 sum=2 e32->fp32
    vfmacc.vv       widening=false float=true one=0x3f800000 sum=64 e32->fp32
    vfmul.vv        widening=false float=true one=0x3f800000 sum=1 e32->fp32
    vfwmaccbf16.vv  widening=true float=true one=0x3f80 sum=64 e16->fp32
    vmacc.vv        widening=false float=false one=0x1 sum=64 e8->int8 e16->int16 e32->int32
    vmul.vv         widening=false float=false one=0x1 sum=1 e8->int8 e16->int16 e32->int32
    vwmacc.vv       widening=true float=false one=0x1 sum=64 e8->int16 e16->int32 |}]

let%expect_test "a combination with no kernel is refused" =
  let show ~sew ~lmul op =
    match K.elementwise ~sew ~lmul op with
    | Ok _ -> Fmt.pr "ok@."
    | Error e -> Fmt.pr "%s@." e
  in
  show ~sew:16 ~lmul:4 K.Vv_op.Vfadd;
  show ~sew:32 ~lmul:2 K.Vv_op.Vwmacc;
  show ~sew:8 ~lmul:8 K.Vv_op.Vwmacc;
  show ~sew:64 ~lmul:1 K.Vv_op.Vadd;
  show ~sew:32 ~lmul:4 K.Vv_op.Vfmacc;
  [%expect
    {|
    vfadd.vv has no kernel at e16,m4
    vwmacc.vv has no kernel at e32,m2
    vwmacc.vv has no kernel at e8,m8
    vadd.vv has no kernel at e64,m1
    ok |}]
