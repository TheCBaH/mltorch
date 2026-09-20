(* [Eval_direct]'s dtype-preserving tensor-tensor Add/Sub/Mul (a follow-up to
   Reshape/Permute/Mul_scalar's own exact-I64 dispatch): the default
   [Pointwise.{Add,Sub,Mul}.Compute(S).pixel] arm reads both operands through
   [SEMANTICS.load], which round-trips every format through [Payload.get_float]
   and writes back through the ordinary F32 pixel path -- exact for F32 but
   silently lossy above 2^53 for I64 operands, the same defect class Reshape/
   Permute had before their own fixes. [eval_direct.ml]'s new [Add]/[Sub]/[Mul]
   arms branch on BOTH operands' declared signature format and route a pair of
   I64 operands through [Pointwise.{Add,Sub,Mul}.Compute_i64], which reads via
   [Tensor.read_i64_at6] and combines via [Expr.Value.apply_i64_binary] --
   exact 64-bit arithmetic, never touching float. [Graph_builder.{add,sub,mul}]
   thread [~fmt]/[~quant] into the output edge only when BOTH operands are I64,
   matching the "I64-only" restriction [reshape]/[permute] converged on after
   the fmt-threading regression class -- unlike [Mul_scalar] (output stays float
   by design), a genuine
   I64 add/sub/mul's output IS I64, so the builder-side thread is required
   here, not merely a convenience. *)

open Graph_ir
open Graph_direct_fixtures

let big6 =
  Tensor.materialize_i64 (s1c 6) (fun c ->
      Int64.add 9_007_199_254_740_993L
        (Int64.of_int (Dim.to_int (Vec6.get c Axis.C))))

let small6 vals =
  Tensor.materialize_i64 (s1c 6) (fun c ->
      List.nth vals (Dim.to_int (Vec6.get c Axis.C)))

let run_binary ~shape ~op_of a b =
  let open Err.Syntax in
  let* g =
    lift_build
      Graph_builder.(
        build ~name:"binary_i64" ~outputs:(fun r -> [ r ])
        @@
        let* x = input ~shape ~name:"x" ~fmt:Payload.(Fmt I64) () in
        let* y = input ~shape ~name:"y" ~fmt:Payload.(Fmt I64) () in
        op_of x y)
  in
  let* env =
    lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
  in
  tensor_of_name g env "out"

let%expect_test "Direct graph: I64 add stays exact past 2^53" =
  let result =
    run_binary ~shape:(s1c 6)
      ~op_of:(fun x y -> Graph_builder.add ~name:"out" x y)
      big6
      (small6 [ 1L; 2L; 3L; 4L; 5L; 6L ])
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  (* 993+1, 994+2, 995+3, 996+4, 997+5, 998+6, all exact and distinct. *)
  [%expect
    {|
    out = tensor i64 [C=6] {9007199254740994, 9007199254740996, 9007199254740998, 9007199254741000, 9007199254741002, 9007199254741004}
    |}]

let%expect_test "Direct graph: I64 sub stays exact past 2^53" =
  let result =
    run_binary ~shape:(s1c 6)
      ~op_of:(fun x y -> Graph_builder.sub ~name:"out" x y)
      big6
      (small6 [ 1L; 2L; 3L; 4L; 5L; 6L ])
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  (* 993-1, 994-2, ..., 998-6, all exact and equal to 9007199254740992. *)
  [%expect
    {|
    out = tensor i64 [C=6] {9007199254740992, 9007199254740992, 9007199254740992, 9007199254740992, 9007199254740992, 9007199254740992}
    |}]

let%expect_test "Direct graph: I64 mul stays exact past 2^53" =
  let a = Tensor.materialize_i64 (s1c 1) (fun _ -> 100_000_003L)
  and b = Tensor.materialize_i64 (s1c 1) (fun _ -> 100_000_003L) in
  let result =
    run_binary ~shape:(s1c 1)
      ~op_of:(fun x y -> Graph_builder.mul ~name:"out" x y)
      a b
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  (* 100000003 * 100000003 = 10000000600000009, past 2^53 and odd -- a real
     double multiply of the same two (exactly representable) float operands
     would round this product to the nearest even value, so an exact match
     here is proof the path never went through float. *)
  [%expect {|
    out = tensor i64 [C=1] {10000000600000009}
    |}]

(* Proves [Graph_builder.add]'s own output edge declares I64 (not [op1]'s F32
   default), matching what [Eval_direct] actually produces -- the same class
   of regression [reshape]/[permute]'s own builder fix closed for those ops,
   checked here from the start via a chained add so it never ships with the
   gap in between. *)
let%expect_test
    "Direct graph: chained I64 adds -- the intermediate edge's own declared \
     signature stays I64" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"add_i64_chain" ~outputs:(fun (mid, out) -> [ mid; out ])
          @@
          let* x = input ~shape:(s1c 6) ~name:"x" ~fmt:Payload.(Fmt I64) () in
          let* y = input ~shape:(s1c 6) ~name:"y" ~fmt:Payload.(Fmt I64) () in
          let* mid = add x y in
          let* out = add mid y in
          return (mid, out))
    in
    let mid_id, out_id =
      match g.Graph.outputs with [ a; b ] -> (a, b) | _ -> assert false
    in
    let y = small6 [ 1L; 1L; 1L; 1L; 1L; 1L ] in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ big6; y ]))
    in
    let (Payload.Fmt mid_fmt) =
      (Tensor_id.Map.find mid_id g.Graph.tensors).Tensor_sig.fmt
    in
    let mid_tensor = Tensor_id.Map.find mid_id env in
    let out_tensor = Tensor_id.Map.find out_id env in
    Err.return (Payload.fmt_name mid_fmt, mid_tensor, out_tensor)
  in
  let pp_ok ppf (mid_fmt_name, mid_tensor, out_tensor) =
    Format.fprintf ppf "mid declared fmt = %s@.mid = %a@.out = %a" mid_fmt_name
      Tensor.pp mid_tensor Tensor.pp out_tensor
  in
  Format.printf "%a@." (pp_result pp_ok) result;
  [%expect
    {|
    mid declared fmt = i64
    mid = tensor i64 [C=6] {9007199254740994, 9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998, 9007199254740999}
    out = tensor i64 [C=6] {9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998, 9007199254740999, 9007199254741000}
    |}]

(* A mismatched I64/F32 pair fails at checked admission rather than
   silently computing through the default float path -- which would add in
   DOUBLE precision (Int64.to_float's own promotion) and round once to F32 at
   the very end, not provably the same result as real ATen's int64->float32
   promote-then-add (computed entirely at F32 precision, so a DIFFERENT
   rounding sequence -- double rounding is not generally equivalent to single
   rounding). No validated policy for this combination exists yet, so all
   three ops reject it explicitly instead of guessing. *)
let%expect_test "Direct graph: mixed I64/F32 add/sub/mul are rejected" =
  let open Err.Syntax in
  let run op_of =
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mixed_dtype" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 3) ~name:"x" ~fmt:Payload.(Fmt I64) () in
          let* y = input ~shape:(s1c 3) ~name:"y" () in
          op_of x y)
    in
    let x = small6 [ 1L; 2L; 3L; 0L; 0L; 0L ] in
    let y = Tensor.materialize (s1c 3) (fun _ -> 1.0) in
    lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x; y ]))
  in
  let pp_ok ppf (_ : Tensor.packed Tensor_id.Map.t) = Fmt.string ppf "ok" in
  Format.printf "%a@." (pp_result pp_ok)
    (run (fun x y -> Graph_builder.add ~name:"out" x y));
  Format.printf "%a@." (pp_result pp_ok)
    (run (fun x y -> Graph_builder.sub ~name:"out" x y));
  Format.printf "%a@." (pp_result pp_ok)
    (run (fun x y -> Graph_builder.mul ~name:"out" x y));
  [%expect
    {|
    add: unsupported mixed dtype, a=i64 b=f32
    sub: unsupported mixed dtype, a=i64 b=f32
    mul: unsupported mixed dtype, a=i64 b=f32
    |}]

(* Arithmetic on Bool stays rejected: before this check, a genuine
   [Payload.Bool] operand paired with an F32 one
   would fall through to the default float path, silently reading the Bool
   cells as 0./1. via [Payload.get_float] and adding them as ordinary floats
   -- no validated promotion policy exists for that either, the same reasoning
   the I64 case above already applies, just for a different
   unsupported pair. Checked BEFORE the I64 guard, so a Bool paired with I64
   (third case below) reports the Bool reason specifically, not the
   unrelated I64-mixing one. *)
let%expect_test "Direct graph: arithmetic on a Bool operand is rejected" =
  let open Err.Syntax in
  let run ~y_fmt op_of =
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"bool_arithmetic" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 3) ~name:"x" ~fmt:Payload.(Fmt Bool) () in
          let* y = input ~shape:(s1c 3) ~name:"y" ~fmt:y_fmt () in
          op_of x y)
    in
    let x = Tensor.materialize_bool (s1c 3) (fun _ -> true) in
    let y =
      match y_fmt with
      | Payload.Fmt Payload.I64 -> Tensor.materialize_i64 (s1c 3) (fun _ -> 1L)
      | _ -> Tensor.materialize (s1c 3) (fun _ -> 1.0)
    in
    lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x; y ]))
  in
  let pp_ok ppf (_ : Tensor.packed Tensor_id.Map.t) = Fmt.string ppf "ok" in
  let run_all ~y_fmt =
    Format.printf "%a@." (pp_result pp_ok)
      (run ~y_fmt (fun x y -> Graph_builder.add ~name:"out" x y));
    Format.printf "%a@." (pp_result pp_ok)
      (run ~y_fmt (fun x y -> Graph_builder.sub ~name:"out" x y));
    Format.printf "%a@." (pp_result pp_ok)
      (run ~y_fmt (fun x y -> Graph_builder.mul ~name:"out" x y))
  in
  run_all ~y_fmt:Payload.(Fmt F32);
  run_all ~y_fmt:Payload.(Fmt I64);
  [%expect
    {|
    add: arithmetic on a Bool operand is not supported, a=bool b=f32
    sub: arithmetic on a Bool operand is not supported, a=bool b=f32
    mul: arithmetic on a Bool operand is not supported, a=bool b=f32
    add: arithmetic on a Bool operand is not supported, a=bool b=i64
    sub: arithmetic on a Bool operand is not supported, a=bool b=i64
    mul: arithmetic on a Bool operand is not supported, a=bool b=i64
    |}]
