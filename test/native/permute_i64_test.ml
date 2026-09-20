(* [Eval_direct]'s dtype-preserving [Permute] dispatch (a follow-up to Reshape's
   own exact-I64 dispatch): the default [Permute.Compute(S).pixel]
   arm reads through [SEMANTICS.load], which round-trips every format
   through [Payload.get_float] and is therefore silently lossy for an exact
   I64 value above float's 2^53 mantissa -- and, unlike Reshape at the time,
   [Permute] had NO dispatch arm of its own at all, so this
   was live for any I64 input, not merely undocumented. [eval_direct.ml]'s
   new [Permute] arm branches on the operand's declared signature format and
   routes an I64 source through [Permute.Permute.Compute_i64], which reads
   via [Tensor.read_i64_at6] instead -- never touching float.
   [Graph_builder.permute] also threads [~fmt]/[~quant] from its operand,
   landing both halves of the fix together this time (unlike [reshape],
   whose builder-side gap was found and fixed later). *)

open Graph_ir
open Graph_direct_fixtures

let swap_wc =
  [
    (Axis.N, Axis.N);
    (Axis.T, Axis.T);
    (Axis.D, Axis.D);
    (Axis.H, Axis.H);
    (Axis.W, Axis.C);
    (Axis.C, Axis.W);
  ]

let%expect_test "Direct graph: I64 permute [W=2 C=3] swap stays exact past 2^53"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"permute_i64" ~outputs:(fun r -> [ r ])
          @@
          let* x =
            input ~shape:(s 1 1 1 1 2 3) ~name:"x" ~fmt:Payload.(Fmt I64) ()
          in
          permute ~name:"out" swap_wc x)
    in
    let x =
      Tensor.materialize_i64 (s 1 1 1 1 2 3) (fun c ->
          Int64.add 9_007_199_254_740_993L
            (Int64.of_int
               ((Dim.to_int (Vec6.get c Axis.W) * 3)
               + Dim.to_int (Vec6.get c Axis.C))))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  (* transpose of [[993,994,995],[996,997,998]] is [[993,996],[994,997],
     [995,998]] -- row-major 993,996,994,997,995,998, all distinct and
     above 2^53. *)
  [%expect
    {|
    out = tensor i64 [W=3 C=2] {9007199254740993, 9007199254740996, 9007199254740994, 9007199254740997, 9007199254740995, 9007199254740998}
    |}]

(* Chained permutes: proves [Graph_builder.permute]'s own output edge
   declares I64 (not [op1]'s F32 default), matching what [Eval_direct]
   actually produces -- the same class of regression [reshape]'s own builder
   fix closed for Reshape, checked here from the start so Permute never
   ships with the gap in between. *)
let%expect_test
    "Direct graph: chained I64 permutes -- the intermediate edge's own \
     declared signature stays I64" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"permute_i64_chain" ~outputs:(fun (mid, out) ->
              [ mid; out ])
          @@
          let* x =
            input ~shape:(s 1 1 1 1 2 3) ~name:"x" ~fmt:Payload.(Fmt I64) ()
          in
          let* mid = permute swap_wc x in
          let* out = permute swap_wc mid in
          return (mid, out))
    in
    let mid_id, out_id =
      match g.Graph.outputs with [ a; b ] -> (a, b) | _ -> assert false
    in
    let x =
      Tensor.materialize_i64 (s 1 1 1 1 2 3) (fun c ->
          Int64.add 9_007_199_254_740_993L
            (Int64.of_int
               ((Dim.to_int (Vec6.get c Axis.W) * 3)
               + Dim.to_int (Vec6.get c Axis.C))))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
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
  (* Two W<->C swaps compose to the identity, so [out] equals [x]. *)
  [%expect
    {|
    mid declared fmt = i64
    mid = tensor i64 [W=3 C=2] {9007199254740993, 9007199254740996, 9007199254740994, 9007199254740997, 9007199254740995, 9007199254740998}
    out = tensor i64 [W=2 C=3] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998}
    |}]
