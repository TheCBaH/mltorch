(* EdgeNeXt's Float -> Bool cast -> bitwise-not -> Float32 cumsum path, with real
   ATen as the oracle at every step. ATen is driven
   through [Aten_c.Aten_operations] directly, the same way
   test/aten_tensor_test.ml does, because the schema-driven dispatch decoder
   refuses any present [dtype] argument (see to_copy_test.ml). The input carries
   the values that separate the policies: NaN and infinity (both true), -0. and
   0. (both false), so a nonzero test written as [x > 0] or [x <> 0. && x = x]
   disagrees with ATen here. Each ATen tensor is compared with what Native
   Direct writes, including the Bool storage of the mask itself. *)

open Helpers
module O = Aten_c.Aten_operations

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:4
let values = [ 0.; Float.nan; -0.; 2.; 3.; Float.infinity; 0.; -5. ]

let native_graph () =
  Graph_builder.build ~name:"edgenext_bool"
    ~outputs:(fun (m, i, w, c) -> [ m; i; w; c ])
    Graph_builder.(
      let* x = input ~shape ~name:"x" () in
      let* mask = to_copy Pointwise.To_copy.Bool x in
      let* inv = bitwise_not mask in
      let* cum_w = cumsum { Reduce.Cumsum.axis = Axis.W } inv in
      let* cum_c = cumsum { Reduce.Cumsum.axis = Axis.C } inv in
      return (mask, inv, cum_w, cum_c))
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test "native Bool mask, logical-not and Float32 cumsums equal ATen's"
    =
  let g = native_graph () in
  let x =
    Tensor.materialize shape (fun c ->
        List.nth values
          ((Dim.to_int (Vec6.get c Axis.W) * 4) + Dim.to_int (Vec6.get c Axis.C)))
  in
  let env =
    Eval_direct.run g ~inputs:(List.combine g.Graph_ir.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  let native id = Graph_ir.Tensor_id.Map.find id env in
  let mask_id, inv_id, cw_id, cc_id =
    match g.Graph_ir.Graph.outputs with
    | [ a; b; c; d ] -> (a, b, c, d)
    | _ -> assert false
  in
  let aten_x = float_tensor [ 2; 4 ] values in
  let mask =
    Aten_tensor.manage (O.to_dtype aten_x Stype.Bool false false None)
  in
  let dispatch target bindings inputs =
    let env = List.fold_left (fun m (k, t) -> Sm.add k t m) Sm.empty bindings in
    let node =
      PT.Node.make target inputs [ targ "out0" ] Sm.empty None (Some "test")
    in
    match Interp_dispatch.dispatch env node with
    | Ok out -> Sm.find "out0" out
    | Error e ->
        failwith
          (Format.asprintf "%a" Interp_verify.pp_interp_error (Err.Error.kind e))
  in
  (* The ATen interpreter binds [logical_not], not [bitwise_not]; on a Bool
     tensor the two are the same function, which is the only case the exported
     graph applies it to. *)
  let inv =
    dispatch "torch.ops.aten.logical_not.default"
      [ ("self", mask) ]
      [ in_tensor "self" ]
  in
  (* ATen's cumsum of a bool tensor is int64; the explicit dtype=float32 of the
     real graph is that count cast to float, exact for counts this small. *)
  let cum dim =
    let c =
      dispatch "torch.ops.aten.cumsum.default"
        [ ("self", inv) ]
        [ in_tensor "self"; in_int "dim" dim ]
    in
    Aten_tensor.manage (O.to_dtype c Stype.Float false false None)
  in
  let check name aten native =
    Format.printf "%-6s %a@." name pp_result
      (Verify.compare_tensors ~atol:0. ~output:name aten native)
  in
  check "mask" mask (native mask_id);
  check "inv" inv (native inv_id);
  check "cum_w" (cum 0) (native cw_id);
  check "cum_c" (cum 1) (native cc_id);
  Format.printf "native mask: %a@." Tensor.pp (native mask_id);
  [%expect
    {|
    mask   Ok
    inv    Ok
    cum_w  Ok
    cum_c  Ok
    native mask: tensor bool [W=2 C=4] {0, 1, 0, 1, 1, 1, 0, 1} |}]
