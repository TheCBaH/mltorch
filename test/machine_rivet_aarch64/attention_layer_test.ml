(* A granted attention layer on the real CPU: the layer's sequence, expanded
   and planned at binary32, published as AArch64 code and run natively,
   against [Attention_layer_block.eval] bit for bit. *)

module T = Machine_model_test.Model_test
module AT = Machine_model_test.Attention_layer_test
module LF = Native_test.Attention_layer_fixtures
module L = Attention_layer.Attention_layer
module H = Machine_rivet_aarch64.Rivet_a64_host
module Mm = Machine_model.Mir_model
module A = Attention_block
open Graph_ir

let verdict g site cx =
  let inputs = LF.inputs g in
  let want =
    match AT.layer_oracle site with
    | T.Custom f -> f g [] inputs
    | _ -> assert false
  in
  match H.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs) with
  | Error s -> Fmt.str "%a" Mm.Stop.pp s
  | Ok outs ->
      if
        List.for_all2
          (fun id t -> T.same_portable t (Tensor_id.Map.find id want))
          g.Graph.outputs outs
      then "bitwise"
      else "DIFFERS"

let case (precision, pipeline) =
  let c =
    LF.with_norm L.Norm.Kind.Layer L.Placement.Pre
      { (LF.all_flags LF.small) with LF.Config.rope = LF.Rope_form.Shared }
  in
  let g = LF.node c in
  let node = AT.layer_node g in
  let site = AT.site_of ~precision g in
  let layer id = if Node_id.equal id node then Some site else None in
  let b =
    Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
      (Loop_ir.Loop_bundle.build ~granted:(Node_id.equal node) g)
  in
  let name = A.Precision.name precision in
  match H.prepare ~layer ~pipeline b with
  | Error rs ->
      Fmt.pr "%s: refused: %a@." name
        Fmt.(list ~sep:(any "; ") Mm.Refusal.pp)
        rs
  | Ok host -> (
      match H.Context.create host ~constants:(fun _ -> None) with
      | Error s -> Fmt.pr "%s: %a@." name Mm.Stop.pp s
      | Ok cx ->
          Fmt.pr "%s (%d invocations): %s@." name (H.invocations host)
            (verdict g site cx))

let%expect_test "a granted attention layer runs natively as its descriptor" =
  List.iter case
    [
      (A.Precision.F32_ordered, T.fp32_ordered);
      (A.Precision.F32_fused, AT.fp32_contracted);
    ];
  [%expect
    {|
    f32_ordered (1 invocations): bitwise
    f32_fused (1 invocations): bitwise |}]
