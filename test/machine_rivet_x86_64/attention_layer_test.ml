(* A granted attention layer as x86-64 processes (on an x86-64 CPU, under
   qemu-user elsewhere): the layer's sequence, expanded and planned at
   binary32, published as x86-64 code and run, against
   [Attention_layer_block.eval] bit for bit. *)

module T = Machine_model_test.Model_test
module AT = Machine_model_test.Attention_layer_test
module LF = Native_test.Attention_layer_fixtures
module L = Attention_layer.Attention_layer
module Rt = Machine_rivet_x86_64.Rivet_x64_route
module A = Attention_block
open Graph_ir

let route = Rt.route ()

let case (precision, pipeline) (name, c) =
  let g = LF.node c in
  let node = AT.layer_node g in
  let site = AT.site_of ~precision g in
  let values = LF.inputs g in
  T.check ~route
    ~layer:(fun id -> if Node_id.equal id node then Some site else None)
    ~pipeline ~oracle:(AT.layer_oracle site) ~same:T.same_portable ~calls:1
    ~input:(fun ~salt:_ _ (sg : Tensor_sig.t) ->
      List.assoc sg.Tensor_sig.id values)
    (Fmt.str "%s, %s" (A.Precision.name precision) name)
    (fun () -> g)

let%expect_test "a granted attention layer runs as x86-64 processes" =
  List.iter
    (fun p ->
      List.iter (case p)
        [
          ("bare", LF.small);
          ( "pre layer, all flags, rope",
            LF.with_norm L.Norm.Kind.Layer L.Placement.Pre
              {
                (LF.all_flags LF.small) with
                LF.Config.rope = LF.Rope_form.Shared;
              } );
        ])
    [
      (A.Precision.F32_ordered, T.fp32_ordered);
      (A.Precision.F32_fused, AT.fp32_contracted);
    ];
  [%expect
    {|
    f32_ordered, bare (1 invocations): bitwise
    f32_ordered, pre layer, all flags, rope (1 invocations): bitwise
    f32_fused, bare (1 invocations): bitwise
    f32_fused, pre layer, all flags, rope (1 invocations): bitwise |}]
