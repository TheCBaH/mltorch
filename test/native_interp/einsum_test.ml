(* `torch.ops.aten.einsum.default`, restricted to [Aten_shape.Einsum]'s two
   evidenced equations (`.ai/einsum_design.md`): exactly two operands, ATen
   rank 5 then 3. Metadata-only decode (unlike [Op_bridge]'s dispatch, tested
   separately in test/native_bridge/einsum_dispatch_test.ml): no numeric
   values here, since [Native_interp] never materializes one. *)

open Programs

let einsum_node ~equation ~tensors ~out =
  jstr
    {|{"target":"torch.ops.aten.einsum.default","inputs":[{"name":"equation","arg":{"as_string":"%s"},"kind":1},{"name":"tensors","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    equation (as_tensors tensors) (as_tensor out)

let%expect_test "einsum.default: Shared_h and Shared_w both lower" =
  let prog equation =
    program ~x_sizes:[ 1; 1; 2; 2; 2 ] ~params:[ "other3" ]
      ~extra_tensor_values:[ ("other3", tensor_meta [ 2; 1; 2 ]) ]
      ~nodes:[ einsum_node ~equation ~tensors:[ "x"; "other3" ] ~out:"y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "Shared_h:" (prog "byhwc,hkc->byhwk");
  show "Shared_w:" (prog "byhwc,wkc->byhwk");
  [%expect
    {|
    Shared_h:                  lowered, nodes=2
    Shared_w:                  lowered, nodes=4
    |}]

let%expect_test "einsum.default: rejects an unrecognized equation" =
  let prog =
    program ~x_sizes:[ 1; 1; 2; 2; 2 ] ~params:[ "other3" ]
      ~extra_tensor_values:[ ("other3", tensor_meta [ 2; 1; 2 ]) ]
      ~nodes:
        [
          einsum_node ~equation:"byhwc,hc->byhw" ~tensors:[ "x"; "other3" ]
            ~out:"y";
        ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "unrecognized:" prog;
  [%expect
    {|
    unrecognized:              malformed PT2 graph: einsum.default: unsupported equation "byhwc,hc->byhw" with operand ranks 5, 3 (only "byhwc,hkc->byhwk"/"byhwc,wkc->byhwk", each with a rank-5 self and rank-3 other, are recognized)
    |}]

let%expect_test "einsum.default: rejects a wrong number of operands" =
  let prog =
    program ~x_sizes:[ 1; 1; 2; 2; 2 ]
      ~nodes:
        [ einsum_node ~equation:"byhwc,hkc->byhwk" ~tensors:[ "x" ] ~out:"y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "wrong count:" prog;
  [%expect
    {|
    wrong count:               malformed PT2 graph: einsum.default: unsupported equation "byhwc,hkc->byhwk" with operand ranks 5 (only "byhwc,hkc->byhwk"/"byhwc,wkc->byhwk", each with a rank-5 self and rank-3 other, are recognized)
    |}]

let%expect_test "einsum.default: rejects a wrong operand rank" =
  let prog =
    program ~x_sizes:[ 1; 1; 2; 2; 2 ] ~params:[ "other4" ]
      ~extra_tensor_values:[ ("other4", tensor_meta [ 1; 2; 1; 2 ]) ]
      ~nodes:
        [
          einsum_node ~equation:"byhwc,hkc->byhwk" ~tensors:[ "x"; "other4" ]
            ~out:"y";
        ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "wrong rank:" prog;
  [%expect
    {|
    wrong rank:                malformed PT2 graph: einsum.default: unsupported equation "byhwc,hkc->byhwk" with operand ranks 5, 4 (only "byhwc,hkc->byhwk"/"byhwc,wkc->byhwk", each with a rank-5 self and rank-3 other, are recognized)
    |}]
