(* `torch.ops.aten.einsum.default`: [Op_bridge]'s dispatch arm, restricted to
   [Aten_shape.Einsum]'s two evidenced equations -- exactly two operands, ATen
   rank 5 then 3. No real ATen call is made ([Tensor[]] plus a string argument
   has no [lib/aten_gen] C-shim support), so coverage is hand-derived, the
   same choice already made for `addcmul.default`/`group_norm.default`/
   `index.Tensor`. The [Shared_h]/[Shared_w] math itself is verified
   independently in `test/native/graph_direct_einsum_test.ml`; this file
   exercises the IMPORTER'S OWN decode and rejection surface -- both tests
   below share [self]'s [h] and [w] extents (both 2) with [other]'s own
   leading extent, so the SAME pair of tensors is a genuinely valid einsum
   call under EITHER equation (never a shape real ATen would reject), the
   same discipline the numeric fixtures elsewhere in this session hold to. *)

open Helpers

(* self[b=0,y=0,h,w,c]: h=0: w=0:[1,2], w=1:[3,4]; h=1: w=0:[9,10], w=1:[11,12].
   other[dim0,k=0,c]: dim0=0:[5,6], dim0=1:[7,8]. *)
let self5 = float_tensor [ 1; 1; 2; 2; 2 ] [ 1.; 2.; 3.; 4.; 9.; 10.; 11.; 12. ]
let other3 = float_tensor [ 2; 1; 2 ] [ 5.; 6.; 7.; 8. ]

(* [Shared_h]: other's leading extent is read as [h] -- output[h,w,k=0] =
   sum_c self[h,w,c]*other[h,k=0,c]:
   h=0,w=0: [1,2].[5,6]=17   h=0,w=1: [3,4].[5,6]=39
   h=1,w=0: [9,10].[7,8]=143 h=1,w=1: [11,12].[7,8]=173 *)
let%expect_test "dispatch: einsum.default Shared_h" =
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.einsum.default"
    ~bindings:[ ("self5", self5); ("other3", other3) ]
    ~inputs:
      [
        in_string "equation" "byhwc,hkc->byhwk";
        in_tensors "tensors" [ "self5"; "other3" ];
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=2 C=2] ->[n1], t1 f32 [H=2 W=1 C=2] ->[n0]]
    nodes:
      n0: [t2 f32 [H=2 W=2 C=1] ->[n1]] = permute x=t1 perm=[W<-C, C<-W]
      n1: [t3 f32 [H=2 W=2 C=1]] = batched_matmul input=t0 mat2=t2 <-n0
    outputs: [t3 f32 [H=2 W=2 C=1] <-n1]
    tensor f32 [H=2 W=2 C=1] {17, 39, 143, 173} |}]

(* [Shared_w]: the SAME two tensors, but other's leading extent is now read
   as [w] instead -- output[h,w,k=0] = sum_c self[h,w,c]*other[w,k=0,c]:
   h=0,w=0: [1,2].[5,6]=17   h=0,w=1: [3,4].[7,8]=53
   h=1,w=0: [9,10].[5,6]=105 h=1,w=1: [11,12].[7,8]=173
   Three of the four entries differ from [Shared_h]'s own result on the
   IDENTICAL inputs (the fourth, h=1/w=1, coincides because position 1 feeds
   the same [other] row under both readings) -- proof the plan choice, not
   the data, decides which axis is contracted against which. *)
let%expect_test "dispatch: einsum.default Shared_w" =
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.einsum.default"
    ~bindings:[ ("self5", self5); ("other3", other3) ]
    ~inputs:
      [
        in_string "equation" "byhwc,wkc->byhwk";
        in_tensors "tensors" [ "self5"; "other3" ];
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=2 C=2] ->[n0], t1 f32 [H=2 W=1 C=2] ->[n1]]
    nodes:
      n0: [t2 f32 [H=2 W=2 C=2] ->[n2]] = permute x=t0 perm=[H<-W, W<-H]
      n1: [t3 f32 [H=2 W=2 C=1] ->[n2]] = permute x=t1 perm=[W<-C, C<-W]
      n2: [t4 f32 [H=2 W=2 C=1] ->[n3]] =
        batched_matmul input=t2 <-n0 mat2=t3 <-n1
      n3: [t5 f32 [H=2 W=2 C=1]] = permute x=t4 <-n2 perm=[H<-W, W<-H]
    outputs: [t5 f32 [H=2 W=2 C=1] <-n3]
    tensor f32 [H=2 W=2 C=1] {17, 53, 105, 173} |}]

let%expect_test "dispatch: einsum.default rejects an unrecognized equation" =
  dispatch_print ~target:"torch.ops.aten.einsum.default"
    ~bindings:[ ("self5", self5); ("other3", other3) ]
    ~inputs:
      [
        in_string "equation" "byhwc,hc->byhw";
        in_tensors "tensors" [ "self5"; "other3" ];
      ]
    ~noutputs:1;
  [%expect
    {| error: einsum.default: unsupported equation "byhwc,hc->byhw" with operand ranks [5, 3] (only "byhwc,hkc->byhwk"/"byhwc,wkc->byhwk", each with a rank-5 self and rank-3 other, are recognized) |}]

let%expect_test "dispatch: einsum.default rejects a wrong number of operands" =
  dispatch_print ~target:"torch.ops.aten.einsum.default"
    ~bindings:[ ("self5", self5) ]
    ~inputs:
      [
        in_string "equation" "byhwc,hkc->byhwk";
        in_tensors "tensors" [ "self5" ];
      ]
    ~noutputs:1;
  [%expect
    {| error: einsum.default: unsupported equation "byhwc,hkc->byhwk" with operand ranks [5] (only "byhwc,hkc->byhwk"/"byhwc,wkc->byhwk", each with a rank-5 self and rank-3 other, are recognized) |}]

let%expect_test "dispatch: einsum.default rejects a wrong operand rank" =
  let other4 = float_tensor [ 1; 2; 1; 2 ] [ 5.; 6.; 7.; 8. ] in
  dispatch_print ~target:"torch.ops.aten.einsum.default"
    ~bindings:[ ("self5", self5); ("other4", other4) ]
    ~inputs:
      [
        in_string "equation" "byhwc,hkc->byhwk";
        in_tensors "tensors" [ "self5"; "other4" ];
      ]
    ~noutputs:1;
  [%expect
    {| error: einsum.default: unsupported equation "byhwc,hkc->byhwk" with operand ranks [5, 4] (only "byhwc,hkc->byhwk"/"byhwc,wkc->byhwk", each with a rank-5 self and rank-3 other, are recognized) |}]
