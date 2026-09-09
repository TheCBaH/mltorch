(* `aten.einsum.default`'s [Shared_h] plan (`Graph_builder.einsum`,
   `.ai/einsum_design.md`), exercised as a real graph: two independent 2x2
   matrix multiplications, one per [h], each hand-computed with ordinary
   matrix arithmetic (not the digit-encoded self-describing values most
   other Direct fixtures use) so a wrong axis pairing produces a plainly
   wrong number rather than one that happens to still look plausible. *)

open Graph_ir
open Graph_direct_fixtures

let row c = Dim.to_int (Vec6.get c Axis.H)
let col c = Dim.to_int (Vec6.get c Axis.W)

let self_array =
  [| [| [| 1.; 2. |]; [| 3.; 4. |] |]; [| [| 9.; 10. |]; [| 11.; 12. |] |] |]

let other_array =
  [| [| [| 5.; 6. |]; [| 7.; 8. |] |]; [| [| 13.; 14. |]; [| 15.; 16. |] |] |]

(* h=0: [[1,2],[3,4]] @ [[5,6],[7,8]]^T = [[17,23],[39,53]]
   h=1: [[9,10],[11,12]] @ [[13,14],[15,16]]^T = [[257,295],[311,357]] *)
let self_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2
let other_shape = self_shape

let self_tensor =
  Tensor.materialize self_shape (fun c -> self_array.(row c).(col c).(chan c))

let other_tensor =
  Tensor.materialize other_shape (fun c -> other_array.(row c).(col c).(chan c))

let self_and_other_ids (g : graph) =
  match g.Graph.inputs with
  | [ self_id; other_id ] -> (self_id, other_id)
  | _ -> assert false

let build_graph () =
  Graph_builder.(
    build ~name:"einsum" ~outputs:(fun r -> [ r ])
    @@
    let* self = input ~shape:self_shape ~name:"self" () in
    let* other = input ~shape:other_shape ~name:"other" () in
    einsum ~name:"out" Aten_shape.Einsum.Shared_h self other)

let%expect_test "Direct graph: einsum Shared_h computes one matmul per shared H"
    =
  let result =
    let open Err.Syntax in
    let* g = lift_build (build_graph ()) in
    Format.printf "%a@." Graph_ir.pp g;
    let self_id, other_id = self_and_other_ids g in
    let* env =
      lift_eval
        (Eval_direct.run g
           ~inputs:[ (self_id, self_tensor); (other_id, other_tensor) ]
           ~constants:[])
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=2 C=2] ->[n1], t1 f32 [H=2 W=2 C=2] ->[n0]]
    nodes:
      n0: [t2 f32 [H=2 W=2 C=2] ->[n1]] = permute x=t1 perm=[W<-C, C<-W]
      n1: [t3 f32 [H=2 W=2 C=2]] = batched_matmul input=t0 mat2=t2 <-n0
    outputs: [t3 f32 [H=2 W=2 C=2] <-n1]
    out = tensor f32 [H=2 W=2 C=2] {17, 23, 39, 53, 257, 295, 311, 357}
    |}]

(* [Shared_w] reuses the SAME two tensors, just reinterpreting [other]'s own
   leading frame axis as the index shared with [self]'s [W] instead of its
   [H] -- so for fixed [w], [output[h,w,:] = self[h,w,:] @ other[w,:,:]^T]:
   w=0: self rows [1,2]/[9,10] against other[w=0]=[[5,6],[7,8]];
   w=1: self rows [3,4]/[11,12] against other[w=1]=[[13,14],[15,16]]. A
   DIFFERENT result from [Shared_h] on the identical inputs is itself part
   of the proof: it shows the plan choice, not the data, drives which axis
   is contracted against which. *)
let build_graph_shared_w () =
  Graph_builder.(
    build ~name:"einsum" ~outputs:(fun r -> [ r ])
    @@
    let* self = input ~shape:self_shape ~name:"self" () in
    let* other = input ~shape:other_shape ~name:"other" () in
    einsum ~name:"out" Aten_shape.Einsum.Shared_w self other)

let%expect_test "Direct graph: einsum Shared_w computes one matmul per shared W"
    =
  let result =
    let open Err.Syntax in
    let* g = lift_build (build_graph_shared_w ()) in
    Format.printf "%a@." Graph_ir.pp g;
    let self_id, other_id = self_and_other_ids g in
    let* env =
      lift_eval
        (Eval_direct.run g
           ~inputs:[ (self_id, self_tensor); (other_id, other_tensor) ]
           ~constants:[])
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=2 C=2] ->[n0], t1 f32 [H=2 W=2 C=2] ->[n1]]
    nodes:
      n0: [t2 f32 [H=2 W=2 C=2] ->[n2]] = permute x=t0 perm=[H<-W, W<-H]
      n1: [t3 f32 [H=2 W=2 C=2] ->[n2]] = permute x=t1 perm=[W<-C, C<-W]
      n2: [t4 f32 [H=2 W=2 C=2] ->[n3]] =
        batched_matmul input=t2 <-n0 mat2=t3 <-n1
      n3: [t5 f32 [H=2 W=2 C=2]] = permute x=t4 <-n2 perm=[H<-W, W<-H]
    outputs: [t5 f32 [H=2 W=2 C=2] <-n3]
    out = tensor f32 [H=2 W=2 C=2] {17, 23, 95, 109, 105, 143, 311, 357} |}]
