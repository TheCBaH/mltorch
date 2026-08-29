(* Matmul-family (bmm/addmm/linear) op-dispatch arms for [Op_bridge], split from op_bridge.ml. See op_bridge.ml for the [dispatch] facade that
   folds every family module (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.addmm.default" ->
      Some
        ((* addmm(bias, mat1, mat2) = bias + mat1 @ mat2; alpha=beta=1 assumed. *)
         let* aten_bias = tensor_arg aten_env node "self" in
         let* aten_x = tensor_arg aten_env node "mat1" in
         let* aten_w = tensor_arg aten_env node "mat2" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 2 then
           fail (`Addmm_invalid_weight_rank w_shape)
         else
           let* bias = native_of_aten "self" aten_bias in
           let* x = native_of_aten "mat1" aten_x in
           let* w = native_of_aten "mat2" aten_w in
           try
             let params =
               { Linear.Linear.in_features = Dim.extent w_shape.(0) }
             in
             build_g ~name:"addmm_relayout" [ bias; x; w ] (function
               | [ bias_id; x_id; w_id ] ->
                   let open Graph_builder in
                   let* w' = permute perm_addmm_weight w_id in
                   let+ y = linear params ~x:x_id ~weight:w' ~bias:bias_id () in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.bmm.default" -> (
      match
        ( native_tensor_arg aten_env node "self",
          native_tensor_arg aten_env node "mat2" )
      with
      | Error e, _ | _, Error e -> Some (Error e)
      | Ok a, Ok b ->
          build_g ~name:"bmm" [ a; b ] (function
            | [ a_id; b_id ] ->
                let open Graph_builder in
                let+ y = bmm a_id b_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  | "torch.ops.aten.linear.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "input" in
         let* aten_w = tensor_arg aten_env node "weight" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 2 then
           fail (`Linear_invalid_weight_rank w_shape)
         else
           let* bias_opt =
             if optional_tensor_present node "bias" then
               let* bias = tensor_arg aten_env node "bias" in
               let* () = require_rank "bias" ~expected:1 bias in
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let params =
               { Linear.Linear.in_features = Dim.extent w_shape.(1) }
             in
             let tensors = [ x; w ] @ Option.to_list bias_opt in
             build_g ~name:"linear_relayout" tensors (function
               | [ x_id; w_id ] ->
                   let open Graph_builder in
                   let* w' = permute perm_linear_weight w_id in
                   let+ y = linear params ~x:x_id ~weight:w' () in
                   [ y ]
               | [ x_id; w_id; b_id ] ->
                   let open Graph_builder in
                   let* w' = permute perm_linear_weight w_id in
                   let+ y = linear params ~x:x_id ~weight:w' ~bias:b_id () in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | _ -> None
