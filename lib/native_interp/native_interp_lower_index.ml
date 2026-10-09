(* `index.Tensor` with exactly two live, leading indices -- `self[i0, i1]`.
   The single-live-index forms stay in [Native_interp_lower_compute]; this
   family claims a node only when its `indices` list is two tensors (a list
   of two live entries, or an `as_tensors` pair) and defers otherwise, so a
   node of any other shape reaches the arm that already handles it. See
   [Index_tensor.Index_pair]. *)

open Pytorch_types
open Native_interp_decode
open Native_interp_decode_shape

let target = "torch.ops.aten.index.Tensor"

(* Whether [indices] is a pair of live tensors, without raising: any other
   spelling is not this family's. *)
let two_live (node : Node.t) =
  List.exists
    (fun (a : NamedArgument.t) ->
      a.name = "indices"
      &&
      match a.arg with
      | Argument.Tensors [ _; _ ] -> true
      | Argument.Optional_tensors
          [ OptionalTensorArgument.Tensor _; OptionalTensorArgument.Tensor _ ]
        ->
          true
      | _ -> false)
    node.Node.inputs

let dispatch ~ctx ~env (node : Node.t) =
  if not (String.equal node.target target && two_live node) then None
  else
    Some
      (let open Graph_builder in
       let esc = ctx.Native_interp_lower_context.esc in
       let graph = ctx.Native_interp_lower_context.graph in
       let self_name = tensor_name esc node "self" in
       let self_rank =
         meta_rank (tensor_meta esc graph ~ssa:self_name ~role:`Index_pair_self)
       in
       let names =
         match optional_tensor_names_arg esc node "indices" with
         | [ Some a; Some b ] -> (a, b)
         | _ -> assert false
       in
       let rank_of name =
         meta_rank (tensor_meta esc graph ~ssa:name ~role:`Index_pair_index)
       in
       let n0, n1 = names in
       let* y =
         index_pair
           {
             Index_tensor.Index_pair.self_rank;
             index0_rank = rank_of n0;
             index1_rank = rank_of n1;
           }
           ~self:(env_find esc env self_name)
           ~index0:(env_find esc env n0) ~index1:(env_find esc env n1)
       in
       return [ y ])
