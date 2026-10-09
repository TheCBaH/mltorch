(* Embedding-lookup operator dispatch for [Native_interp_lower]:
   `aten.embedding.default`. Its own module keeps [Native_interp_lower_compute]
   under the file-size cap. *)

open Pytorch_types
open Native_interp_decode
open Native_interp_decode_shape

let targets = [ "torch.ops.aten.embedding.default" ]

let dispatch ~ctx ~env (node : Node.t) =
  if not (List.mem node.target targets) then None
  else
    Some
      (let open Graph_builder in
       let esc = ctx.Native_interp_lower_context.esc in
       let graph = ctx.Native_interp_lower_context.graph in
       let get = Native_interp_lower_context.get ctx env node in
       match node.target with
       (* `embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1,
         bool scale_grad_by_freq=False, bool sparse=False) -> Tensor`. The weight
         is the [V, D] table. [padding_idx] only steers ATen's backward pass; it
         is kept for provenance and changes no value (the stored padding row is
         returned as stored). [scale_grad_by_freq] and [sparse] are gradient
         options with no forward effect, so they are read and dropped, as
         [expand]'s [implicit] is. The indices' ATen rank is read here, before
         the six-axis frame erases it, because the lookup's output shape
         depends on it. *)
       | "torch.ops.aten.embedding.default" ->
           let w_name = tensor_name esc node "weight" in
           require_rank esc graph ~ssa:w_name ~role:`Embedding_weight
             ~expected:2;
           let i_name = tensor_name esc node "indices" in
           let indices_rank =
             meta_rank
               (tensor_meta esc graph ~ssa:i_name ~role:`Embedding_indices)
           in
           let padding_idx = int_arg esc ~default:(-1) node "padding_idx" in
           let (_ : bool) =
             bool_arg esc ~default:false node "scale_grad_by_freq"
           in
           let (_ : bool) = bool_arg esc ~default:false node "sparse" in
           let* y =
             embedding
               {
                 Embedding.Embedding.indices_rank;
                 padding_idx = Aten_int.Index.of_int padding_idx;
               }
               ~weight:(get "weight") ~indices:(get "indices")
           in
           return [ y ]
       | _ -> assert false)
