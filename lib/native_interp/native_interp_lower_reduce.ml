(* Reduction-family operator dispatch for [Native_interp_lower].  Each family
   returns [None] for targets it does not own; the lowerer owns the one
   unsupported-operator fallback. Split out of native_interp_lower_compute.ml
   (which crossed the tracked 1000-line ceiling, scripts/check-file-size.sh)
   rather than folded into it -- the [Op_bridge] side of this same family has
   its own file, op_bridge_reduce.ml, for the same reason. *)

open Pytorch_types
open Schema_runtime
open Native_interp_decode
open Native_interp_decode_shape

let targets = [ "torch.ops.aten.cumsum.default"; "torch.ops.aten.max.dim" ]

let dispatch ~ctx ~env (node : Node.t) =
  if not (List.mem node.target targets) then None
  else
    Some
      (let open Graph_builder in
       let esc = ctx.Native_interp_lower_context.esc in
       let graph = ctx.Native_interp_lower_context.graph in
       let reads = ctx.Native_interp_lower_context.reads in
       let get = Native_interp_lower_context.get ctx env node in
       match node.target with
       (* The serialized output tuple is (values, indices), a genuine paired
          value/index reduction -- not [Amax] plus a dropped output, since
          [Amax] folds with [Float_max] while both of [Max_dim]'s outputs must
          fold with the same [Max_op.pool_better] predicate to stay in step
          (see [Semantics.SEMANTICS.max_dim]). Unlike
          [max_pool2d_with_indices.default]/[adaptive_max_pool2d.default],
          whose index is dead in every corpus occurrence and so is
          unconditionally routed to [Discard], `max.dim`'s index is a
          plausible live consumer (`values, indices = x.max(dim)` unpacks
          both in ordinary PyTorch code) -- so this importer checks liveness
          per output ([ctx.reads], the same mechanism [lstm.input]'s own
          three-output arm uses) rather than assuming deadness. *)
       | "torch.ops.aten.max.dim" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Amax_input)
           in
           let axis =
             match
               axes_for_rank esc ~tensor:x_name rank [ int_arg esc node "dim" ]
             with
             | [ axis ] -> axis
             | _ -> invalid_arg "Native_interp: max.dim lost its singleton axis"
           in
           let* values_id, indices_id =
             max_dim
               { Reduce.MaxDim.axis; keepdim = bool_arg esc node "keepdim" }
               (get "self")
           in
           let values_name, indices_name =
             match output_names esc node with
             | [ values_name; indices_name ] -> (values_name, indices_name)
             | names ->
                 malformed esc
                   (`Output_arity
                      {
                        op = node.target;
                        serialized = List.length names;
                        derived = 2;
                      })
           in
           let discard_if_dead name id =
             if String_map.mem name (Lazy.force reads) then return ()
             else discard id
           in
           let* () = discard_if_dead values_name values_id in
           let* () = discard_if_dead indices_name indices_id in
           return [ values_id; indices_id ]
       (* [dim] is required by the schema (no default), the same singleton-
          axis convention [softmax.int]'s arm (native_interp_lower_compute.ml)
          takes. Unlike every [reject_dtype] user in that file, [dtype] here
          is restricted to the schema default (None) or FLOAT rather than
          rejected outright: Native's compute domain is always float
          (semantics.ml), so a FLOAT target is the identity -- exactly
          [_to_copy.default]'s own [Float] case -- and every other target is
          rejected, since there is no corpus evidence and no Native value
          representation for it. *)
       | "torch.ops.aten.cumsum.default" ->
           let optional name =
             List.find_opt
               (fun (a : NamedArgument.t) -> a.name = name)
               node.Node.inputs
             |> Option.map (fun a -> a.NamedArgument.arg)
           in
           (match optional "dtype" with
           | None | Some (Argument.None _) -> ()
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) -> ()
           | Some _ ->
               malformed esc
                 (`Unsupported_option { op = node.target; option = `Dtype }));
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Cumsum_input)
           in
           let axis =
             match
               axes_for_rank esc ~tensor:x_name rank [ int_arg esc node "dim" ]
             with
             | [ a ] -> a
             | _ ->
                 invalid_arg "Native_interp: axes_for_rank lost its singleton"
           in
           let* y = cumsum { Reduce.Cumsum.axis } (get "self") in
           return [ y ]
       | _ -> assert false)
