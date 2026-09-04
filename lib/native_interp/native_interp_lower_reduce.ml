(* Reduction-family operator dispatch for [Native_interp_lower].  Each family
   returns [None] for targets it does not own; the lowerer owns the one
   unsupported-operator fallback. Split out of native_interp_lower_compute.ml
   (which crossed the tracked 1000-line ceiling, scripts/check-file-size.sh)
   rather than folded into it -- the [Op_bridge] side of this same family has
   its own file, op_bridge_reduce.ml, for the same reason. *)

open Pytorch_types
open Native_interp_decode

let targets = [ "torch.ops.aten.cumsum.default" ]

let dispatch ~ctx ~env (node : Node.t) =
  if not (List.mem node.target targets) then None
  else
    Some
      (let open Graph_builder in
       let esc = ctx.Native_interp_lower_context.esc in
       let graph = ctx.Native_interp_lower_context.graph in
       let get = Native_interp_lower_context.get ctx env node in
       match node.target with
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
