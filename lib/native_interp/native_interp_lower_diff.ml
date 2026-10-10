(* Preserve source rank before right-alignment hides leading singleton axes.
   All shared dtype, extent and order checks belong to Diff.output_shape. *)
open Pytorch_types
open Native_interp_decode
open Native_interp_decode_shape

let dispatch ~ctx ~env (node : Node.t) =
  if node.target <> "torch.ops.aten.diff.default" then None
  else
    Some
      (let open Graph_builder in
       let esc = ctx.Native_interp_lower_context.esc in
       let graph = ctx.Native_interp_lower_context.graph in
       let get = Native_interp_lower_context.get ctx env node in
       let name = tensor_name esc node "self" in
       let metadata name =
         let meta = tensor_meta esc graph ~ssa:name ~role:`Tensor in
         (match meta.TensorMeta.dtype with
         | ScalarType.BOOL | ScalarType.FLOAT | ScalarType.LONG -> ()
         | _ ->
             malformed esc
               (`Unsupported_option { op = node.target; option = `Dtype }));
         meta
       in
       let rank = meta_rank (metadata name) in
       let dim = dim_arg esc ~default:(-1) node "dim" in
       let axis =
         match axes_for_rank esc ~tensor:name rank [ dim ] with
         | [ axis ] -> axis
         | _ -> assert false
       in
       let optional arg =
         let ssa = optional_tensor_name ~absent_ok:true esc node arg in
         Option.iter
           (fun ssa ->
             let (_ : TensorMeta.t) = metadata ssa in
             require_rank esc graph ~ssa ~role:`Tensor ~expected:(rank :> int))
           ssa;
         Option.map (env_find esc env) ssa
       in
       let prepend = optional "prepend" in
       let append = optional "append" in
       let params =
         Diff.Diff.
           {
             axis;
             order = Aten_int.Size.of_int (int_arg esc ~default:1 node "n");
           }
       in
       let* y = diff params (get "self") ?prepend ?append () in
       return [ y ])
