(* Dual-dispatch wrapper for the graph interpreter.

   When [~verify:true], each node is executed on both the ATen path
   (Interp_dispatch) and the native path (Op_bridge), and outputs are compared
   element-wise by Verify.  Per-output errors are printed to [ppf] (default:
   stderr).  When [~verify:false] this is a thin pass-through with no overhead. *)

let pp_interp_error ppf = function
  | #Interp_decode.error as e -> Interp_decode.pp_error ppf e
  | `Aten_runtime_failure (op, st) ->
      Format.fprintf ppf "ATen op %s failed with status %d" op st
  | `Unhandled_op target -> Format.fprintf ppf "unhandled op %S" target

let dispatch ~verify ?(ppf = Format.err_formatter) (env : Interp_decode.env)
    (node : Pytorch_types.Node.t) =
  let open Core.Syntax in
  let* env' = Interp_dispatch.dispatch env node in
  (if verify then
     match Op_bridge.dispatch ~aten_env:env node with
     | None -> ()
     | Some (Error e) ->
         Format.fprintf ppf "[verify] %s: bridge error: %a@." node.target
           Op_bridge.pp_error e.Core.Error.kind
     | Some (Ok (graph, bindings)) -> (
         match Eval_direct.run graph ~inputs:bindings with
         | Error e ->
             Format.fprintf ppf "[verify] %s: eval error: %a@." node.target
               Eval_direct.pp_error e.Core.Error.kind
         | Ok result_env ->
             let native_outputs =
               List.map
                 (fun oid -> Graph_ir.Tensor_id.Map.find oid result_env)
                 graph.Graph_ir.Graph.outputs
             in
             let errors =
               Verify.verify_node ~atol:1e-5 ~aten_env:env' node native_outputs
             in
             Verify.report ppf node.target errors));
  Core.return env'
