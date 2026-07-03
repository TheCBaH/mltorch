(* Dual-dispatch wrapper for the graph interpreter.

   When [~verify:true], each node is executed on both the ATen path
   (Interp_dispatch) and the native path (Op_bridge), and outputs are compared
   element-wise by Verify.  Per-output errors are printed to [ppf] (default:
   stderr).  When [~verify:false] this is a thin pass-through with no overhead. *)

let dispatch ~verify ?(ppf = Format.err_formatter) (env : Interp_decode.env)
    (node : Pytorch_types.Node.t) : Interp_decode.env =
  let env' = Interp_dispatch.dispatch env node in
  (if verify then
     match Op_bridge.dispatch ~aten_env:env node with
     | None -> ()
     | Some (Error msg) ->
         Format.fprintf ppf "[verify] %s: bridge error: %s@." node.target msg
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
  env'
