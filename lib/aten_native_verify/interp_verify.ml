(* Dual-dispatch wrapper for the graph interpreter.

   When [~verify:true], each node is executed on both the ATen path
   (Interp_dispatch) and the native path (Op_bridge), and outputs are compared
   element-wise by Verify.  Per-output errors are printed to [ppf] (default:
   stderr).  When [~verify:false] this is a thin pass-through with no overhead. *)

let pp_interp_error ppf = function
  | #Interp_decode.error as e -> Interp_decode.pp_error ppf e
  | `Aten_runtime_failure (op, st) ->
      Fmt.pf ppf "ATen op %s failed with status %d" op st
  | `Unhandled_op target -> Fmt.pf ppf "unhandled op %S" target

let dispatch ~verify ?(ppf = Format.err_formatter) (env : Interp_decode.env)
    (node : Pytorch_types.Node.t) =
  let open Err.Syntax in
  (* Native reads [env] BEFORE [Interp_dispatch.dispatch] runs, mirroring
     [Aten_spec_run.run]: a real in-place ATen op (e.g. [silu_]) mutates its
     `Tensor(a!)` argument through the very handle [env] holds, so capturing
     native's bridge dispatch + eval first is what keeps this comparison
     honest for an in-place target instead of comparing against a value ATen
     already overwrote (op5-impl). *)
  let native =
    if not verify then None else Some (Op_bridge.dispatch ~aten_env:env node)
  in
  let* env' = Interp_dispatch.dispatch env node in
  (match native with
  | None -> ()
  | Some None -> ()
  | Some (Some (Error e)) ->
      Fmt.pf ppf "[verify] %s: bridge error: %a@." node.target
        Op_bridge.pp_error (Err.Error.kind e)
  | Some (Some (Ok (graph, bindings))) -> (
      match Eval_direct.run graph ~inputs:bindings with
      | Error e ->
          Fmt.pf ppf "[verify] %s: eval error: %a@." node.target
            Eval_direct.pp_error (Err.Error.kind e)
      | Ok result_env ->
          let native_outputs =
            List.map
              (fun oid -> Graph_ir.Tensor_id.Map.find oid result_env)
              graph.Graph_ir.Graph.outputs
          in
          let atol = Verify.atol_for_target node.target in
          let errors =
            Verify.verify_node ~atol ~aten_env:env' node native_outputs
          in
          Verify.report ppf node.target errors));
  Err.return env'
