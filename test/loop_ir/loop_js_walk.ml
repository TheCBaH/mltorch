(* The JavaScript one walked op lowers to, at its walk's initial config: every
   output of every node, through [Loop_node_program.lower] and [Loop_js.emit].
   Used by the per-op goldens and from the toplevel:

     ocaml
     # #use_output "dune top test/loop_ir";;
     # Loop_ir_test.Loop_js_walk.print "bmm";; *)

open Loop_ir

let print target =
  let m =
    match Native_op_walk.find target with
    | Some m -> m
    | None -> invalid_arg ("Loop_js_walk.print: no walk " ^ target)
  in
  let module M =
    (val m : Walk_core.Walk.Op with type subject = Native_op_walk.Subject.t)
  in
  let s, _ = M.build (Walk_core.Pcg.seed ~seed:0L ~seq:1L) M.initial in
  let g = s.Native_op_walk.Subject.graph in
  Fmt.pr "// %s %a@." target M.pp M.initial;
  List.iter
    (fun (node : Graph_ir.node) ->
      List.iter
        (fun (output, _) ->
          match Err.payload (Loop_node_program.lower g node ~output) with
          | Ok p -> print_string (Loop_js.emit p)
          | Error e -> Fmt.pr "// refused: %a@." Loop_node_program.pp_error e)
        (Output_ordinal.indexed node.Graph_ir.Node.outputs))
    g.Graph_ir.Graph.nodes
