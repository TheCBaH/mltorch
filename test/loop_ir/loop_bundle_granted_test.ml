(* A node no Loop body realizes (an attention layer) is bundled only under a
   grant: refused with the typed row otherwise, and granted, one invocation
   whose program declares a buffer for every operand and the output, for the
   Machine IR site to replace. *)

module F = Native_test.Attention_layer_fixtures

let%expect_test "a bundle lowers the node only under a grant" =
  let g =
    F.node
      (F.with_norm F.L.Norm.Kind.Layer F.L.Placement.Pre
         {
           (F.all_flags F.small) with
           F.Config.rope = F.Rope_form.Shared;
           mask = true;
         })
  in
  let node = (List.hd g.Graph_ir.Graph.nodes).Graph_ir.Node.id in
  (match Loop_ir.Loop_bundle.build g with
  | Ok _ -> Fmt.pr "ungranted: built@."
  | Error e ->
      Fmt.pr "ungranted: %a@." Loop_ir.Loop_bundle.pp_error (Err.Error.kind e));
  match Loop_ir.Loop_bundle.build ~granted:(Graph_ir.Node_id.equal node) g with
  | Error e ->
      Fmt.pr "granted: %a@." Loop_ir.Loop_bundle.pp_error (Err.Error.kind e)
  | Ok b ->
      List.iter
        (fun (inv : Loop_ir.Loop_bundle.invocation) ->
          Fmt.pr "granted: %d buffers for %d operands and one output@."
            (List.length
               inv.Loop_ir.Loop_bundle.program.Loop_ir.Loop_program.buffers)
            (List.length
               (List.sort_uniq Tensor_id.compare
                  (Graph_ir.operands
                     (List.hd g.Graph_ir.Graph.nodes).Graph_ir.Node.op))))
        b.Loop_ir.Loop_bundle.invocations;
      [%expect
        {|
    ungranted: n0 is realized only under a grant, which this bundle was not given
    granted: 16 buffers for 15 operands and one output |}]
