(* T1.3: [Eval_symbolic.node_program] must build, for each of a node's own
   outputs, the identical stage [Eval_symbolic.run] builds for it in the
   whole (one-node) subject graph -- proven structurally, by rendering each
   side's computation and comparing the text -- and grounding the one-node
   program directly against the node's own operands must bitwise-agree with
   [Eval_direct.run]'s per-edge tensor for those same outputs. Both checks run
   over every walked subject (`Native_op_walk.all_walks`), the deepest
   fixture set already proven Direct==Symbolic bitwise
   (`Native_verify`/`test/loop_ir/loop_sweep_test.ml`), so a `node_program`
   defect surfaces on the same graphs, not a smaller hand-picked set. *)

(* [Native_verify.tensors_match]'s own check, inlined: that module isn't
   re-exported by [Native_op_walk]'s wrapping. *)
let tensors_match shape a b =
  let ok = ref true in
  Vec6.iter shape (fun c ->
      if not (Float.equal (Tensor.read a c) (Tensor.read b c)) then ok := false);
  !ok

let render_stage (st : Stage_program.Stage.t) =
  match
    Region_group.Ref.pixel_expression st.Stage_program.Stage.computation
  with
  | Some pixel -> Fmt.str "%a" Expr.Pp.value pixel
  | None -> Fmt.str "%a" Region_group.Ref.pp st.Stage_program.Stage.computation

let render_stage_i64 (st : Stage_program.Stage_i64.t) =
  Fmt.str "%a" Expr.Pp.value_i64 st.Stage_program.Stage_i64.pixel

let find_stage id stages =
  List.find_opt
    (fun (st : Stage_program.Stage.t) ->
      Tensor_id.equal st.Stage_program.Stage.id id)
    stages

let find_stage_i64 id stages =
  List.find_opt
    (fun (st : Stage_program.Stage_i64.t) ->
      Tensor_id.equal st.Stage_program.Stage_i64.id id)
    stages

(* [None, None] falls through to [stages_i64] rather than reporting a
   mismatch immediately: an output can legitimately live in either list
   depending on its declared format (e.g. an index output), and both sides
   agreeing on WHICH list it lives in is itself part of what this checks. *)
let structural_mismatches ~target (whole : Stage_program.t)
    (one : Stage_program.t) (node : Graph_ir.node) =
  List.filter_map
    (fun oid ->
      let id_str = Fmt.str "%a" Tensor_id.pp oid in
      match
        ( find_stage oid whole.Stage_program.stages,
          find_stage oid one.Stage_program.stages )
      with
      | Some sp, Some sq ->
          if String.equal (render_stage sp) (render_stage sq) then None
          else Some (Printf.sprintf "%s: stage mismatch at %s" target id_str)
      | None, None -> (
          match
            ( find_stage_i64 oid whole.Stage_program.stages_i64,
              find_stage_i64 oid one.Stage_program.stages_i64 )
          with
          | Some sp, Some sq ->
              if String.equal (render_stage_i64 sp) (render_stage_i64 sq) then
                None
              else
                Some
                  (Printf.sprintf "%s: stage_i64 mismatch at %s" target id_str)
          | _ -> Some (Printf.sprintf "%s: missing stage at %s" target id_str))
      | _ -> Some (Printf.sprintf "%s: stage kind mismatch at %s" target id_str))
    node.Graph_ir.Node.outputs

let ground_mismatches ~target (g : Graph_ir.graph) (node : Graph_ir.node)
    (one : Stage_program.t) (direct : Tensor.packed Tensor_id.Map.t) =
  let bind id = Tensor_id.Map.find id direct in
  match Stage_program.ground one ~bind with
  | Error e ->
      [
        Printf.sprintf "%s: node_program ground error: %s" target
          (Fmt.str "%a" Stage_program.pp_error (Err.Error.kind e));
      ]
  | Ok grounded ->
      List.filter_map
        (fun oid ->
          let d = Tensor_id.Map.find oid direct in
          let gd = Tensor_id.Map.find oid grounded in
          let shape =
            (Tensor_id.Map.find oid g.Graph_ir.Graph.tensors).Tensor_sig.shape
          in
          if tensors_match shape d gd then None
          else
            Some
              (Printf.sprintf "%s: grounded mismatch at %s" target
                 (Fmt.str "%a" Tensor_id.pp oid)))
        node.Graph_ir.Node.outputs

let verify _ppf (s : Native_op_walk.Subject.t) =
  let g = s.Native_op_walk.Subject.graph in
  let target = s.Native_op_walk.Subject.target in
  let whole = Eval_symbolic.run g in
  (match Eval_direct.run g ~inputs:s.Native_op_walk.Subject.inputs with
  | Error e ->
      Fmt.pr "%s: eval_direct error: %a@." target Eval_direct.pp_error
        (Err.Error.kind e)
  | Ok direct ->
      List.iter
        (fun (node : Graph_ir.node) ->
          let one = Eval_symbolic.node_program g node in
          List.iter (Fmt.pr "%s@.")
            (structural_mismatches ~target whole one node);
          List.iter (Fmt.pr "%s@.")
            (ground_mismatches ~target g node one direct))
        g.Graph_ir.Graph.nodes);
  true

let silent = Format.make_formatter (fun _ _ _ -> ()) (fun () -> ())

let sweep () =
  List.iteri
    (fun index (m : Native_op_walk.op) ->
      ignore
        (Walk_core.Walk.run m ~verify ~ppf:silent
           ~pcg:(Walk_core.Pcg.seed ~seed:(Int64.of_int index) ~seq:1L)
           ~steps:5))
    Native_op_walk.all_walks

let%expect_test "node_program agrees with run and Eval_direct, every walked op"
    =
  sweep ();
  [%expect {||}]
