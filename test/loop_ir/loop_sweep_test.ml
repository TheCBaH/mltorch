open Loop_ir
open Loop_sweep_common

(* conv_add: the conv's buffer is eliminated. The exit criterion for the slice is
   that this agrees bitwise with the reference under BOTH placements while the
   program holds no buffer for the conv. *)
let%expect_test "conv_add is fused with the conv buffer eliminated" =
  let g = Native_test.Graph_fixtures.conv_add () in
  let kernel =
    Err.or_raise ~pp_error:Kernel_adapt.pp_error
      (Kernel_adapt.of_stage_program (Eval_symbolic.run g))
  in
  let bind id =
    if List.mem id g.Graph_ir.Graph.inputs then
      Some
        (Tensor.materialize
           (Tensor_id.Map.find id g.Graph_ir.Graph.tensors).Tensor_sig.shape
           (fun c -> float_of_int (Dim.to_int (Vec6.get c Axis.C) + 1)))
    else None
  in
  let fused, _ = Fusion_plan.plan kernel in
  List.iter
    (fun (name, plan) ->
      let buffers =
        match Loop_lower.lower plan with
        | Ok p ->
            List.map
              (fun (b : Loop_buffer.t) ->
                Fmt.str "%a" Tensor_id.pp b.Loop_buffer.id)
              p.Loop_program.buffers
        | Error _ -> []
      in
      Fmt.pr "%s: %a; buffers %s@." name Loop_check.pp_verdict
        (Loop_check.run plan ~bind)
        (String.concat "," buffers))
    [ ("default", Fusion_plan.default kernel); ("fused", fused) ];
  [%expect
    {|
    default: agree; buffers t0,t1,t2,t3,t4,t5
    fused: agree; buffers t0,t1,t2,t3,t5 |}]

(* ---- LSTM: one shared recurrence per canonical key -------------------------- *)

(* The reference's counters for every Region value of the kernel, summed: a group
   is counted under its first member and the others stay zero. *)
let reference_totals kernel ~bind =
  let per_value =
    List.map
      (fun (v : Kernel.Value.t) ->
        (v.Kernel.Value.id, Region_execution.counters ()))
      kernel.Kernel.values
  in
  ignore
    (Kernel_eval.run
       ~region_counters:
         (List.fold_left
            (fun m (id, c) -> Tensor_id.Map.add id c m)
            Tensor_id.Map.empty per_value)
       kernel ~bind);
  List.fold_left
    (fun (k, l, e, s, u) (_, (c : Region_execution.counters)) ->
      ( k + c.Region_execution.keys,
        l + c.Region_execution.locals,
        e + c.Region_execution.emitters,
        s + c.Region_execution.scans,
        u + c.Region_execution.scan_updates ))
    (0, 0, 0, 0, 0) per_value

let loop_totals kernel ~bind =
  let counters = Loop_interp.counters () in
  match Err.payload (Loop_lower.lower (Fusion_plan.default kernel)) with
  | Error _ -> None
  | Ok program ->
      ignore (Loop_interp.run ~counters program ~bind);
      Some
        ( counters.Loop_interp.keys,
          counters.Loop_interp.locals,
          counters.Loop_interp.emitters,
          counters.Loop_interp.scans,
          counters.Loop_interp.scan_updates )

let%expect_test
    "lstm agrees over many configurations and counts one recurrence per key" =
  let agree = ref 0 and counted = ref 0 and total = ref 0 and skipped = ref 0 in
  let verify _ppf (s : Native_op_walk.Subject.t) =
    (* Some configurations are beyond the expression budget and the symbolic
       builder raises: they have no kernel, so there is nothing to compare. *)
    (match
       Err.payload
         (Kernel_adapt.of_stage_program
            (Eval_symbolic.run s.Native_op_walk.Subject.graph))
     with
    | exception _ -> incr skipped
    | Error _ -> incr skipped
    | Ok kernel ->
        let bind id = List.assoc_opt id s.Native_op_walk.Subject.inputs in
        incr total;
        (match Loop_check.run (Fusion_plan.default kernel) ~bind with
        | Loop_check.Agree -> incr agree
        | verdict -> Fmt.pr "%a@." Loop_check.pp_verdict verdict);
        if loop_totals kernel ~bind = Some (reference_totals kernel ~bind) then
          incr counted
        else Fmt.pr "counters differ@.");
    true
  in
  let lstm = Option.get (Native_op_walk.find "lstm") in
  List.iter
    (fun seed ->
      ignore
        (Walk_core.Walk.run lstm ~verify ~ppf:silent
           ~pcg:(Walk_core.Pcg.seed ~seed ~seq:1L)
           ~steps:10))
    [ 0L; 1L; 2L; 3L ];
  Fmt.pr
    "configurations %d (%d too large for a kernel), agree %d, counters match \
     %d@."
    !total !skipped !agree !counted;
  [%expect
    {| configurations 44 (0 too large for a kernel), agree 44, counters match 44 |}]
