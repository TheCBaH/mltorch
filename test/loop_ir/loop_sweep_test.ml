open Loop_ir

(* The op sweep: every native op's own random walk, with a Loop verdict beside
   the walk's Direct-vs-Symbolic one. Each subject is adapted to a Kernel and run
   under BOTH placements -- everything stored, and the planner's fused plan -- so
   a fused chain and its unfused twin must each agree with [Kernel_eval].

   A refusal is recorded, not hidden: the table below is the inventory of what
   lowering does not yet cover. Nothing may DISAGREE. *)

type tally = {
  mutable agree : int;
  mutable agree_on_failure : int;
  mutable disagreements : string list;
  mutable refusals : string list;
      (** constructs, deduplicated, in first-seen order *)
  mutable not_a_kernel : int;
}

let tallies : (string, tally) Hashtbl.t = Hashtbl.create 64

let tally target =
  match Hashtbl.find_opt tallies target with
  | Some t -> t
  | None ->
      let t =
        {
          agree = 0;
          agree_on_failure = 0;
          disagreements = [];
          refusals = [];
          not_a_kernel = 0;
        }
      in
      Hashtbl.add tallies target t;
      t

let record t (verdict : Loop_check.verdict) =
  match verdict with
  | Loop_check.Agree -> t.agree <- t.agree + 1
  | Loop_check.Agree_on_failure _ ->
      t.agree_on_failure <- t.agree_on_failure + 1
  | Loop_check.Disagree d ->
      t.disagreements <-
        Fmt.str "%a" Loop_check.Disagreement.pp d :: t.disagreements
  | Loop_check.Refused u ->
      let name = Loop_unsupported.construct_name u.Loop_unsupported.construct in
      if not (List.mem name t.refusals) then t.refusals <- t.refusals @ [ name ]

let verify _ppf (s : Native_op_walk.Subject.t) =
  let t = tally s.Native_op_walk.Subject.target in
  let prog = Eval_symbolic.run s.Native_op_walk.Subject.graph in
  (match Kernel_adapt.of_stage_program prog with
  | Error _ -> t.not_a_kernel <- t.not_a_kernel + 1
  | Ok kernel ->
      let bind id = List.assoc_opt id s.Native_op_walk.Subject.inputs in
      List.iter
        (fun plan -> record t (Loop_check.run plan ~bind))
        [ Fusion_plan.default kernel; fst (Fusion_plan.plan kernel) ]);
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

let%expect_test "no walked op disagrees with the reference" =
  sweep ();
  let rows =
    Hashtbl.fold (fun target t acc -> (target, t) :: acc) tallies []
    |> List.sort (fun (a, _) (b, _) -> String.compare a b)
  in
  List.iter
    (fun (target, t) ->
      List.iter (fun d -> Fmt.pr "%s DISAGREES: %s@." target d) t.disagreements)
    rows;
  Fmt.pr "disagreements: %d@."
    (List.fold_left (fun n (_, t) -> n + List.length t.disagreements) 0 rows);
  [%expect {| disagreements: 0 |}]

let%expect_test "what lowers and what is refused, by op" =
  let rows =
    Hashtbl.fold (fun target t acc -> (target, t) :: acc) tallies []
    |> List.sort (fun (a, _) (b, _) -> String.compare a b)
  in
  List.iter
    (fun (target, t) ->
      Fmt.pr "%-28s agree=%d failed-alike=%d%s%s@." target t.agree
        t.agree_on_failure
        (if t.refusals = [] then ""
         else " refused: " ^ String.concat ", " t.refusals)
        (if t.not_a_kernel > 0 then Fmt.str " not-a-kernel=%d" t.not_a_kernel
         else ""))
    rows;
  [%expect
    {|
    adaptive_avg_pool2d          agree=12 failed-alike=0
    add                          agree=12 failed-alike=0
    add_scalar                   agree=12 failed-alike=0
    amax                         agree=12 failed-alike=0
    avg_pool2d                   agree=12 failed-alike=0
    batch_norm                   agree=12 failed-alike=0
    batch_norm_no_stats          agree=12 failed-alike=0
    batched_matmul               agree=12 failed-alike=0
    bmm                          agree=12 failed-alike=0
    clamp                        agree=12 failed-alike=0
    clone                        agree=12 failed-alike=0
    conv2d                       agree=12 failed-alike=0
    conv2d_padding               agree=12 failed-alike=0
    convolution                  agree=12 failed-alike=0
    cumsum                       agree=12 failed-alike=0
    div                          agree=12 failed-alike=0
    div_scalar                   agree=12 failed-alike=0
    expand                       agree=12 failed-alike=0
    gelu                         agree=12 failed-alike=0
    hardsigmoid                  agree=12 failed-alike=0
    hardswish                    agree=12 failed-alike=0
    hardtanh                     agree=12 failed-alike=0
    index_tensor                 agree=12 failed-alike=0
    layer_norm                   agree=12 failed-alike=0
    linear                       agree=12 failed-alike=0
    lstm                         agree=12 failed-alike=0
    max_dim                      agree=0 failed-alike=0 not-a-kernel=6
    max_pool2d                   agree=12 failed-alike=0
    max_pool2d_with_indices      agree=0 failed-alike=0 not-a-kernel=6
    mean                         agree=12 failed-alike=0
    mul                          agree=12 failed-alike=0
    mul_scalar                   agree=12 failed-alike=0
    pad                          agree=12 failed-alike=0
    permute                      agree=12 failed-alike=0
    pow                          agree=12 failed-alike=0
    relu                         agree=12 failed-alike=0
    reshape                      agree=12 failed-alike=0
    rms_norm                     agree=12 failed-alike=0
    sdpa                         agree=12 failed-alike=0
    sigmoid                      agree=12 failed-alike=0
    silu                         agree=12 failed-alike=0
    slice                        agree=12 failed-alike=0
    softmax                      agree=12 failed-alike=0
    sqrt                         agree=12 failed-alike=0
    sub                          agree=12 failed-alike=0
    sum                          agree=12 failed-alike=0
    unbind                       agree=12 failed-alike=0
    upsample_bicubic2d           agree=12 failed-alike=0
    upsample_bilinear2d          agree=12 failed-alike=0
    upsample_nearest2d           agree=12 failed-alike=0
    vector_norm                  agree=12 failed-alike=0 |}]

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
    {| configurations 41 (3 too large for a kernel), agree 41, counters match 41 |}]
