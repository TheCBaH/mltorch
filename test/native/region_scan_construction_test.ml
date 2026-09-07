(* Region-level scan construction, checking, rendering and (Stage 1's final
   acceptance step, see .ai/native_scan_design.md) execution through the
   whole-graph Stage and Kernel paths -- not just Region's own materializer,
   which test/native/region_scan_execution_test.ml already covers. *)

open Expr

let limits =
  Err.or_raise ~pp_error:Scan_limits.pp_error
    (Scan_limits.create ~max_state:100 ~max_updates:1000L)

let pos n = Index.clamp_low (Index.const n)
let partition = Region_partition.singleton

let pp =
  Core.Pretty.err_result ~ok:Region_program.pp ~error:Region_program.pp_error

(* trace.(0,l) = 0; trace.(s+1,l) = trace.(s,l) + 1 -- the same counter
   [test/expr/scan_test.ml] uses, reused here through
   [Region_program.Builder.scan]. *)
let counter_scan ~width ~steps continue =
  Region_program.Builder.scan ~limits ~width ~steps
    ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
    ~update:(fun ~step:_ ~lane ~previous_at ->
      Builder.return (Value.add (previous_at lane) (Value.const 1.)))
    continue

let build_counter ~width ~steps ~row ~lane =
  Region_program.Builder.run
    (counter_scan ~width ~steps (fun scan_read ->
         Region_program.Builder.finish ~max_size:64 ~max_depth:16 ~partition
           ~output:(scan_read ~row:(pos row) ~lane:(pos lane))))

let scan_local ~width ~steps =
  match
    Region_program.locals
      (Err.or_raise ~pp_error:Region_program.pp_error
         (build_counter ~width ~steps ~row:steps ~lane:0))
  with
  | [ local ] -> local
  | _ -> assert false

let%expect_test "builds, checks and renders a scan-backed Region program" =
  Fmt.pr "%a@." pp (build_counter ~width:2 ~steps:3 ~row:3 ~lane:0);
  [%expect
    {|
    region [N=singleton T=singleton D=singleton H=singleton W=singleton C=singleton]
      let l0 : scan[width=2,steps=3] = scan[w=2,s=3](init[r1]=0; update[r2,r3,p3]=(p3[r2] + 1))
      emit l0@[max(0,3),max(0,0)] |}]

let%expect_test "Builder.scan short-circuits an invalid descriptor" =
  let invoked = ref false in
  let result =
    Region_program.Builder.run
      (Region_program.Builder.scan ~limits ~width:2 ~steps:(-1)
         ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
         ~update:(fun ~step:_ ~lane ~previous_at ->
           Builder.return (Value.add (previous_at lane) (Value.const 1.)))
         (fun _scan_read ->
           invoked := true;
           Region_program.Builder.finish ~max_size:64 ~max_depth:16 ~partition
             ~output:(Value.const 0.)))
  in
  Fmt.pr "invoked=%b result=%a@." !invoked pp result;
  [%expect {| invoked=false result=invalid scan step count -1 |}]

let%expect_test "a scalar read of a declared trace local is rejected" =
  let local = scan_local ~width:2 ~steps:3 in
  Fmt.pr "%a@." pp
    (Region_program.create ~max_size:64 ~max_depth:16 ~partition
       ~locals:[ local ]
       ~output:(Value.local local.Region_local.id));
  [%expect
    {| local #3 is read as a scalar but declared scan[width=2,steps=3] |}]

let%expect_test "a vector read of a declared trace local is rejected" =
  let local = scan_local ~width:2 ~steps:3 in
  Fmt.pr "%a@." pp
    (Region_program.create ~max_size:64 ~max_depth:16 ~partition
       ~locals:[ local ]
       ~output:(Value.local_at local.Region_local.id Index.zero));
  [%expect
    {| local #3 is read as a vector but declared scan[width=2,steps=3] |}]

let%expect_test "a trace read of a declared scalar local is rejected" =
  let id = Builder.run Builder.fresh_local in
  let local = Region_local.scalar ~id ~value:(Value.const 1.) in
  Fmt.pr "%a@." pp
    (Region_program.create ~max_size:64 ~max_depth:16 ~partition
       ~locals:[ local ]
       ~output:(Value.local_scan_at id ~row:Index.zero ~lane:Index.zero));
  [%expect {| local #0 is read as a trace but declared scalar |}]

let%expect_test "a scan's update cannot forward-reference a later local" =
  let (scan_id, later_id), _state =
    Builder.run_from Builder.initial
      (let open Builder.Syntax in
       let* scan_id = Builder.fresh_local in
       let* later_id = Builder.fresh_local in
       Builder.return (scan_id, later_id))
  in
  let forward_scan =
    Err.or_raise ~pp_error:Scan.pp_error
      (Builder.run
         (Builder.scan ~limits ~width:2 ~steps:1
            ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
            ~update:(fun ~step:_ ~lane:_ ~previous_at:_ ->
              Builder.return (Value.local later_id))))
  in
  Fmt.pr "%a@." pp
    (Region_program.create ~max_size:64 ~max_depth:16 ~partition
       ~locals:
         [
           Region_local.scan ~id:scan_id ~scan:forward_scan;
           Region_local.scalar ~id:later_id ~value:(Value.const 1.);
         ]
       ~output:(Value.local_scan_at scan_id ~row:Index.zero ~lane:Index.zero));
  [%expect {| local #0 refers forward to #1 |}]

(* ---- Stage 1 acceptance: the same construction, executed end to end ------ *)

let s1 = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1

let out_sig =
  Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"" ~shape:s1
    ~fmt:(Payload.Fmt Payload.F32) ()

let read_out (m : Tensor.packed Tensor_id.Map.t) =
  match Tensor_id.Map.find_opt (Tensor_id.of_int 0) m with
  | Some t -> Tensor.read t Vec6.origin
  | None -> assert false

(* The scan reads row [steps] -- trace.(steps,_) = steps, so the single
   output cell (a constant expression, independent of the output
   coordinate: [build_counter] does not thread it through) is exactly
   [float_of_int steps]. *)
let%expect_test
    "a scan-backed Region program executes through Stage_program.ground and \
     Kernel_eval.run, and both agree" =
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (build_counter ~width:2 ~steps:5 ~row:5 ~lane:0)
  in
  let stage_program : Stage_program.t =
    {
      Stage_program.inputs = [];
      input_kinds = Tensor_id.Map.empty;
      consts = [];
      stages =
        [
          {
            Stage_program.Stage.id = Tensor_id.of_int 0;
            sg = out_sig;
            computation = program;
          };
        ];
      outputs = [ Tensor_id.of_int 0 ];
    }
  in
  let via_stage =
    Err.or_raise ~pp_error:Stage_program.pp_error
      (Stage_program.ground stage_program ~bind:(fun _ -> assert false))
  in
  let kernel =
    Err.or_raise ~pp_error:Kernel.pp_error
      (Kernel.create ~inputs:[]
         ~values:
           [
             {
               Kernel.Value.id = Tensor_id.of_int 0;
               sg = out_sig;
               computation = program;
               result = Kernel.Result_conversion.Round_f32;
             };
           ]
         ~outputs:[ Tensor_id.of_int 0 ]
         ())
  in
  let via_kernel =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run kernel ~bind:(fun _ -> None))
  in
  Fmt.pr "stage=%g kernel=%g@." (read_out via_stage) (read_out via_kernel);
  [%expect {| stage=5 kernel=5 |}]

(* Closes the remaining leg of the Stage 1 exit conditions' "agrees across
   production/reference Region execution, specialized Expr evaluation":
   [specialize_pixel] fully inlines every local, so the result has no [Local]/
   [Local_at]/[Local_scan_at] left -- evaluating it needs no [env] and no
   [reducer] seed, only a fresh [Scan_meter.t] for its own inline [Scan_at]. *)
let%expect_test
    "specializing and directly evaluating a scan-backed program agrees with \
     materializing it through Region" =
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (build_counter ~width:2 ~steps:5 ~row:5 ~lane:0)
  in
  let specialized =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.specialize_pixel ~max_size:64 ~max_depth:16
         ~scan_limits:limits program)
  in
  let env =
    {
      Eval.Env.load = (fun _ _ -> assert false);
      load_index = (fun _ _ -> assert false);
    }
  in
  let evaluated =
    Err.or_raise ~pp_error:Eval.pp_error
      (Eval.value env
         ~scan_meter:(Scan_meter.create ~limits)
         ~output:(Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int Vec6.origin))
         specialized)
  in
  let materialized =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_eval.materialize program ~output_shape:s1 ~env)
  in
  Fmt.pr "evaluated=%g materialized=%g@." evaluated
    (Tensor.read materialized Vec6.origin);
  [%expect {| evaluated=5 materialized=5 |}]

(* The Stage 1 exit conditions also claim the specialized AST's size is
   INDEPENDENT of [width]/[steps] -- true by construction, since [Scan.t]
   stores both as plain integer fields rather than unrolling, but never
   directly measured until now: two programs differing only in [steps] (5 vs.
   500, a 100x difference) specialize to exactly the same size and depth. *)
let%expect_test "a specialized scan's size and depth do not grow with steps" =
  let specialize ~steps =
    Err.or_raise ~pp_error:Region_program.pp_error
      (build_counter ~width:2 ~steps ~row:steps ~lane:0)
    |> Region_program.specialize_pixel ~max_size:64 ~max_depth:16
         ~scan_limits:limits
    |> Err.or_raise ~pp_error:Region_program.pp_error
  in
  let small = specialize ~steps:5 and large = specialize ~steps:500 in
  Fmt.pr "small=(%d,%d) large=(%d,%d) equal=%b@." (Expr.Fold.size small)
    (Expr.Fold.depth small) (Expr.Fold.size large) (Expr.Fold.depth large)
    (Expr.Fold.size small = Expr.Fold.size large
    && Expr.Fold.depth small = Expr.Fold.depth large);
  [%expect {| small=(10,4) large=(10,4) equal=true |}]

(* Superseded: grounding now executes a scan-backed Region program directly,
   instead of rejecting it with `Scan_at_unsupported. The counter trace's
   row-5 value is 0+1+1+1+1+1 = 5,
   wrapped in the stage's own materialization [Round] -- the same value
   [test/native/region_scan_execution_test.ml]'s production/reference
   execution already established, now also reachable through [Ground_eval]. *)
let%expect_test "grounding executes a scan-backed stage" =
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (build_counter ~width:2 ~steps:5 ~row:5 ~lane:0)
  in
  let stage_program : Stage_program.t =
    {
      Stage_program.inputs = [];
      input_kinds = Tensor_id.Map.empty;
      consts = [];
      stages =
        [
          {
            Stage_program.Stage.id = Tensor_id.of_int 0;
            sg = out_sig;
            computation = program;
          };
        ];
      outputs = [ Tensor_id.of_int 0 ];
    }
  in
  let ground_env = Ground_eval.Env.of_program stage_program ~side:`Src in
  let meter = Ground_eval.Meter.create Ground_eval.default_budget in
  Fmt.pr "%a@."
    (Core.Pretty.err_result ~ok:Ground_expr.pp ~error:Ground_eval.pp_error)
    (Result.map Ground_eval.Term.expression
       (Ground_eval.at ~meter ground_env (Tensor_id.of_int 0) Vec6.origin));
  [%expect
    {| f32((((((0x0p+0 + 0x1p+0) + 0x1p+0) + 0x1p+0) + 0x1p+0) + 0x1p+0)) |}]

let stage_of program =
  {
    Stage_program.inputs = [];
    input_kinds = Tensor_id.Map.empty;
    consts = [];
    stages =
      [
        {
          Stage_program.Stage.id = Tensor_id.of_int 0;
          sg = out_sig;
          computation = program;
        };
      ];
    outputs = [ Tensor_id.of_int 0 ];
  }

let ground_row stage_program =
  let ground_env = Ground_eval.Env.of_program stage_program ~side:`Src in
  let meter = Ground_eval.Meter.create Ground_eval.default_budget in
  let term =
    Err.or_raise ~pp_error:Ground_eval.pp_error
      (Ground_eval.at ~meter ground_env (Tensor_id.of_int 0) Vec6.origin)
  in
  Ground_expr.eval
    (Ground_eval.Term.expression term)
    Ground_expr.Valuation.empty

let region_row stage_program =
  read_out
    (Err.or_raise ~pp_error:Stage_program.pp_error
       (Stage_program.ground stage_program ~bind:(fun _ -> assert false)))

(* Every row of the same counter trace, compared bitwise against the
   production evaluator ([Stage_program.ground], which runs
   [Region_execution.materialize]) -- not just the one row the earlier test
   pins. *)
let%expect_test
    "grounding agrees with Region_execution across every row, including row 0" =
  List.iter
    (fun row ->
      let stage_program =
        stage_of
          (Err.or_raise ~pp_error:Region_program.pp_error
             (build_counter ~width:2 ~steps:5 ~row ~lane:0))
      in
      let g = ground_row stage_program and r = region_row stage_program in
      Fmt.pr "row=%d grounded=%g region=%g agree=%b@." row g r (Float.equal g r))
    [ 0; 1; 2; 3; 4; 5 ];
  [%expect
    {|
    row=0 grounded=0 region=0 agree=true
    row=1 grounded=1 region=1 agree=true
    row=2 grounded=2 region=2 agree=true
    row=3 grounded=3 region=3 agree=true
    row=4 grounded=4 region=4 agree=true
    row=5 grounded=5 region=5 agree=true |}]

(* A two-lane COUPLED recurrence: each lane's update reads the OTHER lane's
   previous row ([other_lane = 1 - lane], via plain [Index] arithmetic on the
   symbolic per-position [lane]), which an accidental in-place row update (row
   r's lane 1 read after lane 0 has already overwritten row r in place) would
   corrupt starting at row 1. Hand-computed: a0=1,b0=2;
   a1=a0+3*b0=7, b1=b0+3*a0=5; a2=a1+3*b1=22, b2=b1+3*a1=26. *)
let coupled_scan ~steps continue =
  let other lane =
    Index.clamp_low
      (Index.add (Index.const 1) (Index.scale (-1) (Index.of_position lane)))
  in
  Region_program.Builder.scan ~limits ~width:2 ~steps
    ~init:(fun ~lane ->
      Builder.return
        (Value.value_of_index
           (Index.add (Index.const 1) (Index.of_position lane))))
    ~update:(fun ~step:_ ~lane ~previous_at ->
      Builder.return
        (Value.add (previous_at lane)
           (Value.mul (previous_at (other lane)) (Value.const 3.))))
    continue

let build_coupled ~steps ~row ~lane =
  Region_program.Builder.run
    (coupled_scan ~steps (fun scan_read ->
         Region_program.Builder.finish ~max_size:64 ~max_depth:16 ~partition
           ~output:(scan_read ~row:(pos row) ~lane:(pos lane))))

let%expect_test
    "grounding a coupled two-lane scan agrees with Region_execution, per lane \
     and per row" =
  List.iter
    (fun (row, lane) ->
      let stage_program =
        stage_of
          (Err.or_raise ~pp_error:Region_program.pp_error
             (build_coupled ~steps:3 ~row ~lane))
      in
      let g = ground_row stage_program and r = region_row stage_program in
      Fmt.pr "row=%d lane=%d grounded=%g region=%g agree=%b@." row lane g r
        (Float.equal g r))
    [ (0, 0); (0, 1); (1, 0); (1, 1); (2, 0); (2, 1) ];
  [%expect
    {|
    row=0 lane=0 grounded=1 region=1 agree=true
    row=0 lane=1 grounded=2 region=2 agree=true
    row=1 lane=0 grounded=7 region=7 agree=true
    row=1 lane=1 grounded=5 region=5 agree=true
    row=2 lane=0 grounded=22 region=22 agree=true
    row=2 lane=1 grounded=26 region=26 agree=true |}]

(* CHAINED scans: local [b]'s own [update] reads an EARLIER local, [a]'s,
   completed trace via [Local_scan_at] ([scan_read_a] below, captured from the
   outer [Builder.scan]'s [continue]) -- not [b]'s own [previous_at], and not
   an inline [Scan_at]. Structurally supported since [Ground_eval.Frame] is
   built in declaration order (every earlier local is already in scope by the
   time a later one is grounded, ground_eval.ml's [body_at]), but until now
   untested. a(s) = s (the same counter); b(0) = 0, b(s+1) = b(s) + a(s), so
   b is the triangular numbers: b(1)=0, b(2)=1, b(3)=3, b(4)=6, b(5)=10. *)
let chained_scan ~steps continue =
  Region_program.Builder.scan ~limits ~width:1 ~steps
    ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
    ~update:(fun ~step:_ ~lane ~previous_at ->
      Builder.return (Value.add (previous_at lane) (Value.const 1.)))
    (fun scan_read_a ->
      Region_program.Builder.scan ~limits ~width:1 ~steps
        ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
        ~update:(fun ~step ~lane ~previous_at ->
          Builder.return
            (Value.add (previous_at lane) (scan_read_a ~row:step ~lane)))
        continue)

let build_chained ~steps ~row =
  Region_program.Builder.run
    (chained_scan ~steps (fun scan_read_b ->
         Region_program.Builder.finish ~max_size:64 ~max_depth:16 ~partition
           ~output:(scan_read_b ~row:(pos row) ~lane:Index.zero)))

let%expect_test
    "grounding a chained scan (one scan's update reading an earlier scan's \
     completed trace) agrees with Region_execution, at every row" =
  List.iter
    (fun row ->
      let stage_program =
        stage_of
          (Err.or_raise ~pp_error:Region_program.pp_error
             (build_chained ~steps:5 ~row))
      in
      let g = ground_row stage_program and r = region_row stage_program in
      Fmt.pr "row=%d grounded=%g region=%g agree=%b@." row g r (Float.equal g r))
    [ 0; 1; 2; 3; 4; 5 ];
  [%expect
    {|
    row=0 grounded=0 region=0 agree=true
    row=1 grounded=0 region=0 agree=true
    row=2 grounded=1 region=1 agree=true
    row=3 grounded=3 region=3 agree=true
    row=4 grounded=6 region=6 agree=true
    row=5 grounded=10 region=10 agree=true |}]
