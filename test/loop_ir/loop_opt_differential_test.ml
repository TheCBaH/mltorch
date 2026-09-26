open Loop_ir

(* Stage 0 of the Loop IR optimization plan (.ai/loop_ir_optimization_design.md):
   the optimized-vs-raw differential. [Loop_lower.lower] applies [Loop_opt.run]
   and [Loop_lower.lower_unoptimized] does not, so this compares the two through
   [Loop_interp] for every walked op, bitwise on values and by kind and payload
   on failures ([Loop_check.compare]'s own semantics, reused rather than
   reimplemented). [Loop_opt.passes] starts empty, so today this only proves the
   harness is sound; the mutation test below proves it is not vacuous, and every
   later stage adds a pass this same sweep must keep agreeing with. *)

type tally = {
  mutable agree : int;
  mutable refused : int;
  mutable disagreements : string list;
}

let tallies : (string, tally) Hashtbl.t = Hashtbl.create 64

let tally target =
  match Hashtbl.find_opt tallies target with
  | Some t -> t
  | None ->
      let t = { agree = 0; refused = 0; disagreements = [] } in
      Hashtbl.add tallies target t;
      t

(* [Loop_interp.error]'s cases are a subset of [Kernel_eval.error]'s with
   identical payload types, so this widening is a plain upcast -- the same one
   [Loop_check.compare_results] applies internally to an executor's error. *)
let widen (e : Loop_interp.error) = (e :> Kernel_eval.error)

let check t (plan : Fusion_plan.t) ~bind =
  match
    ( Err.payload (Loop_lower.lower_unoptimized plan),
      Err.payload (Loop_lower.lower plan) )
  with
  | Error _, Error _ -> t.refused <- t.refused + 1
  | Ok raw, Ok opt -> (
      let raw_result = Loop_interp.run raw ~bind in
      let opt_result = Loop_interp.run opt ~bind in
      let reference = Err.map_error widen raw_result in
      match Loop_check.compare ~reference ~loop:opt_result with
      | Loop_check.Agree | Loop_check.Agree_on_failure _ ->
          t.agree <- t.agree + 1
      | Loop_check.Refused u ->
          t.disagreements <-
            Fmt.str "unexpected refusal: %a" Loop_unsupported.pp u
            :: t.disagreements
      | Loop_check.Disagree d ->
          t.disagreements <-
            Fmt.str "%a" Loop_check.Disagreement.pp d :: t.disagreements)
  | Error _, Ok _ | Ok _, Error _ ->
      t.disagreements <-
        "raw and optimized lowering disagreed on acceptance" :: t.disagreements

let verify _ppf (s : Native_op_walk.Subject.t) =
  let t = tally s.Native_op_walk.Subject.target in
  let prog = Eval_symbolic.run s.Native_op_walk.Subject.graph in
  (match Kernel_adapt.of_stage_program prog with
  | Error _ -> ()
  | Ok kernel ->
      let bind id = List.assoc_opt id s.Native_op_walk.Subject.inputs in
      List.iter
        (fun plan -> check t plan ~bind)
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

let%expect_test
    "the optimized program never disagrees with its own raw lowering" =
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
  [%expect {|
    disagreements: 0 |}]

(* ---- mutation proof: the differential is not vacuous ---------------------- *)

(* A hand-broken "identity" pass: cross two Store statements' destination
   buffers, leaving their coordinates and computed values untouched. Crossing
   destinations (rather than values) keeps every expression evaluated in its
   own original loop scope, so no free variable goes out of scope; requiring
   equal shapes keeps the write in bounds on both sides. [unbind] always offers
   such a pair when its walk draws more than one slice: every output shares the
   input's shape with the split dimension removed, so any two of them match. *)

let rec collect_store_buffers (stmts : Loop_stmt.t list) acc =
  List.fold_left
    (fun acc (stmt : Loop_stmt.t) ->
      match stmt with
      | Loop_stmt.Store { buffer; _ } | Loop_stmt.Store_flat { buffer; _ } ->
          buffer :: acc
      | Loop_stmt.For { body; _ } -> collect_store_buffers body acc
      | Loop_stmt.If (_, a, b) ->
          collect_store_buffers b (collect_store_buffers a acc)
      | Loop_stmt.Alloc _ | Loop_stmt.Array_set _ | Loop_stmt.Assign _
      | Loop_stmt.Assign_index _ | Loop_stmt.Assign_index_of_i64 _
      | Loop_stmt.Charge_scan_update | Loop_stmt.Fail_if _ | Loop_stmt.Mark _
      | Loop_stmt.Release_scan_state _ | Loop_stmt.Reserve_scan_state _
      | Loop_stmt.Reset_meter ->
          acc)
    acc stmts

let find_pair (buffers : Loop_buffer.t list) =
  List.concat_map
    (fun (a : Loop_buffer.t) ->
      List.filter_map
        (fun (b : Loop_buffer.t) ->
          if
            (not (Tensor_id.equal a.Loop_buffer.id b.Loop_buffer.id))
            && Stdlib.( = ) a.Loop_buffer.sg.Tensor_sig.shape
                 b.Loop_buffer.sg.Tensor_sig.shape
          then Some (a, b)
          else None)
        buffers)
    buffers
  |> function
  | pair :: _ -> Some pair
  | [] -> None

let rec retarget_stores (stmts : Loop_stmt.t list) ~(a : Loop_buffer.t)
    ~(b : Loop_buffer.t) =
  List.map
    (fun (stmt : Loop_stmt.t) ->
      match stmt with
      | Loop_stmt.Store { buffer; coord; value }
        when Tensor_id.equal buffer.Loop_buffer.id a.Loop_buffer.id ->
          Loop_stmt.Store { buffer = b; coord; value }
      | Loop_stmt.Store { buffer; coord; value }
        when Tensor_id.equal buffer.Loop_buffer.id b.Loop_buffer.id ->
          Loop_stmt.Store { buffer = a; coord; value }
      | Loop_stmt.Store _ as s -> s
      | Loop_stmt.For f ->
          Loop_stmt.For { f with body = retarget_stores f.body ~a ~b }
      | Loop_stmt.If (p, x, y) ->
          Loop_stmt.If (p, retarget_stores x ~a ~b, retarget_stores y ~a ~b)
      | s -> s)
    stmts

let swap_two_stores : Loop_opt.pass =
 fun program ->
  match find_pair (collect_store_buffers program.Loop_program.body []) with
  | None -> program
  | Some (a, b) ->
      {
        program with
        Loop_program.body = retarget_stores program.Loop_program.body ~a ~b;
      }

let%expect_test
    "the differential is not vacuous: crossing two stores' destinations turns \
     it red" =
  let found = ref None in
  let verify _ppf (s : Native_op_walk.Subject.t) =
    (if !found = None then
       let prog = Eval_symbolic.run s.Native_op_walk.Subject.graph in
       match Kernel_adapt.of_stage_program prog with
       | Error _ -> ()
       | Ok kernel -> (
           let bind id = List.assoc_opt id s.Native_op_walk.Subject.inputs in
           let plan = Fusion_plan.default kernel in
           match Err.payload (Loop_lower.lower_unoptimized plan) with
           | Error _ -> ()
           | Ok raw ->
               if
                 find_pair (collect_store_buffers raw.Loop_program.body [])
                 <> None
               then found := Some (raw, bind)));
    true
  in
  (match Native_op_walk.find "unbind" with
  | None -> ()
  | Some m ->
      List.iter
        (fun seed ->
          if !found = None then
            ignore
              (Walk_core.Walk.run m ~verify ~ppf:silent
                 ~pcg:(Walk_core.Pcg.seed ~seed ~seq:1L)
                 ~steps:5))
        [ 0L; 1L; 2L; 3L ]);
  (match !found with
  | None -> Fmt.pr "no swappable pair found in the unbind walk@."
  | Some (raw, bind) ->
      let raw_result = Loop_interp.run raw ~bind in
      let broken = Loop_opt.run ~passes:[ swap_two_stores ] raw in
      let broken_result = Loop_interp.run broken ~bind in
      let reference = Err.map_error widen raw_result in
      Fmt.pr "%a@." Loop_check.pp_verdict
        (Loop_check.compare ~reference ~loop:broken_result));
  [%expect {| DISAGREE: t3 differs bitwise |}]

(* ---- Stage 1 mutation proof: [lo + 1] instead of [lo] must turn it red --- *)

let rec has_unit_loop (stmts : Loop_stmt.t list) =
  List.exists
    (fun (stmt : Loop_stmt.t) ->
      match stmt with
      | Loop_stmt.For
          { lo = Loop_index.Const l; hi = Loop_index.Const h; body; _ } ->
          h - l = 1 || has_unit_loop body
      | Loop_stmt.For { body; _ } -> has_unit_loop body
      | Loop_stmt.If (_, a, b) -> has_unit_loop a || has_unit_loop b
      | _ -> false)
    stmts

let%expect_test
    "stage 1 mutation proof: substituting lo + 1 instead of lo turns it red" =
  let found = ref None in
  let verify _ppf (s : Native_op_walk.Subject.t) =
    (if !found = None then
       let prog = Eval_symbolic.run s.Native_op_walk.Subject.graph in
       match Kernel_adapt.of_stage_program prog with
       | Error _ -> ()
       | Ok kernel -> (
           let bind id = List.assoc_opt id s.Native_op_walk.Subject.inputs in
           let plan = Fusion_plan.default kernel in
           match Err.payload (Loop_lower.lower_unoptimized plan) with
           | Error _ -> ()
           | Ok raw ->
               if has_unit_loop raw.Loop_program.body then
                 found := Some (raw, bind)));
    true
  in
  (match Native_op_walk.find "add" with
  | None -> ()
  | Some m ->
      ignore
        (Walk_core.Walk.run m ~verify ~ppf:silent
           ~pcg:(Walk_core.Pcg.seed ~seed:0L ~seq:1L)
           ~steps:5));
  (match !found with
  | None -> Fmt.pr "no unit loop found in the add walk@."
  | Some (raw, bind) -> (
      let raw_result = Loop_interp.run raw ~bind in
      let broken =
        Loop_opt.run
          ~passes:
            [
              Loop_opt_unit_loops.with_substitute (fun lo ->
                  Loop_index.Add (lo, Loop_index.Const 1));
            ]
          raw
      in
      (* [lo + 1] shifts a size-1 axis's coordinate past its own extent, so
         this is caught even more directly than by [Loop_check]: the
         interpreter's own unchecked-access invariant (design invariant 4)
         raises before any verdict is even computed. *)
      match Loop_interp.run broken ~bind with
      | broken_result ->
          let reference = Err.map_error widen raw_result in
          Fmt.pr "%a@." Loop_check.pp_verdict
            (Loop_check.compare ~reference ~loop:broken_result)
      | exception Invalid_argument m ->
          Fmt.pr "red: raised Invalid_argument %S@." m));
  [%expect
    {| red: raised Invalid_argument "Loop_interp: unchecked access out of range" |}]

(* ---- Stage 2 mutation proof --------------------------------------------- *)

(* [Loop_opt_fold] deliberately does not implement [Scale (0, a) -> Const 0]
   (see its own comment): folding it away would drop [a]'s own overflow along
   with it. [a] here overflows entirely on its own -- two in-domain Consts
   whose sum leaves the domain -- so the scenario needs no loop variable or
   binding to construct. *)

let overflowing_a =
  Loop_index.Add (Loop_index.Const 2_000_000_000, Loop_index.Const 2_000_000_000)

let scale_zero_a = Loop_index.Scale (0, overflowing_a)

let out_buf =
  Loop_fixtures.buffer 0 (Loop_fixtures.shape_w 1) Loop_fixtures.f32
    Loop_buffer.Output

let guarded_program =
  Loop_fixtures.program ~buffers:[ out_buf ]
    [
      Loop_stmt.Fail_if
        ( Loop_bool.Index_overflows scale_zero_a,
          Loop_failure.Index_overflow { index = scale_zero_a } );
      Loop_stmt.Store
        {
          buffer = out_buf;
          coord = Loop_fixtures.at_w (Loop_index.Const 0);
          value = Loop_stored.F32 (Loop_expr.Const 0.);
        };
    ]

let fold_scale_zero : Loop_opt.pass =
 fun program ->
  {
    program with
    Loop_program.body =
      Loop_index_map.stmts
        ~f:(function
          | Loop_index.Scale (0, _) -> Loop_index.Const 0 | idx -> idx)
        program.Loop_program.body;
  }

let%expect_test
    "stage 2 mutation proof: folding Scale (0, a) to Const 0 discards a's own \
     overflow and turns it red" =
  let bind = Loop_fixtures.bind_none in
  let raw_result = Loop_interp.run guarded_program ~bind in
  let broken = Loop_opt.run ~passes:[ fold_scale_zero ] guarded_program in
  let broken_result = Loop_interp.run broken ~bind in
  let reference = Err.map_error widen raw_result in
  Fmt.pr "%a@." Loop_check.pp_verdict
    (Loop_check.compare ~reference ~loop:broken_result);
  [%expect {| DISAGREE: only the reference failed: index_overflow |}]
