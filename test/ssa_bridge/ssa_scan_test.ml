open Ssa_bridge
open Ssa_ir
open Loop_ir_test
open Loop_fixtures
open Loop_programs

(* Region programs and scans, direct: trace locals in a Region, inline [Scan_at]
   in a Pixel body, and the meter that bounds both. The cases are the Loop
   suite's own, against the reference. *)

let data = [| 1.; 2.; 3.; 10.; 20.; 30. |]

let bind id =
  if Tensor_id.equal id (tid 0) then
    Some
      (f32_tensor rows_shape (fun c -> data.((Vec6.offset rows_shape c :> int))))
  else None

let lowered kernel =
  Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
    (Ssa_lower.Ssa_lower_plan.lower (Fusion_plan.default kernel))

let output_of kernel =
  let plan = Fusion_plan.default kernel in
  match Err.payload (Ssa_lower.Ssa_exec.run plan (lowered kernel) ~bind) with
  | Error e -> Fmt.str "failed: %a" Ssa_lower.Ssa_exec.pp_error e
  | Ok m ->
      let t = Tensor_id.Map.find (tid 1) m in
      let cells = ref [] in
      Vec6.iter rows_shape (fun c ->
          cells := Printf.sprintf "%g" (Tensor.read t c) :: !cells);
      String.concat " " (List.rev !cells)

let show name kernel =
  Fmt.pr "%s: %a; %s@." name Ssa_check.pp_verdict
    (Ssa_check.run (Fusion_plan.default kernel) ~bind)
    (output_of kernel)

let%expect_test "a trace local: init row, then charged update rows" =
  show "steps 2" (region_kernel_of (trace_program ~steps:2));
  show "steps 0" (region_kernel_of (trace_program ~steps:0));
  [%expect
    {|
    steps 2: agree; 3 6 9 30 60 90
    steps 0: agree; 1 2 3 10 20 30
    |}]

let%expect_test "an inline scan is re-executed at every cell" =
  show "steps 2" (inline_scan_kernel ~steps:2);
  [%expect {| steps 2: agree; 3 6 9 30 60 90 |}]

(* ---- counters and the meter ------------------------------------------------- *)

type counts = { keys : int; locals : int; scans : int; scan_updates : int }

let reference_counts kernel =
  let counters = Region_execution.counters () in
  ignore
    (Kernel_eval.run
       ~region_counters:(Tensor_id.Map.singleton (tid 1) counters)
       kernel ~bind);
  {
    keys = counters.Region_execution.keys;
    locals = counters.Region_execution.locals;
    scans = counters.Region_execution.scans;
    scan_updates = counters.Region_execution.scan_updates;
  }

let ssa_counts kernel =
  let counters = Ssa_interp.Counters.create () in
  let plan = Fusion_plan.default kernel in
  ignore (Ssa_lower.Ssa_exec.run ~counters plan (lowered kernel) ~bind);
  let m = Ssa_interp.Counters.mark counters in
  {
    keys = m Ssa_mark.Key;
    locals = m Ssa_mark.Local;
    scans = m Ssa_mark.Scan;
    scan_updates = m Ssa_mark.Scan_update;
  }

let%expect_test "a trace local is counted once per key, and updates per lane" =
  let kernel = region_kernel_of (trace_program ~steps:2) in
  let c = ssa_counts kernel in
  Fmt.pr "keys=%d locals=%d scans=%d scan_updates=%d; match: %b@." c.keys
    c.locals c.scans c.scan_updates
    (c = reference_counts kernel);
  [%expect {| keys=2 locals=18 scans=2 scan_updates=12; match: true |}]

(* Keys 2, width 3, steps 2: six updates per key. The meter is fresh for every
   key, so a budget of exactly six passes both keys; sharing one meter across
   keys would fail the second. *)
let%expect_test "the meter resets at every key" =
  let limits = with_scan_limits ~max_state:8192 ~max_updates:6L (fun l -> l) in
  let kernel =
    limited_kernel ~limits (Region_group.Ref.Solo (trace_program ~steps:2))
  in
  Fmt.pr "budget 6: %a; %s@." Ssa_check.pp_verdict
    (Ssa_check.run (Fusion_plan.default kernel) ~bind)
    (output_of kernel);
  [%expect {| budget 6: agree; 3 6 9 30 60 90 |}]

(* A budget below what a scan costs cannot be built into a kernel, so the
   meter's runtime failure is reachable only by evaluating a body directly: the
   reference is [Expr.Eval.value] with a meter of the same limits, the SSA
   program the one lowering produced with its limits replaced. A load beside the
   failing update is counted to show the charge comes BEFORE the update body
   runs. *)
let scan_limits_of ~max_state ~max_updates =
  Err.or_raise ~pp_error:Expr.Scan_limits.pp_error
    (Expr.Scan_limits.create ~max_state ~max_updates)

let reference_at_first_cell ~limits =
  let loads = ref 0 in
  let base = Expr_bridge.env ~binding:bind in
  let env =
    {
      base with
      Expr.Eval.Env.load =
        (fun s c ->
          incr loads;
          base.Expr.Eval.Env.load s c);
    }
  in
  let result =
    Expr.Eval.value
      ~scan_meter:(Expr.Scan_meter.create ~limits)
      env
      ~output:(Expr.Coord.make ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0)
      (inline_scan_body ~steps:2)
  in
  (Err.payload result, !loads)

let ssa_at_first_cell ~limits =
  let kernel = inline_scan_kernel ~steps:2 in
  let plan = Fusion_plan.default kernel in
  let counters = Ssa_interp.Counters.create () in
  let program = { (lowered kernel) with Ssa_program.scan_limits = limits } in
  let result =
    Err.payload (Ssa_lower.Ssa_exec.run ~counters plan program ~bind)
  in
  (result, Ssa_interp.Counters.loads counters)

(* The reference's row for an SSA failure; an invalid program has none. *)
let row (e : Ssa_lower.Ssa_exec.error) : Kernel_eval.error option =
  match e with
  | `Cfg_lowering _ | `Invalid_cfg _ | `Invalid_program _ -> None
  | (`Binding_mismatch _ | `Unbound_input _ | #Ssa_interp.failure) as e ->
      Some (e :> Kernel_eval.error)

let compare_meter ~max_state ~max_updates =
  let limits = scan_limits_of ~max_state ~max_updates in
  let reference, reference_loads = reference_at_first_cell ~limits in
  let ssa, ssa_loads = ssa_at_first_cell ~limits in
  let describe = function
    | Ok _ -> "ok"
    | Error e -> Loop_ir.Loop_check.kind (e :> Kernel_eval.error)
  in
  let same =
    match (reference, ssa) with
    | Ok _, Ok _ -> true
    | Error r, Error s -> (
        match row s with
        | Some s -> Stdlib.( = ) (r :> Kernel_eval.error) s
        | None -> false)
    | _ -> false
  in
  (* loads are compared only where both stop: the reference evaluates one cell
     and the program every cell, so a completed run counts different totals *)
  let loads =
    match (reference, ssa) with
    | Error _, Error _ -> Fmt.str "; loads %d vs %d" reference_loads ssa_loads
    | _ -> ""
  in
  Fmt.pr "updates %Ld, state %d: reference %s, ssa %s; same: %b%s@." max_updates
    max_state (describe reference)
    (match ssa with
    | Ok _ -> "ok"
    | Error e -> (
        match row e with
        | Some e -> Loop_ir.Loop_check.kind e
        | None -> "invalid"))
    same loads

let%expect_test
    "the update budget fails exactly one past its limit, before the body" =
  compare_meter ~max_state:8192 ~max_updates:6L;
  compare_meter ~max_state:8192 ~max_updates:5L;
  compare_meter ~max_state:8192 ~max_updates:1L;
  compare_meter ~max_state:8192 ~max_updates:0L;
  [%expect
    {|
    updates 6, state 8192: reference ok, ssa ok; same: true
    updates 5, state 8192: reference scan_meter, ssa scan_meter; same: true; loads 8 vs 8
    updates 1, state 8192: reference scan_meter, ssa scan_meter; same: true; loads 4 vs 4
    updates 0, state 8192: reference scan_meter, ssa scan_meter; same: true; loads 3 vs 3 |}]

let%expect_test "inline scan state is reserved against the nesting peak" =
  (* width 3 needs 2 * 3 = 6 live cells *)
  compare_meter ~max_state:6 ~max_updates:8192L;
  compare_meter ~max_state:5 ~max_updates:8192L;
  compare_meter ~max_state:0 ~max_updates:8192L;
  [%expect
    {|
    updates 8192, state 6: reference ok, ssa ok; same: true
    updates 8192, state 5: reference scan_meter, ssa scan_meter; same: true; loads 0 vs 0
    updates 8192, state 0: reference scan_meter, ssa scan_meter; same: true; loads 0 vs 0 |}]

(* ---- projections ------------------------------------------------------------ *)

let const_pos n = Expr.Index.assume_position (Expr.Index.const n)

let%expect_test
    "a projection past the trace fails, and the row wins over the lane" =
  let cached ~row ~lane =
    region_kernel_of
      (trace_program_at ~steps:2 ~row:(const_pos row) ~lane:(const_pos lane))
  in
  let inline ~row ~lane =
    region_kernel_of
      (Region_program.pixel
         (inline_scan_body_at ~steps:2 ~row:(const_pos row)
            ~lane:(const_pos lane)))
  in
  List.iter
    (fun (name, kernel) ->
      Fmt.pr "%s: %a@." name Ssa_check.pp_verdict
        (Ssa_check.run (Fusion_plan.default kernel) ~bind))
    [
      ("cached, last cell", cached ~row:2 ~lane:2);
      ("cached, row past", cached ~row:3 ~lane:0);
      ("cached, lane past", cached ~row:0 ~lane:3);
      ("cached, both past", cached ~row:3 ~lane:3);
      ("cached, negative row", cached ~row:(-1) ~lane:0);
      ("inline, last cell", inline ~row:2 ~lane:2);
      ("inline, row past", inline ~row:3 ~lane:0);
      ("inline, lane past", inline ~row:0 ~lane:3);
      ("inline, both past", inline ~row:3 ~lane:3);
    ];
  [%expect
    {|
    cached, last cell: agree
    cached, row past: agree on failure: scan_projection
    cached, lane past: agree on failure: scan_projection
    cached, both past: agree on failure: scan_projection
    cached, negative row: agree on failure: scan_projection
    inline, last cell: agree
    inline, row past: agree on failure: scan_projection
    inline, lane past: agree on failure: scan_projection
    inline, both past: agree on failure: scan_projection |}]

(* ---- two scans in one cell ---------------------------------------------------- *)

(* Two inline scans in sequence share the cell's meter: the first gives its state
   back before the second reserves, so the state budget of one scan suffices,
   while the update budget must cover both. Dropping the release, or the reset
   between cells, shows here. *)
let twice_body =
  let one = inline_scan_body ~steps:2 in
  Expr.Value.add one one

let twice_kernel = region_kernel_of (Region_program.pixel twice_body)

let reference_twice ~limits =
  Err.payload
    (Expr.Eval.value
       ~scan_meter:(Expr.Scan_meter.create ~limits)
       (Expr_bridge.env ~binding:bind)
       ~output:(Expr.Coord.make ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0)
       twice_body)

let ssa_twice ~limits =
  let plan = Fusion_plan.default twice_kernel in
  let program =
    { (lowered twice_kernel) with Ssa_program.scan_limits = limits }
  in
  Err.payload (Ssa_lower.Ssa_exec.run plan program ~bind)

let%expect_test
    "two inline scans in a cell reuse the state budget and sum the updates" =
  List.iter
    (fun (max_state, max_updates) ->
      let limits = scan_limits_of ~max_state ~max_updates in
      let kind = function
        | Ok _ -> "ok"
        | Error e -> Loop_ir.Loop_check.kind (e :> Kernel_eval.error)
      in
      let ssa =
        match ssa_twice ~limits with
        | Ok _ -> "ok"
        | Error e -> (
            match row e with
            | Some e -> Loop_ir.Loop_check.kind e
            | None -> "invalid")
      in
      Fmt.pr "state %d, updates %Ld: reference %s, ssa %s@." max_state
        max_updates
        (kind (reference_twice ~limits))
        ssa)
    [ (6, 12L); (5, 12L); (6, 11L); (12, 12L) ];
  [%expect
    {|
    state 6, updates 12: reference ok, ssa ok
    state 5, updates 12: reference scan_meter, ssa scan_meter
    state 6, updates 11: reference scan_meter, ssa scan_meter
    state 12, updates 12: reference ok, ssa ok |}]

(* ---- the trace's own meter, against the Loop lowering's --------------------- *)

(* A trace local charges once per lane update before the update runs. Its budget
   cannot be set below what the kernel needs (admission rejects it), so the
   program's limits are replaced as for the inline scan, and the reference is the
   Loop lowering of the same kernel: an independent implementation of the same
   rules, compared on the failure and on how many loads ran before it. *)
let%expect_test "a trace charges before each update, as the Loop program does" =
  let kernel = region_kernel_of (trace_program ~steps:2) in
  let plan = Fusion_plan.default kernel in
  let ssa_program = lowered kernel in
  let loop_program =
    match Err.payload (Loop_ir.Loop_lower.lower plan) with
    | Ok p -> p
    | Error _ -> failwith "refused"
  in
  List.iter
    (fun max_updates ->
      let limits = scan_limits_of ~max_state:8192 ~max_updates in
      let ssa_counters = Ssa_interp.Counters.create () in
      let ssa =
        Err.payload
          (Ssa_lower.Ssa_exec.run ~counters:ssa_counters plan
             { ssa_program with Ssa_program.scan_limits = limits }
             ~bind)
      in
      let loop_counters = Loop_ir.Loop_interp.counters () in
      let loop =
        Err.payload
          (Loop_ir.Loop_interp.run ~counters:loop_counters
             { loop_program with Loop_ir.Loop_program.scan_limits = limits }
             ~bind)
      in
      let kind_ssa =
        match ssa with
        | Ok _ -> "ok"
        | Error e -> (
            match row e with
            | Some e -> Loop_ir.Loop_check.kind e
            | None -> "invalid")
      and kind_loop =
        match loop with
        | Ok _ -> "ok"
        | Error e -> Loop_ir.Loop_check.kind (e :> Kernel_eval.error)
      in
      Fmt.pr "updates %Ld: ssa %s after %d loads, loop %s after %d loads@."
        max_updates kind_ssa
        (Ssa_interp.Counters.loads ssa_counters)
        kind_loop loop_counters.Loop_ir.Loop_interp.loads)
    [ 6L; 5L; 1L; 0L ];
  [%expect
    {|
    updates 6: ssa ok after 18 loads, loop ok after 18 loads
    updates 5: ssa scan_meter after 8 loads, loop scan_meter after 8 loads
    updates 1: ssa scan_meter after 4 loads, loop scan_meter after 4 loads
    updates 0: ssa scan_meter after 3 loads, loop scan_meter after 3 loads |}]
