open Loop_ir
open Loop_fixtures
open Loop_programs

(* Scans: trace locals in a Region, inline [Scan_at] in a Pixel body, and the
   meter that bounds both. *)

let data = [| 1.; 2.; 3.; 10.; 20.; 30. |]

let bind id =
  if Tensor_id.equal id (tid 0) then
    Some
      (f32_tensor rows_shape (fun c -> data.((Vec6.offset rows_shape c :> int))))
  else None

let output_of kernel =
  match Err.payload (Loop_lower.lower (Fusion_plan.default kernel)) with
  | Error (`Unsupported u) -> Fmt.str "refused: %a" Loop_unsupported.pp u
  | Ok program -> (
      match Err.payload (Loop_interp.run program ~bind) with
      | Error e -> Fmt.str "failed: %a" Loop_interp.pp_error e
      | Ok m ->
          let t = Tensor_id.Map.find (tid 1) m in
          let cells = ref [] in
          Vec6.iter rows_shape (fun c ->
              cells := Printf.sprintf "%g" (Tensor.read t c) :: !cells);
          String.concat " " (List.rev !cells))

let show name kernel =
  Fmt.pr "%s: %a; %s@." name Loop_check.pp_verdict
    (Loop_check.run (Fusion_plan.default kernel) ~bind)
    (output_of kernel)

let%expect_test "a trace local: init row, then charged update rows" =
  show "steps 2" (region_kernel_of (trace_program ~steps:2));
  show "steps 0" (region_kernel_of (trace_program ~steps:0));
  [%expect
    {|
    steps 2: agree; 3 6 9 30 60 90
    steps 0: agree; 1 2 3 10 20 30 |}]

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

let loop_counts kernel =
  let counters = Loop_interp.counters () in
  (match Err.payload (Loop_lower.lower (Fusion_plan.default kernel)) with
  | Ok program -> ignore (Loop_interp.run ~counters program ~bind)
  | Error _ -> ());
  {
    keys = counters.Loop_interp.keys;
    locals = counters.Loop_interp.locals;
    scans = counters.Loop_interp.scans;
    scan_updates = counters.Loop_interp.scan_updates;
  }

let%expect_test "a trace local is counted once per key, and updates per lane" =
  let kernel = region_kernel_of (trace_program ~steps:2) in
  let c = loop_counts kernel in
  Fmt.pr "keys=%d locals=%d scans=%d scan_updates=%d; match: %b@." c.keys
    c.locals c.scans c.scan_updates
    (c = reference_counts kernel);
  [%expect {| keys=2 locals=18 scans=2 scan_updates=12; match: true |}]

(* Keys 2, width 3, steps 2: six updates per key. The meter is fresh for every
   key, so a budget of exactly six passes both keys; sharing one meter across keys
   would fail the second. A smaller budget cannot be observed here: [Kernel.create]
   rejects a Region program whose per-key updates exceed it. *)
let%expect_test "the meter resets at every key" =
  let run max_updates =
    let limits = with_scan_limits ~max_state:8192 ~max_updates (fun l -> l) in
    let kernel =
      limited_kernel ~limits (Region_group.Ref.Solo (trace_program ~steps:2))
    in
    Fmt.pr "budget %Ld: %a; %s@." max_updates Loop_check.pp_verdict
      (Loop_check.run (Fusion_plan.default kernel) ~bind)
      (output_of kernel)
  in
  run 6L;
  [%expect {| budget 6: agree; 3 6 9 30 60 90 |}]

(* A budget below what a scan costs cannot be built into a kernel:
   [Kernel.create] runs the admission preflight, so the meter's runtime failure is
   reachable only by evaluating a body directly. The reference here is
   [Expr.Eval.value] with a meter of the same limits; the Loop program is the one
   lowering produced, with its limits replaced. A load beside the failing update
   is counted to show the charge comes BEFORE the update body runs. *)
let inline_program () =
  match
    Err.payload
      (Loop_lower.lower (Fusion_plan.default (inline_scan_kernel ~steps:2)))
  with
  | Ok p -> p
  | Error _ -> failwith "refused"

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

let loop_at_first_cell ~limits =
  let counters = Loop_interp.counters () in
  let program =
    { (inline_program ()) with Loop_program.scan_limits = limits }
  in
  let result = Err.payload (Loop_interp.run ~counters program ~bind) in
  (result, counters.Loop_interp.loads)

let compare_meter ~max_state ~max_updates =
  let limits = scan_limits_of ~max_state ~max_updates in
  let reference, reference_loads = reference_at_first_cell ~limits in
  let loop, loop_loads = loop_at_first_cell ~limits in
  let describe = function
    | Ok _ -> "ok"
    | Error e -> Loop_check.kind (e :> Kernel_eval.error)
  in
  let same =
    match (reference, loop) with
    | Ok _, Ok _ -> true
    | Error r, Error l ->
        Stdlib.( = ) (r :> Kernel_eval.error) (l :> Kernel_eval.error)
    | _ -> false
  in
  (* Loads are compared only where both stop: the reference evaluates one cell
     and the program every cell, so a completed run counts different totals. *)
  let loads =
    match (reference, loop) with
    | Error _, Error _ -> Fmt.str "; loads %d vs %d" reference_loads loop_loads
    | _ -> ""
  in
  Fmt.pr "updates %Ld, state %d: reference %s, loop %s; same: %b%s@."
    max_updates max_state (describe reference) (describe loop) same loads

let%expect_test
    "the update budget fails exactly one past its limit, before the body" =
  compare_meter ~max_state:8192 ~max_updates:6L;
  compare_meter ~max_state:8192 ~max_updates:5L;
  compare_meter ~max_state:8192 ~max_updates:1L;
  compare_meter ~max_state:8192 ~max_updates:0L;
  [%expect
    {|
    updates 6, state 8192: reference ok, loop ok; same: true
    updates 5, state 8192: reference scan_meter, loop scan_meter; same: true; loads 8 vs 8
    updates 1, state 8192: reference scan_meter, loop scan_meter; same: true; loads 4 vs 4
    updates 0, state 8192: reference scan_meter, loop scan_meter; same: true; loads 3 vs 3 |}]

let%expect_test "inline scan state is reserved against the nesting peak" =
  (* width 3 needs 2 * 3 = 6 live cells. *)
  compare_meter ~max_state:6 ~max_updates:8192L;
  compare_meter ~max_state:5 ~max_updates:8192L;
  compare_meter ~max_state:0 ~max_updates:8192L;
  [%expect
    {|
    updates 8192, state 6: reference ok, loop ok; same: true
    updates 8192, state 5: reference scan_meter, loop scan_meter; same: true; loads 0 vs 0
    updates 8192, state 0: reference scan_meter, loop scan_meter; same: true; loads 0 vs 0 |}]

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
      Fmt.pr "%s: %a@." name Loop_check.pp_verdict
        (Loop_check.run (Fusion_plan.default kernel) ~bind))
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
