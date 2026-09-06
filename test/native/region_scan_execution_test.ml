(* Region scan EXECUTION -- the step-8 half of the scan design record's Stage
   1/2 split. [region_scan_construction_test.ml] covers construction,
   checking and rendering only (execution was not landed yet); this covers
   actually RUNNING a scan-backed Region program: trace correctness,
   reference/production agreement, shared per-key meter accounting, and
   genuine runtime update exhaustion -- not just preflight rejection. *)

open Expr

let limits =
  Err.or_raise ~pp_error:Scan_limits.pp_error
    (Scan_limits.create ~max_state:100 ~max_updates:1000L)

let partition = Region_partition.singleton

(* trace.(0,l) = 0; trace.(s+1,l) = trace.(s,l) + 1, so trace.(row,lane) = row
   for every lane -- an easy oracle independent of [lane]. The output reads
   [row]/[lane] from the OUTPUT coordinate itself (C = row, W = lane), so
   under [Region_partition.singleton] (every axis Singleton) each distinct
   (row, lane) output coordinate is its own Region key: the whole trace is
   recomputed once per key, exercising the per-key meter reset across many
   keys, not just one. *)
let counter_program ~width ~steps =
  Region_program.Builder.run
    (Region_program.Builder.scan ~limits ~width ~steps
       ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
       ~update:(fun ~step:_ ~lane ~previous_at ->
         Builder.return (Value.add (previous_at lane) (Value.const 1.)))
       (fun scan_read ->
         Region_program.Builder.finish ~max_size:64 ~max_depth:16 ~partition
           ~output:
             (scan_read ~row:(Index.output Axis.C) ~lane:(Index.output Axis.W))))

let program ~width ~steps =
  Err.or_raise ~pp_error:Region_program.pp_error (counter_program ~width ~steps)

let shape ~rows ~lanes = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:lanes ~c:rows

let env =
  {
    Eval.Env.load = (fun _ _ -> assert false);
    load_index = (fun _ _ -> assert false);
  }

let production_lowered ~scan_limits ~width ~steps ~output_shape =
  Err.or_raise ~pp_error:Region_program.pp_error
    (Region_execution.lower_region ~max_size:64 ~max_depth:16
       ~max_local_slots:8192 ~scan_limits ~output_shape (program ~width ~steps))

let print_tensor ~rows ~lanes t =
  for row = 0 to rows - 1 do
    for lane = 0 to lanes - 1 do
      Fmt.pr "%g"
        (Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:lane ~c:row));
      if lane < lanes - 1 then Fmt.pr ","
    done;
    Fmt.pr "@."
  done

let%expect_test
    "production and reference materialize agree with the counter oracle" =
  let width = 2 and steps = 3 in
  let rows = steps + 1 and lanes = width in
  let output_shape = shape ~rows ~lanes in
  let lowered =
    production_lowered ~scan_limits:limits ~width ~steps ~output_shape
  in
  let production =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_execution.materialize lowered ~env)
  in
  let reference =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_eval.materialize ~scan_limits:limits (program ~width ~steps)
         ~output_shape ~env)
  in
  Fmt.pr "production:@.";
  print_tensor ~rows ~lanes production;
  Fmt.pr "reference:@.";
  print_tensor ~rows ~lanes reference;
  Fmt.pr "bitwise_equal=%b@." (Tensor.equal_bits production reference);
  [%expect
    {|
    production:
    0,0
    1,1
    2,2
    3,3
    reference:
    0,0
    1,1
    2,2
    3,3
    bitwise_equal=true |}]

let%expect_test
    "counters observe scan starts and exact combined update charges, per key" =
  let width = 2 and steps = 3 in
  let rows = steps + 1 and lanes = width in
  let output_shape = shape ~rows ~lanes in
  let lowered =
    production_lowered ~scan_limits:limits ~width ~steps ~output_shape
  in
  let counters = Region_execution.counters () in
  let (_ : Tensor.packed) =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_execution.materialize ~counters lowered ~env)
  in
  let keys = rows * lanes in
  Fmt.pr "keys=%d scans=%d scan_updates=%d locals=%d emitters=%d@."
    counters.keys counters.scans counters.scan_updates counters.locals
    counters.emitters;
  (* One key per (row, lane) output; each key independently re-runs the whole
     trace ([steps*width] charged update lane-evaluations, plus [width]
     uncharged init lane-evaluations -- both counted in [locals]) since no
     sharing exists yet at this stage. *)
  Fmt.pr "expected: keys=%d scans=%d scan_updates=%d locals=%d emitters=%d@."
    keys keys
    (keys * steps * width)
    (keys * (width + (steps * width)))
    keys;
  [%expect
    {|
    keys=8 scans=8 scan_updates=48 locals=64 emitters=8
    expected: keys=8 scans=8 scan_updates=48 locals=64 emitters=8 |}]

let%expect_test "value_at agrees with materialize and is independent per call" =
  let width = 2 and steps = 3 in
  let output_shape = shape ~rows:(steps + 1) ~lanes:width in
  let lowered =
    production_lowered ~scan_limits:limits ~width ~steps ~output_shape
  in
  let output = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:1 ~c:2 in
  let first =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_execution.value_at lowered ~env ~output)
  in
  (* Called again, immediately: a shared (rather than freshly reset)
     per-invocation meter would make this SECOND call see an already-consumed
     budget and fail where the first succeeded. *)
  let second =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_execution.value_at lowered ~env ~output)
  in
  Fmt.pr "first=%g second=%g@." first second;
  [%expect {| first=2 second=2 |}]

let%expect_test
    "a scan-free program's runtime meter is never touched by an unrelated \
     tight limit" =
  (* Not a scan test per se, but pins that reducing [max_scan_state] to 0 --
     which forbids every inline [Scan_at] -- does not, by itself, reject a
     TRACE local's own execution: the trace writes directly into its
     preflighted slot range and never calls [Scan_meter.reserve]/[release] at
     all, unlike an inline projection's rolling two-row buffer. *)
  let width = 2 and steps = 3 in
  let output_shape = shape ~rows:(steps + 1) ~lanes:width in
  let zero_state =
    Err.or_raise ~pp_error:Scan_limits.pp_error
      (Scan_limits.create ~max_state:0 ~max_updates:1000L)
  in
  let result =
    Region_eval.materialize ~scan_limits:zero_state (program ~width ~steps)
      ~output_shape ~env
  in
  Fmt.pr "%a@."
    (Core.Pretty.err_result
       ~ok:(fun fmt _ -> Fmt.string fmt "ok")
       ~error:Region_eval.pp_error)
    result;
  [%expect {| ok |}]

let%expect_test "the runtime meter genuinely exhausts, not just preflight" =
  (* [Region_eval.materialize] never preflights (only [Region_execution.lower]
     does), so a [scan_limits] tighter than what construction would even admit
     reaches the METER, proving [Updates_exhausted] is reachable at runtime
     and not merely a preflight-time rejection under another name. *)
  let width = 2 and steps = 3 in
  let output_shape = shape ~rows:(steps + 1) ~lanes:width in
  let tight =
    Err.or_raise ~pp_error:Scan_limits.pp_error
      (Scan_limits.create ~max_state:100 ~max_updates:5L)
  in
  let result =
    Region_eval.materialize ~scan_limits:tight (program ~width ~steps)
      ~output_shape ~env
  in
  Fmt.pr "%a@."
    (Core.Pretty.err_result
       ~ok:(fun fmt _ -> Fmt.string fmt "ok")
       ~error:Region_eval.pp_error)
    result;
  [%expect {| scan updates exhausted at limit 5 |}]

let%expect_test "an exactly-per-key limit still succeeds across many keys" =
  (* If the meter were wrongly SHARED across keys instead of reset per key,
     only the first of 8 keys would fit inside a limit sized for exactly one
     key's own cost ([steps*width = 6]); all 8 succeeding proves the reset. *)
  let width = 2 and steps = 3 in
  let output_shape = shape ~rows:(steps + 1) ~lanes:width in
  let exactly_one_key =
    Err.or_raise ~pp_error:Scan_limits.pp_error
      (Scan_limits.create ~max_state:100
         ~max_updates:(Int64.of_int (steps * width)))
  in
  let result =
    Region_eval.materialize ~scan_limits:exactly_one_key (program ~width ~steps)
      ~output_shape ~env
  in
  Fmt.pr "%a@."
    (Core.Pretty.err_result
       ~ok:(fun fmt _ -> Fmt.string fmt "ok")
       ~error:Region_eval.pp_error)
    result;
  [%expect {| ok |}]
