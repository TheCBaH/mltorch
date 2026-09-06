(* [Region_program.preflight]: the resource dimensions [check] does not cover
   -- total local/trace storage, peak nested scan state, and one Region key's
   worst-case recurrence-update count. See .ai/native_scan_design.md's
   "Validated execution artifact". *)

open Expr

let scan_limits =
  Err.or_raise ~pp_error:Scan_limits.pp_error
    (Scan_limits.create ~max_state:100 ~max_updates:1000L)

let pos n = Index.clamp_low (Index.const n)
let singleton = Region_partition.singleton

let pp_unit =
  Core.Pretty.err_result
    ~ok:(fun ppf () -> Fmt.string ppf "ok")
    ~error:Region_program.pp_error

(* trace.(0,l) = 0; trace.(s+1,l) = trace.(s,l) + 1 -- the same counter
   [region_scan_construction_test.ml] uses. *)
let counter_program ~width ~steps ~partition =
  Region_program.Builder.run
    (Region_program.Builder.scan ~limits:scan_limits ~width ~steps
       ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
       ~update:(fun ~step:_ ~lane ~previous_at ->
         Builder.return (Value.add (previous_at lane) (Value.const 1.)))
       (fun scan_read ->
         Region_program.Builder.finish ~max_size:64 ~max_depth:16 ~partition
           ~output:(scan_read ~row:(pos steps) ~lane:(pos 0))))

let scan_program ~width ~steps =
  Err.or_raise ~pp_error:Region_program.pp_error
    (counter_program ~width ~steps ~partition:singleton)

let ones = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1

(* [width=2, steps=3]: [trace_slot_total = (steps+1)*width = 8],
   [per_key = steps*width = 6] (the cached [Local_scan_at] emitter read costs
   no update of its own). *)
let%expect_test "preflight admits a program within every configured bound" =
  let program = scan_program ~width:2 ~steps:3 in
  Fmt.pr "%a@." pp_unit
    (Region_program.preflight ~max_local_slots:8 ~max_scan_state:8
       ~max_scan_updates:6L ~output_shape:ones program);
  [%expect {| ok |}]

let%expect_test "preflight rejects total local/trace storage over the limit" =
  let program = scan_program ~width:2 ~steps:3 in
  Fmt.pr "%a@." pp_unit
    (Region_program.preflight ~max_local_slots:7 ~max_scan_state:8
       ~max_scan_updates:6L ~output_shape:ones program);
  [%expect {| total local slot count exceeds limit 7 |}]

let%expect_test "preflight rejects peak scan state over the limit" =
  let program = scan_program ~width:2 ~steps:3 in
  Fmt.pr "%a@." pp_unit
    (Region_program.preflight ~max_local_slots:8 ~max_scan_state:7
       ~max_scan_updates:6L ~output_shape:ones program);
  [%expect {| scan state exceeds limit 7 |}]

let%expect_test "preflight rejects one key's recurrence updates over the limit"
    =
  let program = scan_program ~width:2 ~steps:3 in
  Fmt.pr "%a@." pp_unit
    (Region_program.preflight ~max_local_slots:8 ~max_scan_state:8
       ~max_scan_updates:5L ~output_shape:ones program);
  [%expect {| region key's scan updates exceed limit 5 |}]

(* A [Whole] output axis multiplies [outputs_per_key]: the SAME trace is
   materialized once per key, but the emitter -- here just the cached trace
   read -- runs once per output sharing it. Costing 0 per read, this alone
   would not move [per_key]; the point is that a real emitter WOULD be
   charged [outputs_per_key] times, so the wiring from [output_shape] and the
   program's own partition through to [per_key] has to be exercised even when
   this particular emitter is a free ride. *)
let%expect_test "preflight derives outputs_per_key from the whole axes" =
  let partition =
    Err.or_raise ~pp_error:Region_partition.pp_error
      (Region_partition.of_whole_axes [ Axis.T ])
  in
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (counter_program ~width:2 ~steps:3 ~partition)
  in
  let shape = Vec6.shape ~n:1 ~t:5 ~d:1 ~h:1 ~w:1 ~c:1 in
  Fmt.pr "%a@." pp_unit
    (Region_program.preflight ~max_local_slots:8 ~max_scan_state:8
       ~max_scan_updates:6L ~output_shape:shape program);
  [%expect {| ok |}]

(* [max_scan_state = 0] disables scans entirely -- an existing scan-free
   program must never be rejected by it, per the scan design record's own
   invariant. *)
let%expect_test "a scan-free program passes preflight with scans disabled" =
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.Builder.run
         (Region_program.Builder.scalar (Value.const 1.) (fun read ->
              Region_program.Builder.finish ~max_size:8 ~max_depth:8
                ~partition:singleton ~output:read)))
  in
  Fmt.pr "%a@." pp_unit
    (Region_program.preflight ~max_local_slots:0 ~max_scan_state:0
       ~max_scan_updates:0L ~output_shape:ones program);
  [%expect {| total local slot count exceeds limit 0 |}]

let%expect_test "raising max_local_slots alone admits the scan-free program" =
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.Builder.run
         (Region_program.Builder.scalar (Value.const 1.) (fun read ->
              Region_program.Builder.finish ~max_size:8 ~max_depth:8
                ~partition:singleton ~output:read)))
  in
  Fmt.pr "%a@." pp_unit
    (Region_program.preflight ~max_local_slots:1 ~max_scan_state:0
       ~max_scan_updates:0L ~output_shape:ones program);
  [%expect {| ok |}]
