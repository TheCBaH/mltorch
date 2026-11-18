(* The transfer model against the subsystem: every candidate plan of each
   matrix product, operands in SRAM, measured on subsystem-vector-normal
   beside the cycles {!Coral_transfer.Transfer_model} predicts for it, and
   the plan it chooses against the measured best. Bitwise throughout. About
   five hours of simulation, so a target of its own (make
   coralnpu.transfer.runtest), outside the subsystem suite. *)

module T = Machine_model_test.Model_test
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral
module P = Coral_transfer.Transfer_plan
module Tm = Coral_transfer.Transfer_model

let subsystem =
  {
    C.Config.dir = Sys.getenv "CORALNPU_SUBSYSTEM_DATA";
    configuration = "subsystem-vector-normal";
    deadline = 1800.;
    flags = [];
  }

let home (s : Machine_rivet_common.Rivet_table.slot) =
  match s.Machine_rivet_common.Rivet_table.section with
  | Machine_model.Mir_artifact.Section.Bound -> Coral_layout.Home.Ram
  | _ -> Coral_layout.Home.Any

(* The suite runs for hours inside one expect test, whose stdout and stderr are
   both captured until it ends. Progress goes to a duplicate of stderr taken
   before the capture starts, so a CI log shows the case being simulated. *)
let progress_out = Unix.out_channel_of_descr (Unix.dup Unix.stderr)
let start = Unix.gettimeofday ()

let progress name =
  let now = Unix.gettimeofday () in
  let t = Unix.gmtime now in
  Printf.fprintf progress_out "%04d-%02d-%02dT%02d:%02d:%02dZ +%.0fs %s\n%!"
    (t.Unix.tm_year + 1900) (t.Unix.tm_mon + 1) t.Unix.tm_mday t.Unix.tm_hour
    t.Unix.tm_min t.Unix.tm_sec (now -. start) name

let measure ~prepack ~numerics ~plan name g =
  progress name;
  let cycles = ref 0L and seen = ref None in
  let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
  let transfer ~budget d =
    seen := Some (d, budget);
    match plan with Some p -> p ~budget d | None -> Tm.choose ~budget d
  in
  T.check
    ~route:(C.route ~transfer ~home ~on_run ~budget:32768L subsystem)
    ~contractions:true ~prepack
    ~pipeline:
      (Ssa_backends.Pipeline.Planned
         { numerics; target = Ssa_ir.Ssa_target.zve32f128 })
    ~oracle:
      (match numerics with
      | Ssa_ir.Ssa_numerics.Fp32_contracted -> T.Fp32_contracted
      | _ -> T.Fp32)
    ~constant:T.positive ~calls:1 name g;
  (!cycles, !seen)

(* Every candidate plan of each product, measured beside its prediction;
   the plan chosen against the measured best, and what choosing it costs
   over the best; the worst error. *)
let grid shapes =
  let errors = ref [] in
  List.iter
    (fun (name, prepack, numerics, g) ->
      (* the descriptor and the budget the route offers, from a first run *)
      let _, seen =
        measure ~prepack ~numerics ~plan:None (name ^ ", chosen") g
      in
      match seen with
      | None -> Fmt.pr "%s: no contraction@." name
      | Some (d, budget) ->
          let chosen = Tm.choose ~budget d in
          let runs =
            List.map
              (fun p ->
                let c, _ =
                  measure ~prepack ~numerics
                    ~plan:(Some (fun ~budget:_ _ -> p))
                    (Fmt.str "%s, %a" name P.pp p)
                    g
                in
                let predicted = Tm.predict d p in
                let error =
                  100. *. (predicted -. Int64.to_float c) /. Int64.to_float c
                in
                errors := error :: !errors;
                Fmt.pr "  measured %Ld, predicted %.0f (%+.1f%%)@." c predicted
                  error;
                (p, c))
              (P.candidates ~budget d)
          in
          let best, bc =
            List.fold_left
              (fun (bp, bc) (p, c) -> if c < bc then (p, c) else (bp, bc))
              (List.hd runs) runs
          in
          let cc = List.assoc chosen runs in
          Fmt.pr "  chosen: %a; measured best: %a; chosen/best %.3f@." P.pp
            chosen P.pp best
            (Int64.to_float cc /. Int64.to_float bc))
    shapes;
  let worst =
    List.fold_left (fun a e -> Float.max a (Float.abs e)) 0. !errors
  in
  Fmt.pr "worst error %.1f%% over %d plans@." worst (List.length !errors)

let fused = Ssa_ir.Ssa_numerics.Fp32_contracted
and ordered = Ssa_ir.Ssa_numerics.Fp32_ordered

(* the 16 shapes the rates are fitted to: full and tail tiles of 8 to 32
   lanes, both requirements, B strided and contiguous, A from 9 to 192 rows *)
let shapes =
  [
    ( "linear 48x32 k64",
      false,
      fused,
      T.linear_fixture ~m:48 ~n:32 ~k:64 ~bias:false );
    ( "linear 48x32 k64 packed",
      true,
      fused,
      T.linear_fixture ~m:48 ~n:32 ~k:64 ~bias:false );
    ( "linear 48x32 k64 packed, ordered",
      true,
      ordered,
      T.linear_fixture ~m:48 ~n:32 ~k:64 ~bias:false );
    ( "linear 192x32 k128 packed",
      true,
      fused,
      T.linear_fixture ~m:192 ~n:32 ~k:128 ~bias:false );
    ("bmm 1x9x32 k64", false, fused, T.bmm_fixture ~batch:1 ~m:9 ~n:32 ~k:64);
    ( "linear 24x64 k32 packed",
      true,
      fused,
      T.linear_fixture ~m:24 ~n:64 ~k:32 ~bias:false );
    ( "linear 36x48 k96 packed",
      true,
      fused,
      T.linear_fixture ~m:36 ~n:48 ~k:96 ~bias:false );
    ( "linear 60x32 k64, ordered",
      false,
      ordered,
      T.linear_fixture ~m:60 ~n:32 ~k:64 ~bias:false );
    ( "linear 96x32 k64 packed",
      true,
      fused,
      T.linear_fixture ~m:96 ~n:32 ~k:64 ~bias:false );
    ("bmm 1x12x64 k32", false, fused, T.bmm_fixture ~batch:1 ~m:12 ~n:64 ~k:32);
    ( "linear 30x40 k48 packed, ordered",
      true,
      ordered,
      T.linear_fixture ~m:30 ~n:40 ~k:48 ~bias:false );
    ( "linear 45x56 k80 packed",
      true,
      fused,
      T.linear_fixture ~m:45 ~n:56 ~k:80 ~bias:false );
    ( "linear 33x20 k40, ordered",
      false,
      ordered,
      T.linear_fixture ~m:33 ~n:20 ~k:40 ~bias:false );
    ("bmm 1x18x40 k48", false, fused, T.bmm_fixture ~batch:1 ~m:18 ~n:40 ~k:48);
    ( "linear 120x24 k64 packed, ordered",
      true,
      ordered,
      T.linear_fixture ~m:120 ~n:24 ~k:64 ~bias:false );
    ( "linear 21x72 k56 packed",
      true,
      fused,
      T.linear_fixture ~m:21 ~n:72 ~k:56 ~bias:false );
  ]

(* The CI cut: the shapes whose candidates are fewest, so about thirty
   simulator runs where the full grid is some two hundred, which cost about
   five hours. *)
let reduced =
  List.filter
    (fun (name, _, _, _) ->
      List.mem name
        [
          "bmm 1x9x32 k64"; "linear 36x48 k96 packed"; "linear 45x56 k80 packed";
        ])
    shapes

let%expect_test "the transfer model against the subsystem, reduced" =
  grid reduced;
  [%expect
    {|
    bmm 1x9x32 k64, chosen (1 invocations): bitwise
    bmm 1x9x32 k64, A in place, B in place (1 invocations): bitwise
      measured 7128, predicted 6879 (-3.5%)
    bmm 1x9x32 k64, A copied, B in place (1 invocations): bitwise
      measured 7333, predicted 7246 (-1.2%)
    bmm 1x9x32 k64, A streamed by 9 rows, B in place (1 invocations): bitwise
      measured 7370, predicted 7246 (-1.7%)
    bmm 1x9x32 k64, A streamed by 9 rows, B in place, C by the DMA (1 invocations): bitwise
      measured 8590, predicted 8173 (-4.9%)
    bmm 1x9x32 k64, A in place, B copied (1 invocations): bitwise
      measured 13278, predicted 13382 (+0.8%)
    bmm 1x9x32 k64, A copied, B copied (1 invocations): bitwise
      measured 13446, predicted 13994 (+4.1%)
    bmm 1x9x32 k64, A streamed by 9 rows, B copied (1 invocations): bitwise
      measured 13526, predicted 13994 (+3.5%)
    bmm 1x9x32 k64, A streamed by 9 rows, B copied, C by the DMA (1 invocations): bitwise
      measured 14746, predicted 14921 (+1.2%)
    bmm 1x9x32 k64, A in place, B shared (1 invocations): bitwise
      measured 8327, predicted 8696 (+4.4%)
    bmm 1x9x32 k64, A copied, B shared (1 invocations): bitwise
      measured 8494, predicted 9308 (+9.6%)
    bmm 1x9x32 k64, A streamed by 9 rows, B shared (1 invocations): bitwise
      measured 8575, predicted 9308 (+8.5%)
    bmm 1x9x32 k64, A streamed by 9 rows, B shared, C by the DMA (1 invocations): bitwise
      measured 9795, predicted 10235 (+4.5%)
      chosen: A in place, B in place; measured best: A in place, B in place; chosen/best 1.000
    linear 36x48 k96 packed, chosen (1 invocations, 1 packed): bitwise
    linear 36x48 k96 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 73616, predicted 77654 (+5.5%)
    linear 36x48 k96 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 64173, predicted 69340 (+8.1%)
    linear 36x48 k96 packed, A streamed by 24 rows, B in place (1 invocations, 1 packed): bitwise
      measured 61686, predicted 67299 (+9.1%)
    linear 36x48 k96 packed, A streamed by 15 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 61838, predicted 68771 (+11.2%)
    linear 36x48 k96 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 82954, predicted 87341 (+5.3%)
    linear 36x48 k96 packed, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 71834, predicted 76869 (+7.0%)
      chosen: A streamed by 24 rows, B in place; measured best: A streamed by 24 rows, B in place; chosen/best 1.000
    linear 45x56 k80 packed, chosen (1 invocations, 1 packed): bitwise
    linear 45x56 k80 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 88570, predicted 83306 (-5.9%)
    linear 45x56 k80 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 81251, predicted 74644 (-8.1%)
    linear 45x56 k80 packed, A streamed by 30 rows, B in place (1 invocations, 1 packed): bitwise
      measured 77457, predicted 72518 (-6.4%)
    linear 45x56 k80 packed, A streamed by 18 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 78746, predicted 75453 (-4.2%)
    linear 45x56 k80 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 98602, predicted 91147 (-7.6%)
    linear 45x56 k80 packed, A streamed by 3 rows, B copied (1 invocations, 1 packed): bitwise
      measured 79045, predicted 73274 (-7.3%)
    linear 45x56 k80 packed, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 87893, predicted 80911 (-7.9%)
    linear 45x56 k80 packed, A streamed by 3 rows, B shared (1 invocations, 1 packed): bitwise
      measured 68331, predicted 63038 (-7.7%)
      chosen: A streamed by 3 rows, B shared; measured best: A streamed by 3 rows, B shared; chosen/best 1.000
    worst error 11.2% over 26 plans |}]

let%expect_test ("the transfer model against the subsystem" [@tags "disabled"])
    =
  grid shapes;
  [%expect
    {|
    linear 48x32 k64, chosen (1 invocations): bitwise
    linear 48x32 k64, A in place, B in place (1 invocations): bitwise
      measured 194026, predicted 194174 (+0.1%)
    linear 48x32 k64, A copied, B in place (1 invocations): bitwise
      measured 192554, predicted 193420 (+0.4%)
    linear 48x32 k64, A streamed by 39 rows, B in place (1 invocations): bitwise
      measured 193810, predicted 193560 (-0.1%)
    linear 48x32 k64, A streamed by 24 rows, B in place, C by the DMA (1 invocations): bitwise
      measured 194021, predicted 191065 (-1.5%)
    linear 48x32 k64, A in place, B copied (1 invocations): bitwise
      measured 70187, predicted 71212 (+1.5%)
    linear 48x32 k64, A copied, B copied (1 invocations): bitwise
      measured 71857, predicted 70458 (-1.9%)
    linear 48x32 k64, A streamed by 24 rows, B copied (1 invocations): bitwise
      measured 66617, predicted 65082 (-2.3%)
    linear 48x32 k64, A streamed by 15 rows, B copied, C by the DMA (1 invocations): bitwise
      measured 65343, predicted 60737 (-7.0%)
    linear 48x32 k64, A in place, B shared (1 invocations): bitwise
      measured 65555, predicted 66526 (+1.5%)
    linear 48x32 k64, A copied, B shared (1 invocations): bitwise
      measured 66664, predicted 65772 (-1.3%)
    linear 48x32 k64, A streamed by 24 rows, B shared (1 invocations): bitwise
      measured 61979, predicted 60396 (-2.6%)
    linear 48x32 k64, A streamed by 15 rows, B shared, C by the DMA (1 invocations): bitwise
      measured 60708, predicted 56050 (-7.7%)
      chosen: A streamed by 15 rows, B shared, C by the DMA; measured best: A streamed by 15 rows, B shared, C by the DMA; chosen/best 1.000
    linear 48x32 k64 packed, chosen (1 invocations, 1 packed): bitwise
    linear 48x32 k64 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 71146, predicted 70622 (-0.7%)
    linear 48x32 k64 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 69674, predicted 69868 (+0.3%)
    linear 48x32 k64 packed, A streamed by 39 rows, B in place (1 invocations, 1 packed): bitwise
      measured 70930, predicted 70009 (-1.3%)
    linear 48x32 k64 packed, A streamed by 24 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 71140, predicted 67514 (-5.1%)
    linear 48x32 k64 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 45691, predicted 46193 (+1.1%)
    linear 48x32 k64 packed, A copied, B copied (1 invocations, 1 packed): bitwise
      measured 47361, predicted 46745 (-1.3%)
    linear 48x32 k64 packed, A streamed by 24 rows, B copied (1 invocations, 1 packed): bitwise
      measured 42121, predicted 41369 (-1.8%)
    linear 48x32 k64 packed, A streamed by 15 rows, B copied, C by the DMA (1 invocations, 1 packed): bitwise
      measured 40846, predicted 37024 (-9.4%)
    linear 48x32 k64 packed, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 41059, predicted 41507 (+1.1%)
    linear 48x32 k64 packed, A copied, B shared (1 invocations, 1 packed): bitwise
      measured 42168, predicted 42059 (-0.3%)
    linear 48x32 k64 packed, A streamed by 24 rows, B shared (1 invocations, 1 packed): bitwise
      measured 37483, predicted 36683 (-2.1%)
    linear 48x32 k64 packed, A streamed by 15 rows, B shared, C by the DMA (1 invocations, 1 packed): bitwise
      measured 36211, predicted 32338 (-10.7%)
      chosen: A streamed by 15 rows, B shared, C by the DMA; measured best: A streamed by 15 rows, B shared, C by the DMA; chosen/best 1.000
    linear 48x32 k64 packed, ordered, chosen (1 invocations, 1 packed): bitwise
    linear 48x32 k64 packed, ordered, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 92986, predicted 94972 (+2.1%)
    linear 48x32 k64 packed, ordered, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 91514, predicted 94218 (+3.0%)
    linear 48x32 k64 packed, ordered, A streamed by 40 rows, B in place (1 invocations, 1 packed): bitwise
      measured 92994, predicted 94638 (+1.8%)
    linear 48x32 k64 packed, ordered, A streamed by 26 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 93427, predicted 92478 (-1.0%)
    linear 48x32 k64 packed, ordered, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 52915, predicted 56984 (+7.7%)
    linear 48x32 k64 packed, ordered, A copied, B copied (1 invocations, 1 packed): bitwise
      measured 66489, predicted 67793 (+2.0%)
    linear 48x32 k64 packed, ordered, A streamed by 24 rows, B copied (1 invocations, 1 packed): bitwise
      measured 61249, predicted 62417 (+1.9%)
    linear 48x32 k64 packed, ordered, A streamed by 16 rows, B copied, C by the DMA (1 invocations, 1 packed): bitwise
      measured 61301, predicted 59864 (-2.3%)
    linear 48x32 k64 packed, ordered, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 48283, predicted 52298 (+8.3%)
    linear 48x32 k64 packed, ordered, A copied, B shared (1 invocations, 1 packed): bitwise
      measured 61295, predicted 63107 (+3.0%)
    linear 48x32 k64 packed, ordered, A streamed by 24 rows, B shared (1 invocations, 1 packed): bitwise
      measured 56610, predicted 57731 (+2.0%)
    linear 48x32 k64 packed, ordered, A streamed by 16 rows, B shared, C by the DMA (1 invocations, 1 packed): bitwise
      measured 56662, predicted 55177 (-2.6%)
      chosen: A in place, B shared; measured best: A in place, B shared; chosen/best 1.000
    linear 192x32 k128 packed, chosen (1 invocations, 1 packed): bitwise
    linear 192x32 k128 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 554841, predicted 545833 (-1.6%)
    linear 192x32 k128 packed, A streamed by 18 rows, B in place (1 invocations, 1 packed): bitwise
      measured 465364, predicted 481356 (+3.4%)
    linear 192x32 k128 packed, A streamed by 15 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 459740, predicted 470461 (+2.3%)
    linear 192x32 k128 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 307418, predicted 306995 (-0.1%)
    linear 192x32 k128 packed, A streamed by 3 rows, B copied (1 invocations, 1 packed): bitwise
      measured 214599, predicted 226337 (+5.5%)
    linear 192x32 k128 packed, A streamed by 3 rows, B copied, C by the DMA (1 invocations, 1 packed): bitwise
      measured 208193, predicted 209457 (+0.6%)
    linear 192x32 k128 packed, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 298161, predicted 297680 (-0.2%)
    linear 192x32 k128 packed, A streamed by 3 rows, B shared (1 invocations, 1 packed): bitwise
      measured 205335, predicted 217021 (+5.7%)
    linear 192x32 k128 packed, A streamed by 3 rows, B shared, C by the DMA (1 invocations, 1 packed): bitwise
      measured 198930, predicted 200142 (+0.6%)
      chosen: A streamed by 3 rows, B shared, C by the DMA; measured best: A streamed by 3 rows, B shared, C by the DMA; chosen/best 1.000
    bmm 1x9x32 k64, chosen (1 invocations): bitwise
    bmm 1x9x32 k64, A in place, B in place (1 invocations): bitwise
      measured 13376, predicted 13242 (-1.0%)
    bmm 1x9x32 k64, A copied, B in place (1 invocations): bitwise
      measured 13153, predicted 13147 (-0.0%)
    bmm 1x9x32 k64, A streamed by 9 rows, B in place (1 invocations): bitwise
      measured 13189, predicted 13147 (-0.3%)
    bmm 1x9x32 k64, A streamed by 9 rows, B in place, C by the DMA (1 invocations): bitwise
      measured 14628, predicted 13386 (-8.5%)
    bmm 1x9x32 k64, A in place, B copied (1 invocations): bitwise
      measured 14480, predicted 14532 (+0.4%)
    bmm 1x9x32 k64, A copied, B copied (1 invocations): bitwise
      measured 14231, predicted 14681 (+3.2%)
    bmm 1x9x32 k64, A streamed by 9 rows, B copied (1 invocations): bitwise
      measured 14299, predicted 14681 (+2.7%)
    bmm 1x9x32 k64, A streamed by 9 rows, B copied, C by the DMA (1 invocations): bitwise
      measured 15738, predicted 14921 (-5.2%)
    bmm 1x9x32 k64, A in place, B shared (1 invocations): bitwise
      measured 9848, predicted 9845 (-0.0%)
    bmm 1x9x32 k64, A copied, B shared (1 invocations): bitwise
      measured 9038, predicted 9995 (+10.6%)
    bmm 1x9x32 k64, A streamed by 9 rows, B shared (1 invocations): bitwise
      measured 9667, predicted 9995 (+3.4%)
    bmm 1x9x32 k64, A streamed by 9 rows, B shared, C by the DMA (1 invocations): bitwise
      measured 11106, predicted 10235 (-7.8%)
      chosen: A in place, B shared; measured best: A copied, B shared; chosen/best 1.090
    linear 24x64 k32 packed, chosen (1 invocations, 1 packed): bitwise
    linear 24x64 k32 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 38439, predicted 37704 (-1.9%)
    linear 24x64 k32 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 35049, predicted 34668 (-1.1%)
    linear 24x64 k32 packed, A streamed by 24 rows, B in place (1 invocations, 1 packed): bitwise
      measured 35082, predicted 34668 (-1.2%)
    linear 24x64 k32 packed, A streamed by 24 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 41872, predicted 35699 (-14.7%)
    linear 24x64 k32 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 29368, predicted 29102 (-0.9%)
    linear 24x64 k32 packed, A copied, B copied (1 invocations, 1 packed): bitwise
      measured 25951, predicted 26719 (+3.0%)
    linear 24x64 k32 packed, A streamed by 24 rows, B copied (1 invocations, 1 packed): bitwise
      measured 26018, predicted 26719 (+2.7%)
    linear 24x64 k32 packed, A streamed by 15 rows, B copied, C by the DMA (1 invocations, 1 packed): bitwise
      measured 31916, predicted 26741 (-16.2%)
    linear 24x64 k32 packed, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 24736, predicted 24416 (-1.3%)
    linear 24x64 k32 packed, A copied, B shared (1 invocations, 1 packed): bitwise
      measured 20758, predicted 22032 (+6.1%)
    linear 24x64 k32 packed, A streamed by 24 rows, B shared (1 invocations, 1 packed): bitwise
      measured 21385, predicted 22032 (+3.0%)
    linear 24x64 k32 packed, A streamed by 15 rows, B shared, C by the DMA (1 invocations, 1 packed): bitwise
      measured 27281, predicted 22055 (-19.2%)
      chosen: A copied, B shared; measured best: A copied, B shared; chosen/best 1.000
    linear 36x48 k96 packed, chosen (1 invocations, 1 packed): bitwise
    linear 36x48 k96 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 129289, predicted 134245 (+3.8%)
    linear 36x48 k96 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 113802, predicted 120381 (+5.8%)
    linear 36x48 k96 packed, A streamed by 24 rows, B in place (1 invocations, 1 packed): bitwise
      measured 113502, predicted 118340 (+4.3%)
    linear 36x48 k96 packed, A streamed by 15 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 112054, predicted 115690 (+3.2%)
    linear 36x48 k96 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 93907, predicted 97014 (+3.3%)
    linear 36x48 k96 packed, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 83494, predicted 86541 (+3.6%)
      chosen: A in place, B shared; measured best: A in place, B shared; chosen/best 1.000
    linear 60x32 k64, ordered, chosen (1 invocations): bitwise
    linear 60x32 k64, ordered, A in place, B in place (1 invocations): bitwise
      measured 346594, predicted 350375 (+1.1%)
    linear 60x32 k64, ordered, A copied, B in place (1 invocations): bitwise
      measured 344739, predicted 349418 (+1.4%)
    linear 60x32 k64, ordered, A streamed by 40 rows, B in place (1 invocations): bitwise
      measured 344321, predicted 347150 (+0.8%)
    linear 60x32 k64, ordered, A streamed by 26 rows, B in place, C by the DMA (1 invocations): bitwise
      measured 342786, predicted 344478 (+0.5%)
    linear 60x32 k64, ordered, A in place, B copied (1 invocations): bitwise
      measured 108067, predicted 113499 (+5.0%)
    linear 60x32 k64, ordered, A streamed by 24 rows, B copied (1 invocations): bitwise
      measured 102283, predicted 104478 (+2.1%)
    linear 60x32 k64, ordered, A streamed by 16 rows, B copied, C by the DMA (1 invocations): bitwise
      measured 101414, predicted 100376 (-1.0%)
    linear 60x32 k64, ordered, A in place, B shared (1 invocations): bitwise
      measured 103435, predicted 108813 (+5.2%)
    linear 60x32 k64, ordered, A streamed by 24 rows, B shared (1 invocations): bitwise
      measured 97644, predicted 99792 (+2.2%)
    linear 60x32 k64, ordered, A streamed by 16 rows, B shared, C by the DMA (1 invocations): bitwise
      measured 96777, predicted 95690 (-1.1%)
      chosen: A streamed by 16 rows, B shared, C by the DMA; measured best: A streamed by 16 rows, B shared, C by the DMA; chosen/best 1.000
    linear 96x32 k64 packed, chosen (1 invocations, 1 packed): bitwise
    linear 96x32 k64 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 142137, predicted 141244 (-0.6%)
    linear 96x32 k64 packed, A streamed by 39 rows, B in place (1 invocations, 1 packed): bitwise
      measured 132868, predicted 131224 (-1.2%)
    linear 96x32 k64 packed, A streamed by 24 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 131249, predicted 126152 (-3.9%)
    linear 96x32 k64 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 83994, predicted 85162 (+1.4%)
    linear 96x32 k64 packed, A streamed by 24 rows, B copied (1 invocations, 1 packed): bitwise
      measured 71437, predicted 70080 (-1.9%)
    linear 96x32 k64 packed, A streamed by 15 rows, B copied, C by the DMA (1 invocations, 1 packed): bitwise
      measured 69815, predicted 61668 (-11.7%)
    linear 96x32 k64 packed, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 79362, predicted 80475 (+1.4%)
    linear 96x32 k64 packed, A streamed by 24 rows, B shared (1 invocations, 1 packed): bitwise
      measured 66799, predicted 65394 (-2.1%)
    linear 96x32 k64 packed, A streamed by 15 rows, B shared, C by the DMA (1 invocations, 1 packed): bitwise
      measured 65181, predicted 56982 (-12.6%)
      chosen: A streamed by 15 rows, B shared, C by the DMA; measured best: A streamed by 15 rows, B shared, C by the DMA; chosen/best 1.000
    bmm 1x12x64 k32, chosen (1 invocations): bitwise
    bmm 1x12x64 k32, A in place, B in place (1 invocations): bitwise
      measured 19111, predicted 18852 (-1.4%)
    bmm 1x12x64 k32, A copied, B in place (1 invocations): bitwise
      measured 17449, predicted 17362 (-0.5%)
    bmm 1x12x64 k32, A streamed by 12 rows, B in place (1 invocations): bitwise
      measured 17485, predicted 17362 (-0.7%)
    bmm 1x12x64 k32, A streamed by 12 rows, B in place, C by the DMA (1 invocations): bitwise
      measured 21205, predicted 17906 (-15.6%)
    bmm 1x12x64 k32, A in place, B copied (1 invocations): bitwise
      measured 18192, predicted 18164 (-0.2%)
    bmm 1x12x64 k32, A copied, B copied (1 invocations): bitwise
      measured 16503, predicted 17000 (+3.0%)
    bmm 1x12x64 k32, A streamed by 12 rows, B copied (1 invocations): bitwise
      measured 16571, predicted 17000 (+2.6%)
    bmm 1x12x64 k32, A streamed by 12 rows, B copied, C by the DMA (1 invocations): bitwise
      measured 20291, predicted 17544 (-13.5%)
    bmm 1x12x64 k32, A in place, B shared (1 invocations): bitwise
      measured 13560, predicted 13477 (-0.6%)
    bmm 1x12x64 k32, A copied, B shared (1 invocations): bitwise
      measured 11310, predicted 12314 (+8.9%)
    bmm 1x12x64 k32, A streamed by 12 rows, B shared (1 invocations): bitwise
      measured 11939, predicted 12314 (+3.1%)
    bmm 1x12x64 k32, A streamed by 12 rows, B shared, C by the DMA (1 invocations): bitwise
      measured 15659, predicted 12858 (-17.9%)
      chosen: A copied, B shared; measured best: A copied, B shared; chosen/best 1.000
    linear 30x40 k48 packed, ordered, chosen (1 invocations, 1 packed): bitwise
    linear 30x40 k48 packed, ordered, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 67624, predicted 70223 (+3.8%)
    linear 30x40 k48 packed, ordered, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 61209, predicted 64741 (+5.8%)
    linear 30x40 k48 packed, ordered, A streamed by 30 rows, B in place (1 invocations, 1 packed): bitwise
      measured 61244, predicted 64741 (+5.7%)
    linear 30x40 k48 packed, ordered, A streamed by 28 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 67028, predicted 67352 (+0.5%)
    linear 30x40 k48 packed, ordered, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 46645, predicted 50648 (+8.6%)
    linear 30x40 k48 packed, ordered, A copied, B copied (1 invocations, 1 packed): bitwise
      measured 50898, predicted 55745 (+9.5%)
    linear 30x40 k48 packed, ordered, A streamed by 30 rows, B copied (1 invocations, 1 packed): bitwise
      measured 54176, predicted 55745 (+2.9%)
    linear 30x40 k48 packed, ordered, A streamed by 18 rows, B copied, C by the DMA (1 invocations, 1 packed): bitwise
      measured 55075, predicted 54546 (-1.0%)
    linear 30x40 k48 packed, ordered, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 42247, predicted 46198 (+9.4%)
    linear 30x40 k48 packed, ordered, A copied, B shared (1 invocations, 1 packed): bitwise
      measured 46136, predicted 51295 (+11.2%)
    linear 30x40 k48 packed, ordered, A streamed by 30 rows, B shared (1 invocations, 1 packed): bitwise
      measured 49778, predicted 51295 (+3.0%)
    linear 30x40 k48 packed, ordered, A streamed by 18 rows, B shared, C by the DMA (1 invocations, 1 packed): bitwise
      measured 50674, predicted 50097 (-1.1%)
      chosen: A in place, B shared; measured best: A in place, B shared; chosen/best 1.000
    linear 45x56 k80 packed, chosen (1 invocations, 1 packed): bitwise
    linear 45x56 k80 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 156245, predicted 152119 (-2.6%)
    linear 45x56 k80 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 146545, predicted 137675 (-6.1%)
    linear 45x56 k80 packed, A streamed by 30 rows, B in place (1 invocations, 1 packed): bitwise
      measured 142453, predicted 135549 (-4.8%)
    linear 45x56 k80 packed, A streamed by 18 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 141431, predicted 132472 (-6.3%)
    linear 45x56 k80 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 111376, predicted 102941 (-7.6%)
    linear 45x56 k80 packed, A streamed by 3 rows, B copied (1 invocations, 1 packed): bitwise
      measured 84417, predicted 79286 (-6.1%)
    linear 45x56 k80 packed, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 101199, predicted 92704 (-8.4%)
    linear 45x56 k80 packed, A streamed by 3 rows, B shared (1 invocations, 1 packed): bitwise
      measured 74234, predicted 69049 (-7.0%)
      chosen: A streamed by 3 rows, B shared; measured best: A streamed by 3 rows, B shared; chosen/best 1.000
    linear 33x20 k40, ordered, chosen (1 invocations): bitwise
    linear 33x20 k40, ordered, A in place, B in place (1 invocations): bitwise
      measured 85839, predicted 84507 (-1.6%)
    linear 33x20 k40, ordered, A copied, B in place (1 invocations): bitwise
      measured 85244, predicted 84216 (-1.2%)
    linear 33x20 k40, ordered, A streamed by 34 rows, B in place (1 invocations): bitwise
      measured 85279, predicted 84216 (-1.2%)
    linear 33x20 k40, ordered, A streamed by 34 rows, B in place, C by the DMA (1 invocations): bitwise
      measured 88483, predicted 84691 (-4.3%)
    linear 33x20 k40, ordered, A in place, B copied (1 invocations): bitwise
      measured 34305, predicted 33331 (-2.8%)
    linear 33x20 k40, ordered, A copied, B copied (1 invocations): bitwise
      measured 33683, predicted 33040 (-1.9%)
    linear 33x20 k40, ordered, A streamed by 34 rows, B copied (1 invocations): bitwise
      measured 33751, predicted 33040 (-2.1%)
    linear 33x20 k40, ordered, A streamed by 34 rows, B copied, C by the DMA (1 invocations): bitwise
      measured 36955, predicted 33515 (-9.3%)
    linear 33x20 k40, ordered, A in place, B shared (1 invocations): bitwise
      measured 32448, predicted 31426 (-3.1%)
    linear 33x20 k40, ordered, A copied, B shared (1 invocations): bitwise
      measured 31728, predicted 31135 (-1.9%)
    linear 33x20 k40, ordered, A streamed by 34 rows, B shared (1 invocations): bitwise
      measured 31894, predicted 31135 (-2.4%)
    linear 33x20 k40, ordered, A streamed by 34 rows, B shared, C by the DMA (1 invocations): bitwise
      measured 35099, predicted 31610 (-9.9%)
      chosen: A copied, B shared; measured best: A copied, B shared; chosen/best 1.000
    bmm 1x18x40 k48, chosen (1 invocations): bitwise
    bmm 1x18x40 k48, A in place, B in place (1 invocations): bitwise
      measured 32092, predicted 32050 (-0.1%)
    bmm 1x18x40 k48, A copied, B in place (1 invocations): bitwise
      measured 28269, predicted 28627 (+1.3%)
    bmm 1x18x40 k48, A streamed by 18 rows, B in place (1 invocations): bitwise
      measured 28305, predicted 28627 (+1.1%)
    bmm 1x18x40 k48, A streamed by 18 rows, B in place, C by the DMA (1 invocations): bitwise
      measured 31803, predicted 29140 (-8.4%)
    bmm 1x18x40 k48, A in place, B copied (1 invocations): bitwise
      measured 27406, predicted 27698 (+1.1%)
    bmm 1x18x40 k48, A copied, B copied (1 invocations): bitwise
      measured 23556, predicted 24642 (+4.6%)
    bmm 1x18x40 k48, A streamed by 18 rows, B copied (1 invocations): bitwise
      measured 23625, predicted 24642 (+4.3%)
    bmm 1x18x40 k48, A streamed by 18 rows, B copied, C by the DMA (1 invocations): bitwise
      measured 27123, predicted 25156 (-7.3%)
    bmm 1x18x40 k48, A in place, B shared (1 invocations): bitwise
      measured 23009, predicted 23249 (+1.0%)
    bmm 1x18x40 k48, A copied, B shared (1 invocations): bitwise
      measured 18795, predicted 20193 (+7.4%)
    bmm 1x18x40 k48, A streamed by 18 rows, B shared (1 invocations): bitwise
      measured 19228, predicted 20193 (+5.0%)
    bmm 1x18x40 k48, A streamed by 18 rows, B shared, C by the DMA (1 invocations): bitwise
      measured 22726, predicted 20706 (-8.9%)
      chosen: A copied, B shared; measured best: A copied, B shared; chosen/best 1.000
    linear 120x24 k64 packed, ordered, chosen (1 invocations, 1 packed): bitwise
    linear 120x24 k64 packed, ordered, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 210031, predicted 202322 (-3.7%)
    linear 120x24 k64 packed, ordered, A streamed by 40 rows, B in place (1 invocations, 1 packed): bitwise
      measured 197266, predicted 186855 (-5.3%)
    linear 120x24 k64 packed, ordered, A streamed by 28 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 194282, predicted 183086 (-5.8%)
    linear 120x24 k64 packed, ordered, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 127933, predicted 123229 (-3.7%)
    linear 120x24 k64 packed, ordered, A streamed by 28 rows, B copied (1 invocations, 1 packed): bitwise
      measured 142378, predicted 129558 (-9.0%)
    linear 120x24 k64 packed, ordered, A streamed by 20 rows, B copied, C by the DMA (1 invocations, 1 packed): bitwise
      measured 140566, predicted 122929 (-12.5%)
    linear 120x24 k64 packed, ordered, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 124454, predicted 119700 (-3.8%)
    linear 120x24 k64 packed, ordered, A streamed by 28 rows, B shared (1 invocations, 1 packed): bitwise
      measured 138893, predicted 126029 (-9.3%)
    linear 120x24 k64 packed, ordered, A streamed by 20 rows, B shared, C by the DMA (1 invocations, 1 packed): bitwise
      measured 137085, predicted 119400 (-12.9%)
      chosen: A streamed by 20 rows, B shared, C by the DMA; measured best: A in place, B shared; chosen/best 1.101
    linear 21x72 k56 packed, chosen (1 invocations, 1 packed): bitwise
    linear 21x72 k56 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 70483, predicted 70484 (+0.0%)
    linear 21x72 k56 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 60553, predicted 61378 (+1.4%)
    linear 21x72 k56 packed, A streamed by 21 rows, B in place (1 invocations, 1 packed): bitwise
      measured 60587, predicted 61378 (+1.3%)
    linear 21x72 k56 packed, A streamed by 18 rows, B in place, C by the DMA (1 invocations, 1 packed): bitwise
      measured 67325, predicted 63796 (-5.2%)
    linear 21x72 k56 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 56528, predicted 57389 (+1.5%)
    linear 21x72 k56 packed, A streamed by 9 rows, B copied (1 invocations, 1 packed): bitwise
      measured 48174, predicted 46930 (-2.6%)
    linear 21x72 k56 packed, A streamed by 3 rows, B copied, C by the DMA (1 invocations, 1 packed): bitwise
      measured 47035, predicted 42989 (-8.6%)
    linear 21x72 k56 packed, A in place, B shared (1 invocations, 1 packed): bitwise
      measured 47388, predicted 48191 (+1.7%)
    linear 21x72 k56 packed, A streamed by 9 rows, B shared (1 invocations, 1 packed): bitwise
      measured 39030, predicted 37733 (-3.3%)
    linear 21x72 k56 packed, A streamed by 3 rows, B shared, C by the DMA (1 invocations, 1 packed): bitwise
      measured 37889, predicted 33792 (-10.8%)
      chosen: A streamed by 3 rows, B shared, C by the DMA; measured best: A streamed by 3 rows, B shared, C by the DMA; chosen/best 1.000
    worst error 19.2% over 169 plans |}]
