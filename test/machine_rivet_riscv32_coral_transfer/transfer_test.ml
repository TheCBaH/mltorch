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

let measure ~prepack ~numerics ~plan name g =
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
let%expect_test "the transfer model against the subsystem" =
  grid
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
    ];
  [%expect
    {|
    linear 48x32 k64, chosen (1 invocations): bitwise
    linear 48x32 k64, A in place, B in place (1 invocations): bitwise
      measured 194026, predicted 194174 (+0.1%)
    linear 48x32 k64, A copied, B in place (1 invocations): bitwise
      measured 192554, predicted 193420 (+0.4%)
    linear 48x32 k64, A streamed by 39 rows, B in place (1 invocations): bitwise
      measured 193810, predicted 193560 (-0.1%)
    linear 48x32 k64, A in place, B copied (1 invocations): bitwise
      measured 70187, predicted 71212 (+1.5%)
    linear 48x32 k64, A copied, B copied (1 invocations): bitwise
      measured 71857, predicted 70458 (-1.9%)
    linear 48x32 k64, A streamed by 24 rows, B copied (1 invocations): bitwise
      measured 66617, predicted 65082 (-2.3%)
      chosen: A streamed by 24 rows, B copied; measured best: A streamed by 24 rows, B copied; chosen/best 1.000
    linear 48x32 k64 packed, chosen (1 invocations, 1 packed): bitwise
    linear 48x32 k64 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 71146, predicted 70622 (-0.7%)
    linear 48x32 k64 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 69674, predicted 69868 (+0.3%)
    linear 48x32 k64 packed, A streamed by 39 rows, B in place (1 invocations, 1 packed): bitwise
      measured 70930, predicted 70009 (-1.3%)
    linear 48x32 k64 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 45691, predicted 46193 (+1.1%)
    linear 48x32 k64 packed, A copied, B copied (1 invocations, 1 packed): bitwise
      measured 47361, predicted 46745 (-1.3%)
    linear 48x32 k64 packed, A streamed by 24 rows, B copied (1 invocations, 1 packed): bitwise
      measured 42121, predicted 41369 (-1.8%)
      chosen: A streamed by 24 rows, B copied; measured best: A streamed by 24 rows, B copied; chosen/best 1.000
    linear 48x32 k64 packed, ordered, chosen (1 invocations, 1 packed): bitwise
    linear 48x32 k64 packed, ordered, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 92986, predicted 94972 (+2.1%)
    linear 48x32 k64 packed, ordered, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 91514, predicted 94218 (+3.0%)
    linear 48x32 k64 packed, ordered, A streamed by 40 rows, B in place (1 invocations, 1 packed): bitwise
      measured 92994, predicted 94638 (+1.8%)
    linear 48x32 k64 packed, ordered, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 52915, predicted 56984 (+7.7%)
    linear 48x32 k64 packed, ordered, A copied, B copied (1 invocations, 1 packed): bitwise
      measured 66489, predicted 67793 (+2.0%)
    linear 48x32 k64 packed, ordered, A streamed by 24 rows, B copied (1 invocations, 1 packed): bitwise
      measured 61249, predicted 62417 (+1.9%)
      chosen: A in place, B copied; measured best: A in place, B copied; chosen/best 1.000
    linear 192x32 k128 packed, chosen (1 invocations, 1 packed): bitwise
    linear 192x32 k128 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 554841, predicted 545833 (-1.6%)
    linear 192x32 k128 packed, A streamed by 18 rows, B in place (1 invocations, 1 packed): bitwise
      measured 465364, predicted 481356 (+3.4%)
    linear 192x32 k128 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 307418, predicted 306995 (-0.1%)
    linear 192x32 k128 packed, A streamed by 3 rows, B copied (1 invocations, 1 packed): bitwise
      measured 214599, predicted 226337 (+5.5%)
      chosen: A streamed by 3 rows, B copied; measured best: A streamed by 3 rows, B copied; chosen/best 1.000
    bmm 1x9x32 k64, chosen (1 invocations): bitwise
    bmm 1x9x32 k64, A in place, B in place (1 invocations): bitwise
      measured 13376, predicted 13242 (-1.0%)
    bmm 1x9x32 k64, A copied, B in place (1 invocations): bitwise
      measured 13153, predicted 13147 (-0.0%)
    bmm 1x9x32 k64, A streamed by 9 rows, B in place (1 invocations): bitwise
      measured 13189, predicted 13147 (-0.3%)
    bmm 1x9x32 k64, A in place, B copied (1 invocations): bitwise
      measured 14480, predicted 14532 (+0.4%)
    bmm 1x9x32 k64, A copied, B copied (1 invocations): bitwise
      measured 14231, predicted 14681 (+3.2%)
    bmm 1x9x32 k64, A streamed by 9 rows, B copied (1 invocations): bitwise
      measured 14299, predicted 14681 (+2.7%)
      chosen: A copied, B in place; measured best: A copied, B in place; chosen/best 1.000
    linear 24x64 k32 packed, chosen (1 invocations, 1 packed): bitwise
    linear 24x64 k32 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 38439, predicted 37704 (-1.9%)
    linear 24x64 k32 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 35049, predicted 34668 (-1.1%)
    linear 24x64 k32 packed, A streamed by 24 rows, B in place (1 invocations, 1 packed): bitwise
      measured 35082, predicted 34668 (-1.2%)
    linear 24x64 k32 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 29368, predicted 29102 (-0.9%)
    linear 24x64 k32 packed, A copied, B copied (1 invocations, 1 packed): bitwise
      measured 25951, predicted 26719 (+3.0%)
    linear 24x64 k32 packed, A streamed by 24 rows, B copied (1 invocations, 1 packed): bitwise
      measured 26018, predicted 26719 (+2.7%)
      chosen: A copied, B copied; measured best: A copied, B copied; chosen/best 1.000
    linear 36x48 k96 packed, chosen (1 invocations, 1 packed): bitwise
    linear 36x48 k96 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 129289, predicted 134245 (+3.8%)
    linear 36x48 k96 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 113802, predicted 120381 (+5.8%)
    linear 36x48 k96 packed, A streamed by 24 rows, B in place (1 invocations, 1 packed): bitwise
      measured 113502, predicted 118340 (+4.3%)
    linear 36x48 k96 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 93907, predicted 97014 (+3.3%)
      chosen: A in place, B copied; measured best: A in place, B copied; chosen/best 1.000
    linear 60x32 k64, ordered, chosen (1 invocations): bitwise
    linear 60x32 k64, ordered, A in place, B in place (1 invocations): bitwise
      measured 346594, predicted 350375 (+1.1%)
    linear 60x32 k64, ordered, A copied, B in place (1 invocations): bitwise
      measured 344739, predicted 349418 (+1.4%)
    linear 60x32 k64, ordered, A streamed by 40 rows, B in place (1 invocations): bitwise
      measured 344321, predicted 347150 (+0.8%)
    linear 60x32 k64, ordered, A in place, B copied (1 invocations): bitwise
      measured 108067, predicted 113499 (+5.0%)
    linear 60x32 k64, ordered, A streamed by 24 rows, B copied (1 invocations): bitwise
      measured 102283, predicted 104478 (+2.1%)
      chosen: A streamed by 24 rows, B copied; measured best: A streamed by 24 rows, B copied; chosen/best 1.000
    linear 96x32 k64 packed, chosen (1 invocations, 1 packed): bitwise
    linear 96x32 k64 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 142137, predicted 141244 (-0.6%)
    linear 96x32 k64 packed, A streamed by 39 rows, B in place (1 invocations, 1 packed): bitwise
      measured 132868, predicted 131224 (-1.2%)
    linear 96x32 k64 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 83994, predicted 85162 (+1.4%)
    linear 96x32 k64 packed, A streamed by 24 rows, B copied (1 invocations, 1 packed): bitwise
      measured 71437, predicted 70080 (-1.9%)
      chosen: A streamed by 24 rows, B copied; measured best: A streamed by 24 rows, B copied; chosen/best 1.000
    bmm 1x12x64 k32, chosen (1 invocations): bitwise
    bmm 1x12x64 k32, A in place, B in place (1 invocations): bitwise
      measured 19111, predicted 18852 (-1.4%)
    bmm 1x12x64 k32, A copied, B in place (1 invocations): bitwise
      measured 17449, predicted 17362 (-0.5%)
    bmm 1x12x64 k32, A streamed by 12 rows, B in place (1 invocations): bitwise
      measured 17485, predicted 17362 (-0.7%)
    bmm 1x12x64 k32, A in place, B copied (1 invocations): bitwise
      measured 18192, predicted 18164 (-0.2%)
    bmm 1x12x64 k32, A copied, B copied (1 invocations): bitwise
      measured 16503, predicted 17000 (+3.0%)
    bmm 1x12x64 k32, A streamed by 12 rows, B copied (1 invocations): bitwise
      measured 16571, predicted 17000 (+2.6%)
      chosen: A copied, B copied; measured best: A copied, B copied; chosen/best 1.000
    linear 30x40 k48 packed, ordered, chosen (1 invocations, 1 packed): bitwise
    linear 30x40 k48 packed, ordered, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 67624, predicted 70223 (+3.8%)
    linear 30x40 k48 packed, ordered, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 61209, predicted 64741 (+5.8%)
    linear 30x40 k48 packed, ordered, A streamed by 30 rows, B in place (1 invocations, 1 packed): bitwise
      measured 61244, predicted 64741 (+5.7%)
    linear 30x40 k48 packed, ordered, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 46645, predicted 50648 (+8.6%)
    linear 30x40 k48 packed, ordered, A copied, B copied (1 invocations, 1 packed): bitwise
      measured 50898, predicted 55745 (+9.5%)
    linear 30x40 k48 packed, ordered, A streamed by 30 rows, B copied (1 invocations, 1 packed): bitwise
      measured 54176, predicted 55745 (+2.9%)
      chosen: A in place, B copied; measured best: A in place, B copied; chosen/best 1.000
    linear 45x56 k80 packed, chosen (1 invocations, 1 packed): bitwise
    linear 45x56 k80 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 156245, predicted 152119 (-2.6%)
    linear 45x56 k80 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 146545, predicted 137675 (-6.1%)
    linear 45x56 k80 packed, A streamed by 30 rows, B in place (1 invocations, 1 packed): bitwise
      measured 142453, predicted 135549 (-4.8%)
    linear 45x56 k80 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 111376, predicted 102941 (-7.6%)
    linear 45x56 k80 packed, A streamed by 3 rows, B copied (1 invocations, 1 packed): bitwise
      measured 84417, predicted 79286 (-6.1%)
      chosen: A streamed by 3 rows, B copied; measured best: A streamed by 3 rows, B copied; chosen/best 1.000
    linear 33x20 k40, ordered, chosen (1 invocations): bitwise
    linear 33x20 k40, ordered, A in place, B in place (1 invocations): bitwise
      measured 85839, predicted 84507 (-1.6%)
    linear 33x20 k40, ordered, A copied, B in place (1 invocations): bitwise
      measured 85244, predicted 84216 (-1.2%)
    linear 33x20 k40, ordered, A streamed by 34 rows, B in place (1 invocations): bitwise
      measured 85279, predicted 84216 (-1.2%)
    linear 33x20 k40, ordered, A in place, B copied (1 invocations): bitwise
      measured 34305, predicted 33331 (-2.8%)
    linear 33x20 k40, ordered, A copied, B copied (1 invocations): bitwise
      measured 33683, predicted 33040 (-1.9%)
    linear 33x20 k40, ordered, A streamed by 34 rows, B copied (1 invocations): bitwise
      measured 33751, predicted 33040 (-2.1%)
      chosen: A copied, B copied; measured best: A copied, B copied; chosen/best 1.000
    bmm 1x18x40 k48, chosen (1 invocations): bitwise
    bmm 1x18x40 k48, A in place, B in place (1 invocations): bitwise
      measured 32092, predicted 32050 (-0.1%)
    bmm 1x18x40 k48, A copied, B in place (1 invocations): bitwise
      measured 28269, predicted 28627 (+1.3%)
    bmm 1x18x40 k48, A streamed by 18 rows, B in place (1 invocations): bitwise
      measured 28305, predicted 28627 (+1.1%)
    bmm 1x18x40 k48, A in place, B copied (1 invocations): bitwise
      measured 27406, predicted 27698 (+1.1%)
    bmm 1x18x40 k48, A copied, B copied (1 invocations): bitwise
      measured 23556, predicted 24642 (+4.6%)
    bmm 1x18x40 k48, A streamed by 18 rows, B copied (1 invocations): bitwise
      measured 23625, predicted 24642 (+4.3%)
      chosen: A copied, B copied; measured best: A copied, B copied; chosen/best 1.000
    linear 120x24 k64 packed, ordered, chosen (1 invocations, 1 packed): bitwise
    linear 120x24 k64 packed, ordered, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 210031, predicted 202322 (-3.7%)
    linear 120x24 k64 packed, ordered, A streamed by 40 rows, B in place (1 invocations, 1 packed): bitwise
      measured 197266, predicted 186855 (-5.3%)
    linear 120x24 k64 packed, ordered, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 127933, predicted 123229 (-3.7%)
    linear 120x24 k64 packed, ordered, A streamed by 28 rows, B copied (1 invocations, 1 packed): bitwise
      measured 142378, predicted 129558 (-9.0%)
      chosen: A in place, B copied; measured best: A in place, B copied; chosen/best 1.000
    linear 21x72 k56 packed, chosen (1 invocations, 1 packed): bitwise
    linear 21x72 k56 packed, A in place, B in place (1 invocations, 1 packed): bitwise
      measured 70483, predicted 70484 (+0.0%)
    linear 21x72 k56 packed, A copied, B in place (1 invocations, 1 packed): bitwise
      measured 60553, predicted 61378 (+1.4%)
    linear 21x72 k56 packed, A streamed by 21 rows, B in place (1 invocations, 1 packed): bitwise
      measured 60587, predicted 61378 (+1.3%)
    linear 21x72 k56 packed, A in place, B copied (1 invocations, 1 packed): bitwise
      measured 56528, predicted 57389 (+1.5%)
    linear 21x72 k56 packed, A streamed by 9 rows, B copied (1 invocations, 1 packed): bitwise
      measured 48174, predicted 46930 (-2.6%)
      chosen: A streamed by 9 rows, B copied; measured best: A streamed by 9 rows, B copied; chosen/best 1.000
    worst error 9.5% over 85 plans |}]
