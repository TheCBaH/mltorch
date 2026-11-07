(* The rates fitted to the RTL calibration data, and the estimate checked
   against cases the fit never saw. The measured cycles are read from the
   calibration suite's own golden, so this suite is hermetic: it re-plans every
   case, counts it, and compares. *)

module Cases = Coral_estimate_cases
module CE = Coral_estimate.Coral_census
module CM = Coral_estimate.Coral_model

(* name -> each invocation's measured cycles, from the calibration golden *)
let measured_in path =
  let text = In_channel.with_open_bin path In_channel.input_all in
  let lines = List.map String.trim (String.split_on_char '\n' text) in
  let rec go acc current = function
    | [] -> List.rev (match current with Some c -> c :: acc | None -> acc)
    | l :: rest -> (
        match String.index_opt l '(' with
        | Some i
          when String.length l > i + 1
               &&
                 try
                   ignore
                     (Scanf.sscanf
                        (String.sub l i (String.length l - i))
                        "(%d invocations)" Fun.id);
                   true
                 with Scanf.Scan_failure _ | Failure _ | End_of_file -> false ->
            let name = String.trim (String.sub l 0 i) in
            go
              (match current with Some c -> c :: acc | None -> acc)
              (Some (name, []))
              rest
        | _ -> (
            match
              ( current,
                try
                  Some
                    (Scanf.sscanf l "[%d] cycles %Ld instret %Ld%s@\n"
                       (fun _ c _ rest ->
                         let ram =
                           match
                             String.split_on_char ' ' (String.trim rest)
                           with
                           | [ "ram"; ids ] ->
                               List.map int_of_string
                                 (String.split_on_char ',' ids)
                           | _ -> []
                         in
                         (Int64.to_float c, ram)))
                with Scanf.Scan_failure _ | End_of_file | Failure _ -> (
                  (* another suite's [cycles N]: one run, no placement *)
                  try
                    Some
                      (Scanf.sscanf l "cycles %Ld" (fun c ->
                           (Int64.to_float c, [])))
                  with Scanf.Scan_failure _ | End_of_file | Failure _ -> None)
              )
            with
            | Some (n, cs), Some c -> go acc (Some (n, cs @ [ c ])) rest
            | _ -> go acc current rest))
  in
  go [] None lines

let measured = measured_in "../coral_estimate_rtl/calibrate_test.ml"
let cycles name = Option.map (List.map fst) (List.assoc_opt name measured)

let ram name i =
  match List.assoc_opt name measured with
  | Some l when i < List.length l -> snd (List.nth l i)
  | _ -> []

(* The route's rule: the packed function for a paired bfloat16 product whose
   panels fit the five eighths of the DTCM every function's panels share
   (taken in order; one product here at a time). [fast] refuses any other
   product with an exception it does not export. *)
let core =
  let module P = Coralnpu_kernels_zvt.Zvt_packed_contraction in
  CM.Core.vme ~packed:(fun d ->
      d.Contraction.numerics = Contraction.Numerics.Paired
      &&
      (* [panel_bytes] refuses first: [fast] raises on what it refuses *)
      match P.panel_bytes d with
      | Ok b -> b <= 32768 * 5 / 8 && P.fast d
      | Error _ -> false)

(* planning a case is the slow part, and the fit asks for its features once
   per round *)
let censuses =
  let memo = Hashtbl.create 64 in
  fun (c : Cases.t) ->
    match Hashtbl.find_opt memo c.Cases.name with
    | Some cs -> cs
    | None ->
        let cs =
          match Census_test.censuses ~ram:(ram c.Cases.name) c with
          | Ok cs -> cs
          | Error e -> Fmt.failwith "%s: %s" c.Cases.name e
        in
        Hashtbl.replace memo c.Cases.name cs;
        cs

let features ?rates c = List.map (CM.features ?rates core) (censuses c)

let samples ?rates role =
  List.concat_map
    (fun (c : Cases.t) ->
      if c.Cases.role <> role then []
      else
        match cycles c.Cases.name with
        | None -> []
        | Some ys -> List.combine (features ?rates c) ys)
    Cases.all

(* What the memory instructions cost on the VME core, read from waveforms
   with coralview (`commits --profile`, plan of commits, stalls, memory and
   tiles): lw/flw 4 cycles from the TCM, 7 from the testbench RAM; sw/fsw/sh 3
   and 8; vle32.v 3 to 5 and 6 to 8; vse32.v 4 and 8. In the cases each memory
   class rises with the others, so on their own they are not told apart; these
   rows pull each toward what the hardware does. *)
let priors =
  [
    ("class.load_dtcm", 4.);
    ("class.load_ram", 7.);
    ("class.store_dtcm", 3.);
    ("class.store_ram", 8.);
    ("class.vec_load_dtcm", 4.);
    ("class.vec_load_ram", 7.);
    ("class.vec_store_dtcm", 4.);
    ("class.vec_store_ram", 8.);
  ]

(* The overlap feature is made from rates, so the fit is a fixed point: fit
   without it, make it from those rates, fit again, until the rates stop
   moving. *)
let rates =
  let fit ?rates () =
    Coral_estimate.Coral_fit.fit ~priors ~weight:2.
      (samples ?rates Cases.Role.Fit)
  in
  let rec go n r =
    let r' = fit ~rates:r () in
    let moved =
      List.exists
        (fun (k, v) ->
          Float.abs (v -. Option.value (List.assoc_opt k r) ~default:0.)
          > 1e-6 *. Float.max 1. v)
        r'
    in
    if (not moved) || n = 0 then r' else go (n - 1) r'
  in
  go 30 (fit ())

(* The error in percent, rounded to the printed tenth. An exact fit leaves a
   residual of either sign, and "%+.1f" prints it as "-0.0" on one platform and
   "+0.0" on another; adding 0. turns a rounded -0. into +0. *)
let percent ~predicted ~measured =
  (Float.round (1000. *. (predicted -. measured) /. measured) /. 10.) +. 0.

let report (c : Cases.t) =
  match cycles c.Cases.name with
  | None -> Fmt.pr "%s: not measured@." c.Cases.name
  | Some ys ->
      let fs = features ~rates c in
      let ps = List.map (CM.Rates.predict rates) fs in
      let m = List.fold_left ( +. ) 0. ys and p = List.fold_left ( +. ) 0. ps in
      Fmt.pr "%-34s measured %10.0f predicted %10.0f %+6.1f%%@." c.Cases.name m
        p
        (percent ~predicted:p ~measured:m)

let%expect_test "rates" =
  List.iter (fun (k, v) -> Fmt.pr "%-28s %10.3f@." k v) rates;
  [%expect
    {|
    attend.dwork.ram                  0.372
    attend.out.ram                   98.456
    attend.qblocks.dtcm            6570.839
    attend.qblocks.ram             3082.995
    attend.scores                    56.941
    attend.stage_q.dtcm               9.162
    attend.stage_q.ram                4.418
    class.f32_cos                   276.879
    class.f32_div                    17.081
    class.f32_exp                   451.507
    class.f32_log                   417.333
    class.f32_op                      5.444
    class.f32_sin                   272.557
    class.f32_sqrt                   23.893
    class.int_op                     14.456
    class.load_dtcm                   4.115
    class.load_ram                    6.590
    class.store_dtcm                  3.105
    class.store_ram                   7.286
    class.vec_cos                   353.147
    class.vec_exp                   582.934
    class.vec_f32_div                 9.180
    class.vec_load_dtcm               3.207
    class.vec_load_ram                6.438
    class.vec_load_strided_dtcm       6.487
    class.vec_load_strided_ram       19.324
    class.vec_log                   427.897
    class.vec_other                  57.556
    class.vec_sin                   347.647
    class.vec_sqrt                   12.112
    class.vec_store_dtcm              3.568
    class.vec_store_ram               7.260
    overlap                           0.621
    zvt.bf16.cols_k.ram               4.140
    zvt.bf16.packed.call           3218.476
    zvt.bf16.packed.cols_k.ram        0.384
    zvt.bf16.packed.rows_k.dtcm       0.322
    zvt.bf16.packed.rows_k.ram        1.027
    zvt.bf16.packed.tiles.ram       415.253
    zvt.bf16.rows_k.dtcm              1.139
    zvt.bf16.rows_k.ram               4.156
    zvt.bf16.tiles.ram              269.836
    zvt.f32.cols_k.dtcm               0.989
    zvt.f32.cols_k.ram                4.009
    zvt.f32.rows_k.dtcm               1.247
    zvt.f32.rows_k.ram                4.268
    zvt.f32.tiles.dtcm              267.387
    zvt.f32.tiles.ram               350.590 |}]

let%expect_test "fit and check" =
  List.iter report Cases.all;
  [%expect
    {|
    cos 16x16                          measured      23061 predicted      23035   -0.1%
    cos 16x16, scalar                  measured      72794 predicted      72729   -0.1%
    cos 32x32                          measured      92037 predicted      92140   +0.1%
    cos 32x32, scalar                  measured     290659 predicted     290917   +0.1%
    exp 16x16                          measured      37685 predicted      37741   +0.1%
    exp 16x16, scalar                  measured     118054 predicted     117434   -0.5%
    exp 32x32                          measured     150533 predicted     150966   +0.3%
    exp 32x32, scalar                  measured     471670 predicted     469736   -0.4%
    log 16x16                          measured      27845 predicted      27819   -0.1%
    log 16x16, scalar                  measured     108751 predicted     108686   -0.1%
    log 32x32                          measured     111173 predicted     111276   +0.1%
    log 32x32, scalar                  measured     434482 predicted     434743   +0.1%
    sin 16x16                          measured      22709 predicted      22683   -0.1%
    sin 16x16, scalar                  measured      71689 predicted      71623   -0.1%
    sin 32x32                          measured      90629 predicted      90732   +0.1%
    sin 32x32, scalar                  measured     286229 predicted     286492   +0.1%
    sqrt 16x16                         measured       1235 predicted       1209   -2.1%
    sqrt 16x16, scalar                 measured       8035 predicted       7965   -0.9%
    sqrt 32x32                         measured       4739 predicted       4835   +2.0%
    sqrt 32x32, scalar                 measured      31589 predicted      31860   +0.9%
    add 16x64                          measured       2629 predicted       3018  +14.8%
    add 16x64, scalar                  measured      18690 predicted      17182   -8.1%
    add 32x128                         measured      17477 predicted      19161   +9.6%
    add 32x128, scalar                 measured      86498 predicted      95991  +11.0%
    div 16x64                          measured       4908 predicted       4906   +0.0%
    div 16x64, scalar                  measured      30861 predicted      29098   -5.7%
    div 32x128                         measured      20950 predicted      20870   -0.4%
    div 32x128, scalar                 measured     135194 predicted     143654   +6.3%
    mul 16x64                          measured       2629 predicted       3018  +14.8%
    mul 16x64, scalar                  measured      18690 predicted      17182   -8.1%
    mul 32x128                         measured      17477 predicted      19161   +9.6%
    mul 32x128, scalar                 measured      86498 predicted      95991  +11.0%
    x*2 16x64                          measured       1899 predicted       2427  +27.8%
    x*2 16x64, scalar                  measured      15592 predicted      12968  -16.8%
    x*2 32x128                         measured      11340 predicted      12799  +12.9%
    x*2 32x128, scalar                 measured      61833 predicted      68998  +11.6%
    x*x+x 16x64                        measured       5255 predicted       6036  +14.9%
    x*x+x 16x64, scalar                measured      34274 predicted      30151  -12.0%
    x*x+x 32x128                       measured      31879 predicted      35014   +9.8%
    x*x+x 32x128, scalar               measured     148323 predicted     164990  +11.2%
    x+y+z 16x64                        measured       5258 predicted       6036  +14.8%
    x+y+z 16x64, scalar                measured      37380 predicted      34364   -8.1%
    x+y+z 32x128                       measured      34954 predicted      38321   +9.6%
    x+y+z 32x128, scalar               measured     172996 predicted     191983  +11.0%
    add 16x64, in RAM                  measured       5188 predicted       5617   +8.3%
    add 32x128, in RAM                 measured      20548 predicted      22468   +9.3%
    div 16x64, in RAM                  measured       6055 predicted       6044   -0.2%
    div 32x128, in RAM                 measured      24021 predicted      24178   +0.7%
    exp 32x32, in RAM                  measured     151683 predicted     150559   -0.7%
    add 16x64, scalar, in RAM          measured      24850 predicted      26532   +6.8%
    softmax over C 16x16               measured     163662 predicted     166176   +1.5%
    softmax over C 32x32               measured     651521 predicted     649971   -0.2%
    sum over C 16x64                   measured       1826 predicted       1891   +3.5%
    sum over C 32x128                  measured       7036 predicted       6872   -2.3%
    sum over W 16x64                   measured       1514 predicted       1108  -26.8%
    transpose 32x32                    measured       3172 predicted       3037   -4.3%
    transpose 64x64                    measured      15492 predicted      15926   +2.8%
    transpose 32x32, in RAM            measured       7268 predicted       7268   +0.0%
    head split 16x64, 2 heads          measured     175464 predicted     159046   -9.4%
    head split 32x128, 4 heads         measured     745315 predicted     656035  -12.0%
    head split 49x64, 2 heads          measured     577704 predicted     502995  -12.9%
    linear 9x32 k64                    measured       4844 predicted       4844   +0.0%
    linear 16x64 k64                   measured      14754 predicted      14758   +0.0%
    linear 64x64 k64                   measured     107721 predicted     107832   +0.1%
    linear 16x128 k32                  measured      19966 predicted      20130   +0.8%
    linear 32x32 k128                  measured      47758 predicted      47714   -0.1%
    linear 48x80 k32                   measured      37517 predicted      37195   -0.9%
    linear bf16 9x32 k64               measured       4192 predicted       4437   +5.9%
    linear bf16 16x64 k64              measured      11236 predicted      10397   -7.5%
    linear bf16 64x64 k64              measured      37530 predicted      37530   +0.0%
    linear bf16 32x128 k32             measured      26959 predicted      28605   +6.1%
    linear bf16 48x48 k48              measured      18355 predicted      17994   -2.0%
    sdpa bf16 32x32 d32                measured      87669 predicted      80831   -7.8%
    sdpa bf16 64x64 d64                measured     386501 predicted     386501   +0.0%
    sdpa bf16 16x48 d32                measured      66170 predicted      69570   +5.1%
    sdpa bf16 16x16 d32 bk16           measured      25119 predicted      25839   +2.9%
    sdpa bf16 32x64 d64 bk16           measured     208447 predicted     209940   +0.7%
    sdpa bf16 64x32 d32                measured     175321 predicted     174265   -0.6%
    linear 9x32 k64, in RAM            measured      15175 predicted      15175   +0.0%
    linear 32x32 k64, in RAM           measured      39504 predicted      39504   +0.0%
    linear bf16 16x256 k64             measured     103124 predicted     103124   +0.0%
    linear bf16 32x128 k128            measured     193384 predicted     193384   +0.0%
    linear bf16 49x512 k64             measured     788876 predicted     788876   +0.0%
    linear bf16 16x256 k64, in RAM     measured     156342 predicted     156342   +0.0%
    linear bf16 16x64 k64, in RAM      measured      18916 predicted      14857  -21.5%
    linear bf16 64x64 k64, in RAM      measured      43674 predicted      49083  +12.4%
    sdpa bf16 32x32 d32, in RAM        measured      89717 predicted      87504   -2.5%
    sdpa bf16 16x16 d32 bk16, in RAM   measured      25759 predicted      26124   +1.4%
    sdpa bf16 64x64 d64, in RAM        measured     386501 predicted     386501   +0.0%
    sdpa bf16 9x29 d64 masked          measured      45247 predicted      45106   -0.3%
    exp 64x64                          measured     606021 predicted     603024   -0.5%
    softmax over C 64x64               measured    2600899 predicted    2569576   -1.2%
    softmax over W 16x16               measured     163763 predicted     166158   +1.5%
    layer norm over C 16x64            measured      40191 predicted      37186   -7.5%
    rms norm over C 16x64              measured      20229 predicted      17507  -13.5%
    linear 32x96 k48                   measured      61522 predicted      61950   +0.7%
    sdpa bf16 41x61 d32                measured     241222 predicted     218160   -9.6%
    layer, bare                        measured     259919 predicted     234577   -9.8%
    layer, all flags                   measured     299379 predicted     296467   -1.0%
    layer, pre layer norm, rope        measured     397265 predicted     497248  +25.2%
    layer, post rms                    measured     343311 predicted     369216   +7.5%
    layer decomposed, bare             measured    2216770 predicted    1722892  -22.3% |}]

(* The committed rates are this fit: a change of the calibration data shows
   here first. *)
let%expect_test "the committed rates are the fit" =
  let committed = Coral_estimate.Coral_rates.vme in
  let names =
    List.sort_uniq compare (List.map fst committed @ List.map fst rates)
  in
  let get l k = Option.value (List.assoc_opt k l) ~default:0. in
  let off =
    List.filter
      (fun k ->
        let a = get committed k and b = get rates k in
        Float.abs (a -. b) > 1e-3 *. Float.max 1. (Float.abs b))
      names
  in
  if off = [] then print_endline "committed rates match the fit"
  else
    List.iter
      (fun k ->
        Fmt.pr "%s: committed %.3f, fit %.3f@." k (get committed k)
          (get rates k))
      off;
  [%expect {|
    class.iteration: committed 0.948, fit 0.000
    class.store_dtcm: committed 3.100, fit 3.105
    zvt.f32.tiles.ram: committed 378.847, fit 350.590 |}]

(* Shapes no calibration case has, predicted from the committed rates: the
   first record of the case in another suite's golden (the VME core's run). *)
let%expect_test "predicted where measured elsewhere" =
  List.iter
    (fun ((c : Cases.t), path) ->
      match List.assoc_opt c.Cases.name (measured_in path) with
      | None -> Fmt.pr "%s: not in %s@." c.Cases.name path
      | Some runs ->
          let m = List.fold_left (fun a (y, _) -> a +. y) 0. runs in
          let p =
            List.fold_left ( +. ) 0.
              (List.map
                 (CM.Rates.predict Coral_estimate.Coral_rates.vme)
                 (features ~rates:Coral_estimate.Coral_rates.vme c))
          in
          Fmt.pr "%s: measured %.0f predicted %.0f %+.1f%%@." c.Cases.name m p
            (percent ~predicted:p ~measured:m))
    Cases.elsewhere;
  [%expect
    {| fastvit 49 x 512, 16 x 32: measured 40109063 predicted 40447653 +0.8% |}]
