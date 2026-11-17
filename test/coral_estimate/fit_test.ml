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
      | Ok b -> (b :> int) <= 32768 * 5 / 8 && P.fast d
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

let view_features ?rates (v : Cases.View.t) =
  [ CM.features ?rates core (Census_test.view_census v) ]

let samples ?rates role =
  List.concat_map
    (fun (c : Cases.t) ->
      if c.Cases.role <> role then []
      else
        match cycles c.Cases.name with
        | None -> []
        | Some ys -> List.combine (features ?rates c) ys)
    Cases.all
  @ List.concat_map
      (fun (v : Cases.View.t) ->
        if v.Cases.View.role <> role then []
        else
          match cycles v.Cases.View.name with
          | None -> []
          | Some ys -> List.combine (view_features ?rates v) ys)
      Cases.View.all

(* What the memory instructions cost on the VME core, read from waveforms
   with coralview (`commits --profile`, plan of commits, stalls, memory and
   tiles): lw/flw 4 cycles from the TCM, 6 from the testbench RAM (7 before the
   release that put the LSU on TileLink-UL); sw/fsw/sh 3 and 8; vle32.v 3 to 5 and 6 to 8; vse32.v 4 and 8. In the cases each memory
   class rises with the others, so on their own they are not told apart; these
   rows pull each toward what the hardware does. *)
let priors =
  [
    ("class.load_dtcm", 4.);
    ("class.load_ram", 6.);
    ("class.store_dtcm", 3.);
    ("class.store_ram", 8.);
    ("class.vec_load_dtcm", 4.);
    ("class.vec_load_ram", 7.);
    ("class.vec_store_dtcm", 4.);
    ("class.vec_store_strided_dtcm", 7.);
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
      let committed = Coral_estimate.Coral_rates.vme in
      let fs = features ~rates:committed c in
      let ps = List.map (CM.Rates.predict committed) fs in
      let m = List.fold_left ( +. ) 0. ys and p = List.fold_left ( +. ) 0. ps in
      Fmt.pr "%-34s measured %10.0f predicted %10.0f %+6.1f%%@." c.Cases.name m
        p
        (percent ~predicted:p ~measured:m)

let report_view (v : Cases.View.t) =
  match cycles v.Cases.View.name with
  | None -> Fmt.pr "%s: not measured@." v.Cases.View.name
  | Some ys ->
      let committed = Coral_estimate.Coral_rates.vme in
      let ps =
        List.map (CM.Rates.predict committed) (view_features ~rates:committed v)
      in
      let m = List.fold_left ( +. ) 0. ys and p = List.fold_left ( +. ) 0. ps in
      Fmt.pr "%-34s measured %10.0f predicted %10.0f %+6.1f%%@."
        v.Cases.View.name m p
        (percent ~predicted:p ~measured:m)

let%expect_test "rates" =
  List.iter (fun (k, v) -> Fmt.pr "%-28s %10.3f@." k v) rates;
  [%expect
    {|
    attend.dwork.ram                  0.389
    attend.kblocks.ram              290.797
    attend.out.ram                   65.100
    attend.qblocks.dtcm           11408.419
    attend.qblocks.ram             4809.113
    attend.scores                    57.510
    class.bf16_round                 17.016
    class.check_index                21.516
    class.f32_cos                   273.381
    class.f32_div                    13.637
    class.f32_exp                   178.747
    class.f32_log                   413.835
    class.f32_op                      1.997
    class.f32_sin                   269.059
    class.f32_sqrt                   20.396
    class.int_div                    22.244
    class.iteration                   3.333
    class.load_dtcm                   4.120
    class.load_ram                    5.656
    class.store_dtcm                  3.109
    class.store_ram                   7.227
    class.vec_cos                   351.730
    class.vec_exp                   192.589
    class.vec_f32_div                 7.578
    class.vec_load_dtcm               3.542
    class.vec_load_ram                6.269
    class.vec_load_strided_dtcm       6.271
    class.vec_load_strided_ram       10.103
    class.vec_log                   426.480
    class.vec_other                  13.198
    class.vec_sin                   346.230
    class.vec_sqrt                   10.685
    class.vec_store_dtcm              3.818
    class.vec_store_ram               7.454
    class.vec_store_strided_dtcm      6.998
    overlap                           0.768
    zvt.bf16.cols_k.ram               1.365
    zvt.bf16.packed.call           2662.899
    zvt.bf16.packed.cols_k.dtcm       0.126
    zvt.bf16.packed.rows_k.dtcm       0.684
    zvt.bf16.packed.rows_k.ram        0.776
    zvt.bf16.rows_k.dtcm              1.038
    zvt.bf16.rows_k.ram               1.184
    zvt.bf16.tiles.ram              153.267
    zvt.f32.call                     20.103
    zvt.f32.cols_k.dtcm               1.431
    zvt.f32.cols_k.ram                1.542
    zvt.f32.rows_k.dtcm               0.868
    zvt.f32.rows_k.ram                1.036
    zvt.f32.tiles.dtcm              125.102
    zvt.f32.tiles.ram               189.896
    zvt.f32.view.cols_k.dtcm          3.045
    zvt.f32.view.rows_k.dtcm          0.201
    zvt.f32.view.tiles.dtcm         106.645 |}]

let%expect_test "committed predictions on fit and check sets" =
  List.iter report Cases.all;
  List.iter report_view Cases.View.all;
  [%expect
    {|
    cos 16x16                          measured      23061 predicted      23035   -0.1%
    cos 16x16, scalar                  measured      72794 predicted      72743   -0.1%
    cos 32x32                          measured      92037 predicted      92140   +0.1%
    cos 32x32, scalar                  measured     290659 predicted     290864   +0.1%
    exp 16x16                          measured      12757 predicted      12850   +0.7%
    exp 16x16, scalar                  measured      48575 predicted      48516   -0.1%
    exp 32x32                          measured      50821 predicted      51400   +1.1%
    exp 32x32, scalar                  measured     193727 predicted     193959   +0.1%
    log 16x16                          measured      27845 predicted      27819   -0.1%
    log 16x16, scalar                  measured     108751 predicted     108699   +0.0%
    log 32x32                          measured     111173 predicted     111276   +0.1%
    log 32x32, scalar                  measured     434482 predicted     434689   +0.0%
    sin 16x16                          measured      22709 predicted      22683   -0.1%
    sin 16x16, scalar                  measured      71689 predicted      71636   -0.1%
    sin 32x32                          measured      90629 predicted      90732   +0.1%
    sin 32x32, scalar                  measured     286229 predicted     286439   +0.1%
    sqrt 16x16                         measured       1234 predicted       1208   -2.1%
    sqrt 16x16, scalar                 measured       8035 predicted       7979   -0.7%
    sqrt 32x32                         measured       4738 predicted       4833   +2.0%
    sqrt 32x32, scalar                 measured      31589 predicted      31808   +0.7%
    add 16x64                          measured       2628 predicted       2898  +10.3%
    add 16x64, scalar                  measured      18690 predicted      17133   -8.3%
    add 32x128                         measured      16452 predicted      18106  +10.1%
    add 32x128, scalar                 measured      82402 predicted      91583  +11.1%
    div 16x64                          measured       4972 predicted       4944   -0.6%
    div 16x64, scalar                  measured      30861 predicted      29052   -5.9%
    div 32x128                         measured      20694 predicted      20333   -1.7%
    div 32x128, scalar                 measured     131098 predicted     139260   +6.2%
    mul 16x64                          measured       2628 predicted       2898  +10.3%
    mul 16x64, scalar                  measured      18690 predicted      17133   -8.3%
    mul 32x128                         measured      16452 predicted      18106  +10.1%
    mul 32x128, scalar                 measured      82402 predicted      91583  +11.1%
    x*2 16x64                          measured       1899 predicted       2044   +7.6%
    x*2 16x64, scalar                  measured      15592 predicted      12914  -17.2%
    x*2 32x128                         measured      11340 predicted      11739   +3.5%
    x*2 32x128, scalar                 measured      61833 predicted      68416  +10.6%
    x*x+x 16x64                        measured       5253 predicted       5795  +10.3%
    x*x+x 16x64, scalar                measured      34274 predicted      30046  -12.3%
    x*x+x 32x128                       measured      30853 predicted      33420   +8.3%
    x*x+x 32x128, scalar               measured     144227 predicted     159998  +10.9%
    x+y+z 16x64                        measured       5256 predicted       5795  +10.3%
    x+y+z 16x64, scalar                measured      37380 predicted      34265   -8.3%
    x+y+z 32x128                       measured      32904 predicted      36212  +10.1%
    x+y+z 32x128, scalar               measured     164804 predicted     183165  +11.1%
    add 16x64, in RAM                  measured       4676 predicted       5225  +11.7%
    add 32x128, in RAM                 measured      18500 predicted      20898  +13.0%
    div 16x64, in RAM                  measured       5735 predicted       5781   +0.8%
    div 32x128, in RAM                 measured      22741 predicted      23125   +1.7%
    exp 32x32, in RAM                  measured      51845 predicted      50331   -2.9%
    add 16x64, scalar, in RAM          measured      22802 predicted      24495   +7.4%
    softmax over C 16x16               measured      34255 predicted      34837   +1.7%
    softmax over C 32x32               measured     134651 predicted     134174   -0.4%
    sum over C 16x64                   measured       1828 predicted       1903   +4.1%
    sum over C 32x128                  measured       7040 predicted       7390   +5.0%
    sum over W 16x64                   measured       1516 predicted       1247  -17.7%
    transpose 32x32                    measured       3172 predicted       2796  -11.9%
    transpose 64x64                    measured      15492 predicted      14694   -5.1%
    transpose 32x32, in RAM            measured       4708 predicted       4708   +0.0%
    head split 16x64, 2 heads          measured      80119 predicted      81413   +1.6%
    head split 32x128, 4 heads         measured     347645 predicted     346123   -0.4%
    head split 49x64, 2 heads          measured     271936 predicted     265079   -2.5%
    linear 9x32 k64                    measured       4844 predicted       4844   +0.0%
    linear 16x64 k64                   measured      13922 predicted      13922   +0.0%
    linear 64x64 k64                   measured      57286 predicted      57289   +0.0%
    linear 16x128 k32                  measured      18302 predicted      18302   +0.0%
    linear 32x32 k128                  measured      23374 predicted      23371   +0.0%
    linear 48x80 k32                   measured      34395 predicted      34392   +0.0%
    linear bf16 9x32 k64               measured       4191 predicted       4352   +3.8%
    linear bf16 16x64 k64              measured      10403 predicted       9707   -6.7%
    linear bf16 64x64 k64              measured      28186 predicted      28615   +1.5%
    linear bf16 32x128 k32             measured      23631 predicted      23937   +1.3%
    linear bf16 48x48 k48              measured      16482 predicted      16699   +1.3%
    sdpa bf16 32x32 d32                measured      87661 predicted      81707   -6.8%
    sdpa bf16 64x64 d64                measured     373801 predicted     375763   +0.5%
    sdpa bf16 16x48 d32                measured      66169 predicted      70299   +6.2%
    sdpa bf16 16x16 d32 bk16           measured      25113 predicted      26131   +4.1%
    sdpa bf16 32x64 d64 bk16           measured     202773 predicted     202244   -0.3%
    sdpa bf16 64x32 d32                measured     175305 predicted     171747   -2.0%
    linear 9x32 k64, in RAM            measured       5853 predicted       5852   +0.0%
    linear 32x32 k64, in RAM           measured      15120 predicted      15118   +0.0%
    linear bf16 16x256 k64             measured      52692 predicted      52685   +0.0%
    linear bf16 32x128 k128            measured      95847 predicted      95832   +0.0%
    linear bf16 49x512 k64             measured     391657 predicted     391597   +0.0%
    linear bf16 16x256 k64, in RAM     measured      58806 predicted      58800   +0.0%
    linear bf16 16x64 k64, in RAM      measured      10595 predicted       9567   -9.7%
    linear bf16 64x64 k64, in RAM      measured      28314 predicted      30122   +6.4%
    sdpa bf16 32x32 d32, in RAM        measured      88109 predicted      86003   -2.4%
    sdpa bf16 16x16 d32 bk16, in RAM   measured      25273 predicted      25092   -0.7%
    sdpa bf16 64x64 d64, in RAM        measured     373801 predicted     375763   +0.5%
    sdpa bf16 9x29 d64 masked          measured      44613 predicted      45020   +0.9%
    layer, bare 16x32                  measured     117502 predicted     127461   +8.5%
    layer, bare 32x64                  measured     560662 predicted     503940  -10.1%
    exp 64x64                          measured     207173 predicted     203462   -1.8%
    softmax over C 64x64               measured     534046 predicted     524209   -1.8%
    softmax over W 16x16               measured      70214 predicted      66054   -5.9%
    layer norm over C 16x64            measured      21422 predicted      42622  +99.0%
    rms norm over C 16x64              measured      12145 predicted      22177  +82.6%
    linear 32x96 k48                   measured      32529 predicted      32736   +0.6%
    sdpa bf16 41x61 d32                measured     241225 predicted     217146  -10.0%
    layer, bare                        measured     243406 predicted     253034   +4.0%
    layer, all flags                   measured     282593 predicted     289701   +2.5%
    layer, pre layer norm, rope        measured     387522 predicted     380833   -1.7%
    layer, post rms                    measured     323484 predicted     326712   +1.0%
    layer decomposed, bare             measured    1392931 predicted    1002631  -28.0%
    view pointwise 3 stages, 16 rows, block 2, looped measured       3979 predicted       3995   +0.4%
    view pointwise 3 stages, 12 rows, block 1, looped measured       3773 predicted       4019   +6.5%
    view pointwise 3 stages, 32 rows, block 1, looped measured      11803 predicted      10948   -7.2%
    view pointwise 2 stages, 32 rows, block 4, looped measured      10352 predicted      11187   +8.1%
    view pointwise 2 stages, 16 rows, block 2, looped measured       8568 predicted       7863   -8.2%
    view pointwise 1 stage, 24 rows, block 2, looped measured       6564 predicted       6610   +0.7%
    view window 1 stage, 3x3, 16 rows, block 2, looped measured      15972 predicted      14684   -8.1%
    view window 1 stage, 3x3, 24 rows, block 1, looped measured      19500 predicted      20505   +5.2%
    view window 1 stage, 3x3, 20 rows, block 4, looped measured      30876 predicted      27033  -12.4%
    view window 1 stage, 3x3, 16 rows, block 2, wide measured      16546 predicted      18108   +9.4%
    view pointwise 3 stages, 20 rows, block 2, wide measured       8147 predicted       7694   -5.6%
    view window 1 stage, 3x3, 18 rows, block 2, wide measured      27744 predicted      26478   -4.6% |}]

(* The committed rates are this fit: a change of the calibration data shows
   here first. *)
let%test_unit "the committed rates are the fit" =
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
  if off <> [] then
    Fmt.failwith "committed rates differ from the fit: %a"
      Fmt.(list ~sep:(any "; ") string)
      (List.map
         (fun k ->
           Fmt.str "%s: committed %.3f, fit %.3f" k (get committed k)
             (get rates k))
         off)

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
    {| fastvit 49 x 512, 16 x 32: measured 40109063 predicted 19691015 -50.9% |}]
