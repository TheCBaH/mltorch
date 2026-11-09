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

let%expect_test "rates" =
  List.iter (fun (k, v) -> Fmt.pr "%-28s %10.3f@." k v) rates;
  [%expect
    {|
    attend.dwork.ram                  0.338
    attend.out.ram                   98.356
    attend.qblocks.dtcm            7188.629
    attend.qblocks.ram             3590.839
    attend.scores                    56.956
    attend.stage_q.dtcm               7.934
    attend.stage_q.ram                3.002
    class.f32_cos                   276.250
    class.f32_div                    16.386
    class.f32_exp                   181.617
    class.f32_log                   416.705
    class.f32_op                      4.740
    class.f32_sin                   271.929
    class.f32_sqrt                   23.265
    class.int_op                     14.201
    class.iteration                   0.586
    class.load_dtcm                   4.125
    class.load_ram                    5.669
    class.store_dtcm                  3.110
    class.store_ram                   7.262
    class.vec_cos                   352.968
    class.vec_exp                   191.738
    class.vec_f32_div                 9.130
    class.vec_load_dtcm               3.231
    class.vec_load_ram                6.070
    class.vec_load_strided_dtcm       6.614
    class.vec_load_strided_ram        9.470
    class.vec_log                   427.718
    class.vec_other                  13.226
    class.vec_sin                   347.468
    class.vec_sqrt                   11.924
    class.vec_store_dtcm              3.578
    class.vec_store_ram               6.999
    class.vec_store_strided_dtcm     10.337
    overlap                           0.635
    zvt.bf16.cols_k.ram               1.480
    zvt.bf16.packed.call           3125.873
    zvt.bf16.packed.cols_k.dtcm       0.009
    zvt.bf16.packed.rows_k.dtcm       0.490
    zvt.bf16.packed.rows_k.ram        0.553
    zvt.bf16.packed.tiles.ram       134.873
    zvt.bf16.rows_k.dtcm              0.939
    zvt.bf16.rows_k.ram               1.098
    zvt.bf16.tiles.ram               75.984
    zvt.f32.cols_k.dtcm               1.473
    zvt.f32.cols_k.ram                1.582
    zvt.f32.rows_k.dtcm               0.819
    zvt.f32.rows_k.ram                0.998
    zvt.f32.tiles.dtcm              101.451
    zvt.f32.tiles.ram               165.507 |}]

let%expect_test "committed predictions on fit and check sets" =
  List.iter report Cases.all;
  [%expect
    {|
    cos 16x16                          measured      23061 predicted      23047   -0.1%
    cos 16x16, scalar                  measured      72794 predicted      72732   -0.1%
    cos 32x32                          measured      92037 predicted      92187   +0.2%
    cos 32x32, scalar                  measured     290659 predicted     290907   +0.1%
    exp 16x16                          measured      12757 predicted      12716   -0.3%
    exp 16x16, scalar                  measured      48575 predicted      48506   -0.1%
    exp 32x32                          measured      50821 predicted      50866   +0.1%
    exp 32x32, scalar                  measured     193727 predicted     194003   +0.1%
    log 16x16                          measured      27845 predicted      27831   -0.1%
    log 16x16, scalar                  measured     108751 predicted     108688   -0.1%
    log 32x32                          measured     111173 predicted     111323   +0.1%
    log 32x32, scalar                  measured     434482 predicted     434733   +0.1%
    sin 16x16                          measured      22709 predicted      22695   -0.1%
    sin 16x16, scalar                  measured      71689 predicted      71625   -0.1%
    sin 32x32                          measured      90629 predicted      90779   +0.2%
    sin 32x32, scalar                  measured     286229 predicted     286483   +0.1%
    sqrt 16x16                         measured       1234 predicted       1208   -2.1%
    sqrt 16x16, scalar                 measured       8035 predicted       7967   -0.8%
    sqrt 32x32                         measured       4738 predicted       4833   +2.0%
    sqrt 32x32, scalar                 measured      31589 predicted      31851   +0.8%
    add 16x64                          measured       2628 predicted       3043  +15.8%
    add 16x64, scalar                  measured      18690 predicted      17096   -8.5%
    add 32x128                         measured      16452 predicted      18584  +13.0%
    add 32x128, scalar                 measured      82402 predicted      91695  +11.3%
    div 16x64                          measured       4972 predicted       4945   -0.5%
    div 16x64, scalar                  measured      30861 predicted      29021   -6.0%
    div 32x128                         measured      20694 predicted      20254   -2.1%
    div 32x128, scalar                 measured     131098 predicted     139397   +6.3%
    mul 16x64                          measured       2628 predicted       3043  +15.8%
    mul 16x64, scalar                  measured      18690 predicted      17096   -8.5%
    mul 32x128                         measured      16452 predicted      18584  +13.0%
    mul 32x128, scalar                 measured      82402 predicted      91695  +11.3%
    x*2 16x64                          measured       1899 predicted       2269  +19.5%
    x*2 16x64, scalar                  measured      15592 predicted      12872  -17.4%
    x*2 32x128                         measured      11340 predicted      12421   +9.5%
    x*2 32x128, scalar                 measured      61833 predicted      68475  +10.7%
    x*x+x 16x64                        measured       5253 predicted       6087  +15.9%
    x*x+x 16x64, scalar                measured      34274 predicted      29968  -12.6%
    x*x+x 32x128                       measured      30853 predicted      34261  +11.0%
    x*x+x 32x128, scalar               measured     144227 predicted     160171  +11.1%
    x+y+z 16x64                        measured       5256 predicted       6087  +15.8%
    x+y+z 16x64, scalar                measured      37380 predicted      34192   -8.5%
    x+y+z 32x128                       measured      32904 predicted      37168  +13.0%
    x+y+z 32x128, scalar               measured     164804 predicted     183391  +11.3%
    add 16x64, in RAM                  measured       4676 predicted       5373  +14.9%
    add 32x128, in RAM                 measured      18500 predicted      21491  +16.2%
    div 16x64, in RAM                  measured       5735 predicted       5790   +1.0%
    div 32x128, in RAM                 measured      22741 predicted      23161   +1.8%
    exp 32x32, in RAM                  measured      51845 predicted      50344   -2.9%
    add 16x64, scalar, in RAM          measured      22802 predicted      24510   +7.5%
    softmax over C 16x16               measured      34924 predicted      35810   +2.5%
    softmax over C 32x32               measured     135916 predicted     138143   +1.6%
    sum over C 16x64                   measured       1828 predicted       1826   -0.1%
    sum over C 32x128                  measured       7040 predicted       7060   +0.3%
    sum over W 16x64                   measured       1516 predicted        977  -35.5%
    transpose 32x32                    measured       3172 predicted       3101   -2.2%
    transpose 64x64                    measured      15492 predicted      15870   +2.4%
    transpose 32x32, in RAM            measured       4708 predicted       4708   +0.0%
    head split 16x64, 2 heads          measured     175432 predicted     157097  -10.5%
    head split 32x128, 4 heads         measured     745187 predicted     647841  -13.1%
    head split 49x64, 2 heads          measured     577606 predicted     496723  -14.0%
    linear 9x32 k64                    measured       4844 predicted       4845   +0.0%
    linear 16x64 k64                   measured      13922 predicted      13931   +0.1%
    linear 64x64 k64                   measured      57286 predicted      57352   +0.1%
    linear 16x128 k32                  measured      18302 predicted      18412   +0.6%
    linear 32x32 k128                  measured      23374 predicted      23346   -0.1%
    linear 48x80 k32                   measured      34395 predicted      34160   -0.7%
    linear bf16 9x32 k64               measured       4191 predicted       4390   +4.8%
    linear bf16 16x64 k64              measured      10403 predicted       9590   -7.8%
    linear bf16 64x64 k64              measured      28186 predicted      28678   +1.7%
    linear bf16 32x128 k32             measured      23631 predicted      24719   +4.6%
    linear bf16 48x48 k48              measured      16482 predicted      16135   -2.1%
    sdpa bf16 32x32 d32                measured      87661 predicted      80809   -7.8%
    sdpa bf16 64x64 d64                measured     373801 predicted     373674   +0.0%
    sdpa bf16 16x48 d32                measured      66169 predicted      69558   +5.1%
    sdpa bf16 16x16 d32 bk16           measured      25113 predicted      25828   +2.8%
    sdpa bf16 32x64 d64 bk16           measured     202773 predicted     204133   +0.7%
    sdpa bf16 64x32 d32                measured     175305 predicted     174208   -0.6%
    linear 9x32 k64, in RAM            measured       5853 predicted       5853   +0.0%
    linear 32x32 k64, in RAM           measured      15120 predicted      15120   +0.0%
    linear bf16 16x256 k64             measured      52692 predicted      52683   +0.0%
    linear bf16 32x128 k128            measured      95847 predicted      95829   +0.0%
    linear bf16 49x512 k64             measured     391657 predicted     391586   +0.0%
    linear bf16 16x256 k64, in RAM     measured      58806 predicted      58791   +0.0%
    linear bf16 16x64 k64, in RAM      measured      10595 predicted       9812   -7.4%
    linear bf16 64x64 k64, in RAM      measured      28314 predicted      29710   +4.9%
    sdpa bf16 32x32 d32, in RAM        measured      88109 predicted      85934   -2.5%
    sdpa bf16 16x16 d32 bk16, in RAM   measured      25273 predicted      25621   +1.4%
    sdpa bf16 64x64 d64, in RAM        measured     373801 predicted     373674   +0.0%
    sdpa bf16 9x29 d64 masked          measured      44613 predicted      44467   -0.3%
    exp 64x64                          measured     207173 predicted     202414   -2.3%
    softmax over C 64x64               measured     536591 predicted     541328   +0.9%
    softmax over W 16x16               measured      70901 predicted      66985   -5.5%
    layer norm over C 16x64            measured      22694 predicted      51051 +125.0%
    rms norm over C 16x64              measured      12611 predicted      26778 +112.3%
    linear 32x96 k48                   measured      32529 predicted      33015   +1.5%
    sdpa bf16 41x61 d32                measured     241225 predicted     218117   -9.6%
    layer, bare                        measured     243406 predicted     218397  -10.3%
    layer, all flags                   measured     282593 predicted     268811   -4.9%
    layer, pre layer norm, rope        measured     387522 predicted     456705  +17.9%
    layer, post rms                    measured     323484 predicted     338136   +4.5%
    layer decomposed, bare             measured    1971913 predicted    1408825  -28.6% |}]

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
    {| fastvit 49 x 512, 16 x 32: measured 40109063 predicted 18959284 -52.7% |}]
