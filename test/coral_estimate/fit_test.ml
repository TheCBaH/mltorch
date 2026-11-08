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
    attend.dwork.ram                  0.372
    attend.out.ram                   98.456
    attend.qblocks.dtcm            6570.839
    attend.qblocks.ram             3082.995
    attend.scores                    56.941
    attend.stage_q.dtcm               9.162
    attend.stage_q.ram                4.418
    class.f32_cos                   275.750
    class.f32_div                    15.986
    class.f32_exp                   181.117
    class.f32_log                   416.205
    class.f32_op                      4.339
    class.f32_sin                   271.429
    class.f32_sqrt                   22.764
    class.int_op                     14.288
    class.iteration                   1.076
    class.load_dtcm                   4.117
    class.load_ram                    6.592
    class.store_dtcm                  3.105
    class.store_ram                   7.291
    class.vec_cos                   352.871
    class.vec_exp                   191.978
    class.vec_f32_div                 8.804
    class.vec_load_dtcm               3.214
    class.vec_load_ram                6.421
    class.vec_load_strided_dtcm       6.436
    class.vec_load_strided_ram       19.128
    class.vec_log                   427.622
    class.vec_other                  13.111
    class.vec_sin                   347.371
    class.vec_sqrt                   11.837
    class.vec_store_dtcm              3.568
    class.vec_store_ram               7.207
    class.vec_store_strided_dtcm     14.120
    overlap                           0.623
    zvt.bf16.cols_k.ram               4.121
    zvt.bf16.packed.call           3390.504
    zvt.bf16.packed.cols_k.ram        0.387
    zvt.bf16.packed.rows_k.dtcm       0.312
    zvt.bf16.packed.rows_k.ram        1.017
    zvt.bf16.packed.tiles.ram       414.910
    zvt.bf16.rows_k.dtcm              1.160
    zvt.bf16.rows_k.ram               4.180
    zvt.bf16.tiles.ram              271.369
    zvt.f32.cols_k.dtcm               1.179
    zvt.f32.cols_k.ram                4.168
    zvt.f32.rows_k.dtcm               1.109
    zvt.f32.rows_k.ram                4.157
    zvt.f32.tiles.dtcm              231.173
    zvt.f32.tiles.ram               333.709 |}]

let%expect_test "committed predictions on fit and check sets" =
  List.iter report Cases.all;
  [%expect
    {|
    cos 16x16                          measured      23061 predicted      23053   +0.0%
    cos 16x16, scalar                  measured      72794 predicted      72733   -0.1%
    cos 32x32                          measured      92037 predicted      92211   +0.2%
    cos 32x32, scalar                  measured     290659 predicted     290898   +0.1%
    exp 16x16                          measured      12757 predicted      12738   -0.2%
    exp 16x16, scalar                  measured      48575 predicted      48507   -0.1%
    exp 32x32                          measured      50821 predicted      50951   +0.3%
    exp 32x32, scalar                  measured     193727 predicted     193993   +0.1%
    log 16x16                          measured      27845 predicted      27837   +0.0%
    log 16x16, scalar                  measured     108751 predicted     108689   -0.1%
    log 32x32                          measured     111173 predicted     111347   +0.2%
    log 32x32, scalar                  measured     434482 predicted     434723   +0.1%
    sin 16x16                          measured      22709 predicted      22701   +0.0%
    sin 16x16, scalar                  measured      71689 predicted      71627   -0.1%
    sin 32x32                          measured      90629 predicted      90803   +0.2%
    sin 32x32, scalar                  measured     286229 predicted     286473   +0.1%
    sqrt 16x16                         measured       1235 predicted       1209   -2.1%
    sqrt 16x16, scalar                 measured       8035 predicted       7969   -0.8%
    sqrt 32x32                         measured       4739 predicted       4835   +2.0%
    sqrt 32x32, scalar                 measured      31589 predicted      31840   +0.8%
    add 16x64                          measured       2629 predicted       3051  +16.0%
    add 16x64, scalar                  measured      18690 predicted      17169   -8.1%
    add 32x128                         measured      17477 predicted      19213   +9.9%
    add 32x128, scalar                 measured      86498 predicted      95901  +10.9%
    div 16x64                          measured       4908 predicted       4882   -0.5%
    div 16x64, scalar                  measured      30861 predicted      29096   -5.7%
    div 32x128                         measured      20950 predicted      20920   -0.1%
    div 32x128, scalar                 measured     135194 predicted     143607   +6.2%
    mul 16x64                          measured       2629 predicted       3051  +16.0%
    mul 16x64, scalar                  measured      18690 predicted      17169   -8.1%
    mul 32x128                         measured      17477 predicted      19213   +9.9%
    mul 32x128, scalar                 measured      86498 predicted      95901  +10.9%
    x*2 16x64                          measured       1899 predicted       2280  +20.1%
    x*2 16x64, scalar                  measured      15592 predicted      12955  -16.9%
    x*2 32x128                         measured      11340 predicted      12690  +11.9%
    x*2 32x128, scalar                 measured      61833 predicted      68909  +11.4%
    x*x+x 16x64                        measured       5255 predicted       6101  +16.1%
    x*x+x 16x64, scalar                measured      34274 predicted      30125  -12.1%
    x*x+x 32x128                       measured      31879 predicted      35142  +10.2%
    x*x+x 32x128, scalar               measured     148323 predicted     164810  +11.1%
    x+y+z 16x64                        measured       5258 predicted       6101  +16.0%
    x+y+z 16x64, scalar                measured      37380 predicted      34338   -8.1%
    x+y+z 32x128                       measured      34954 predicted      38426   +9.9%
    x+y+z 32x128, scalar               measured     172996 predicted     191803  +10.9%
    add 16x64, in RAM                  measured       5188 predicted       5624   +8.4%
    add 32x128, in RAM                 measured      20548 predicted      22497   +9.5%
    div 16x64, in RAM                  measured       6055 predicted       6051   -0.1%
    div 32x128, in RAM                 measured      24021 predicted      24204   +0.8%
    exp 32x32, in RAM                  measured      51971 predicted      50530   -2.8%
    add 16x64, scalar, in RAM          measured      24850 predicted      26518   +6.7%
    softmax over C 16x16               measured      34940 predicted      35709   +2.2%
    softmax over C 32x32               measured     135948 predicted     137769   +1.3%
    sum over C 16x64                   measured       1826 predicted       1826   +0.0%
    sum over C 32x128                  measured       7036 predicted       7033   +0.0%
    sum over W 16x64                   measured       1514 predicted       1005  -33.6%
    transpose 32x32                    measured       3172 predicted       3087   -2.7%
    transpose 64x64                    measured      15492 predicted      16006   +3.3%
    transpose 32x32, in RAM            measured       7268 predicted       7268   +0.0%
    head split 16x64, 2 heads          measured     175464 predicted     158535   -9.6%
    head split 32x128, 4 heads         measured     745315 predicted     653905  -12.3%
    head split 49x64, 2 heads          measured     577704 predicted     501381  -13.2%
    linear 9x32 k64                    measured       4844 predicted       4843   +0.0%
    linear 16x64 k64                   measured      14754 predicted      14757   +0.0%
    linear 64x64 k64                   measured     107721 predicted     107841   +0.1%
    linear 16x128 k32                  measured      19966 predicted      20072   +0.5%
    linear 32x32 k128                  measured      47758 predicted      47700   -0.1%
    linear 48x80 k32                   measured      37517 predicted      37291   -0.6%
    linear bf16 9x32 k64               measured       4192 predicted       4439   +5.9%
    linear bf16 16x64 k64              measured      11236 predicted      10379   -7.6%
    linear bf16 64x64 k64              measured      37530 predicted      37529   +0.0%
    linear bf16 32x128 k32             measured      26959 predicted      28599   +6.1%
    linear bf16 48x48 k48              measured      18355 predicted      18027   -1.8%
    sdpa bf16 32x32 d32                measured      87669 predicted      80831   -7.8%
    sdpa bf16 64x64 d64                measured     386501 predicted     386381   +0.0%
    sdpa bf16 16x48 d32                measured      66170 predicted      69569   +5.1%
    sdpa bf16 16x16 d32 bk16           measured      25119 predicted      25839   +2.9%
    sdpa bf16 32x64 d64 bk16           measured     208447 predicted     209882   +0.7%
    sdpa bf16 64x32 d32                measured     175321 predicted     174265   -0.6%
    linear 9x32 k64, in RAM            measured      15175 predicted      15174   +0.0%
    linear 32x32 k64, in RAM           measured      39504 predicted      39501   +0.0%
    linear bf16 16x256 k64             measured     103124 predicted     103131   +0.0%
    linear bf16 32x128 k128            measured     193384 predicted     193398   +0.0%
    linear bf16 49x512 k64             measured     788876 predicted     788917   +0.0%
    linear bf16 16x256 k64, in RAM     measured     156342 predicted     156337   +0.0%
    linear bf16 16x64 k64, in RAM      measured      18916 predicted      14852  -21.5%
    linear bf16 64x64 k64, in RAM      measured      43674 predicted      49080  +12.4%
    sdpa bf16 32x32 d32, in RAM        measured      89717 predicted      87488   -2.5%
    sdpa bf16 16x16 d32 bk16, in RAM   measured      25759 predicted      26120   +1.4%
    sdpa bf16 64x64 d64, in RAM        measured     386501 predicted     386381   +0.0%
    sdpa bf16 9x29 d64 masked          measured      45247 predicted      45107   -0.3%
    exp 64x64                          measured     207173 predicted     202934   -2.0%
    softmax over C 64x64               measured     536655 predicted     540065   +0.6%
    softmax over W 16x16               measured      70886 predicted      67124   -5.3%
    layer norm over C 16x64            measured      22678 predicted      50027 +120.6%
    rms norm over C 16x64              measured      12627 predicted      26264 +108.0%
    linear 32x96 k48                   measured      61522 predicted      61871   +0.6%
    sdpa bf16 41x61 d32                measured     241222 predicted     218160   -9.6%
    layer, bare                        measured     277342 predicted     237790  -14.3%
    layer, all flags                   measured     320945 predicted     294018   -8.4%
    layer, pre layer norm, rope        measured     429970 predicted     482007  +12.1%
    layer, post rms                    measured     363868 predicted     365174   +0.4%
    layer decomposed, bare             measured    1975370 predicted    1406829  -28.8% |}]

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
    {| fastvit 49 x 512, 16 x 32: measured 40109063 predicted 40358251 +0.6% |}]
