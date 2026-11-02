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

let features c =
  match Census_test.censuses ~ram:(ram c.Cases.name) c with
  | Ok cs -> List.map (CM.features core) cs
  | Error e -> Fmt.failwith "%s: %s" c.Cases.name e

let samples role =
  List.concat_map
    (fun (c : Cases.t) ->
      if c.Cases.role <> role then []
      else
        match cycles c.Cases.name with
        | None -> []
        | Some ys -> List.combine (features c) ys)
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

let rates =
  Coral_estimate.Coral_fit.fit ~priors ~weight:2. (samples Cases.Role.Fit)

let report (c : Cases.t) =
  match cycles c.Cases.name with
  | None -> Fmt.pr "%s: not measured@." c.Cases.name
  | Some ys ->
      let fs = features c in
      let ps = List.map (CM.Rates.predict rates) fs in
      let m = List.fold_left ( +. ) 0. ys and p = List.fold_left ( +. ) 0. ps in
      Fmt.pr "%-34s measured %10.0f predicted %10.0f %+6.1f%%@." c.Cases.name m
        p
        (100. *. (p -. m) /. m)

let%expect_test "rates" =
  List.iter (fun (k, v) -> Fmt.pr "%-28s %10.3f@." k v) rates;
  [%expect
    {|
    attend.dwork.ram                  0.370
    attend.out.ram                  102.516
    attend.qblocks.dtcm            6342.991
    attend.qblocks.ram             2643.202
    attend.scores                    57.084
    attend.stage_q.dtcm               9.038
    attend.stage_q.ram                4.511
    class.f32_cos                   275.672
    class.f32_div                    16.270
    class.f32_exp                   452.489
    class.f32_log                   416.129
    class.f32_op                      4.628
    class.f32_sin                   271.353
    class.f32_sqrt                   22.694
    class.int_op                     16.380
    class.iteration                   0.694
    class.load_dtcm                   4.104
    class.load_ram                    6.599
    class.local                      22.969
    class.store_dtcm                  3.099
    class.store_ram                   7.294
    class.vec_cos                   353.157
    class.vec_exp                   581.657
    class.vec_f32_div                 4.996
    class.vec_load_dtcm               3.847
    class.vec_load_ram                6.735
    class.vec_load_strided_dtcm       6.357
    class.vec_load_strided_ram       20.042
    class.vec_log                   427.897
    class.vec_other                  17.164
    class.vec_sin                   347.657
    class.vec_sqrt                   11.952
    class.vec_store_dtcm              3.979
    class.vec_store_ram               7.933
    invocation                      270.594
    zvt.bf16.cols_k.ram               3.963
    zvt.bf16.packed.call           3268.907
    zvt.bf16.packed.cols_k.ram        0.426
    zvt.bf16.packed.rows_k.dtcm       0.260
    zvt.bf16.packed.rows_k.ram        0.959
    zvt.bf16.packed.tiles.ram       481.179
    zvt.bf16.rows_k.dtcm              1.302
    zvt.bf16.rows_k.ram               4.303
    zvt.bf16.tiles.ram              339.540
    zvt.f32.cols_k.dtcm               0.860
    zvt.f32.cols_k.ram                3.898
    zvt.f32.rows_k.dtcm               1.375
    zvt.f32.rows_k.ram                4.392
    zvt.f32.tiles.dtcm              313.930
    zvt.f32.tiles.ram               380.462 |}]

let%expect_test "fit and check" =
  List.iter report Cases.all;
  [%expect
    {|
    cos 16x16                          measured      23404 predicted      23396   -0.0%
    cos 16x16, scalar                  measured      72851 predicted      72875   +0.0%
    cos 32x32                          measured      92716 predicted      92749   +0.0%
    cos 32x32, scalar                  measured     290764 predicted     290667   -0.0%
    exp 16x16                          measured      38028 predicted      38020   -0.0%
    exp 16x16, scalar                  measured     118121 predicted     118140   +0.0%
    exp 32x32                          measured     151212 predicted     151245   +0.0%
    exp 32x32, scalar                  measured     471801 predicted     471727   -0.0%
    log 16x16                          measured      28187 predicted      28179   -0.0%
    log 16x16, scalar                  measured     108809 predicted     108832   +0.0%
    log 32x32                          measured     111851 predicted     111882   +0.0%
    log 32x32, scalar                  measured     434588 predicted     434495   -0.0%
    sin 16x16                          measured      23052 predicted      23044   -0.0%
    sin 16x16, scalar                  measured      71747 predicted      71770   +0.0%
    sin 32x32                          measured      91308 predicted      91341   +0.0%
    sin 32x32, scalar                  measured     286335 predicted     286245   -0.0%
    sqrt 16x16                         measured       1563 predicted       1559   -0.3%
    sqrt 16x16, scalar                 measured       8093 predicted       8113   +0.2%
    sqrt 32x32                         measured       5387 predicted       5400   +0.2%
    sqrt 32x32, scalar                 measured      31695 predicted      31618   -0.2%
    add 16x64                          measured       3127 predicted       3314   +6.0%
    add 16x64, scalar                  measured      18765 predicted      17309   -7.8%
    add 32x128                         measured      18711 predicted      19430   +3.8%
    add 32x128, scalar                 measured      86637 predicted      95802  +10.6%
    div 16x64                          measured       5274 predicted       4593  -12.9%
    div 16x64, scalar                  measured      30936 predicted      29231   -5.5%
    div 32x128                         measured      21672 predicted      24546  +13.3%
    div 32x128, scalar                 measured     135333 predicted     143490   +6.0%
    mul 16x64                          measured       3127 predicted       3314   +6.0%
    mul 16x64, scalar                  measured      18765 predicted      17309   -7.8%
    mul 32x128                         measured      18711 predicted      19430   +3.8%
    mul 32x128, scalar                 measured      86637 predicted      95802  +10.6%
    x*2 16x64                          measured       2423 predicted       2398   -1.0%
    x*2 16x64, scalar                  measured      15658 predicted      13107  -16.3%
    x*2 32x128                         measured      12761 predicted      12602   -1.2%
    x*2 32x128, scalar                 measured      61963 predicted      68772  +11.0%
    x*x+x 16x64                        measured       6188 predicted       6629   +7.1%
    x*x+x 16x64, scalar                measured      34422 predicted      30415  -11.6%
    x*x+x 32x128                       measured      34158 predicted      35903   +5.1%
    x*x+x 32x128, scalar               measured     148599 predicted     164573  +10.7%
    x+y+z 16x64                        measured       6254 predicted       6629   +6.0%
    x+y+z 16x64, scalar                measured      37530 predicted      34618   -7.8%
    x+y+z 32x128                       measured      37422 predicted      38860   +3.8%
    x+y+z 32x128, scalar               measured     173274 predicted     191603  +10.6%
    add 16x64, in RAM                  measured       5671 predicted       5805   +2.4%
    add 32x128, in RAM                 measured      21751 predicted      22387   +2.9%
    add 16x64, scalar, in RAM          measured      24925 predicted      26714   +7.2%
    softmax over C 16x16               measured     262426 predicted     262528   +0.0%
    softmax over C 32x32               measured    1048224 predicted    1047808   -0.0%
    sum over C 16x64                   measured       1838 predicted       2012   +9.4%
    sum over C 32x128                  measured       7055 predicted       7027   -0.4%
    sum over W 16x64                   measured       1544 predicted       1435   -7.1%
    transpose 32x32                    measured       3403 predicted       2983  -12.3%
    transpose 64x64                    measured      15947 predicted      15126   -5.2%
    transpose 32x32, in RAM            measured       7499 predicted       7499   -0.0%
    head split 16x64, 2 heads          measured     175800 predicted     180084   +2.4%
    head split 32x128, 4 heads         measured     746468 predicted     738721   -1.0%
    head split 49x64, 2 heads          measured     578693 predicted     566540   -2.1%
    linear 9x32 k64                    measured       4895 predicted       4895   -0.0%
    linear 16x64 k64                   measured      14824 predicted      14827   +0.0%
    linear 64x64 k64                   measured     107982 predicted     108054   +0.1%
    linear 16x128 k32                  measured      20036 predicted      20148   +0.6%
    linear 32x32 k128                  measured      47891 predicted      47863   -0.1%
    linear 48x80 k32                   measured      37714 predicted      37493   -0.6%
    linear bf16 9x32 k64               measured       4241 predicted       4489   +5.9%
    linear bf16 16x64 k64              measured      11304 predicted      10407   -7.9%
    linear bf16 64x64 k64              measured      37789 predicted      37789   +0.0%
    linear bf16 32x128 k32             measured      27091 predicted      28654   +5.8%
    linear bf16 48x48 k48              measured      18544 predicted      18315   -1.2%
    sdpa bf16 32x32 d32                measured      87669 predicted      80665   -8.0%
    sdpa bf16 64x64 d64                measured     386501 predicted     386501   +0.0%
    sdpa bf16 16x48 d32                measured      66170 predicted      69695   +5.3%
    sdpa bf16 16x16 d32 bk16           measured      25119 predicted      25854   +2.9%
    sdpa bf16 32x64 d64 bk16           measured     208447 predicted     210057   +0.8%
    sdpa bf16 64x32 d32                measured     175321 predicted     174182   -0.6%
    linear 9x32 k64, in RAM            measured      15217 predicted      15217   -0.0%
    linear 32x32 k64, in RAM           measured      39637 predicted      39637   -0.0%
    linear bf16 16x256 k64             measured     103194 predicted     103194   +0.0%
    linear bf16 32x128 k128            measured     193518 predicted     193518   +0.0%
    linear bf16 49x512 k64             measured     787031 predicted     787031   -0.0%
    linear bf16 16x256 k64, in RAM     measured     156412 predicted     156412   -0.0%
    linear bf16 16x64 k64, in RAM      measured      18984 predicted      15017  -20.9%
    linear bf16 64x64 k64, in RAM      measured      43933 predicted      49244  +12.1%
    sdpa bf16 32x32 d32, in RAM        measured      89717 predicted      87331   -2.7%
    sdpa bf16 16x16 d32 bk16, in RAM   measured      25759 predicted      26152   +1.5%
    sdpa bf16 64x64 d64, in RAM        measured     386501 predicted     386501   +0.0%
    sdpa bf16 9x29 d64 masked          measured      45247 predicted      45095   -0.3%
    exp 64x64                          measured     607246 predicted     608173   +0.2%
    softmax over C 64x64               measured    4190117 predicted    4204617   +0.3%
    softmax over W 16x16               measured     262426 predicted     262528   +0.0%
    layer norm over C 16x64            measured     148915 predicted     157273   +5.6%
    rms norm over C 16x64              measured      61083 predicted      57279   -6.2%
    linear 32x96 k48                   measured      61656 predicted      61879   +0.4%
    sdpa bf16 41x61 d32                measured     241222 predicted     218386   -9.5%
    layer, bare                        measured     260314 predicted     245385   -5.7%
    layer, all flags                   measured     323334 predicted     302976   -6.3%
    layer, pre layer norm, rope        measured     436619 predicted     503711  +15.4%
    layer, post rms                    measured     373465 predicted     378575   +1.4%
    layer decomposed, bare             measured    2812560 predicted    2851623   +1.4% |}]

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
    committed rates match the fit |}]

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
                 (features c))
          in
          Fmt.pr "%s: measured %.0f predicted %.0f %+.1f%%@." c.Cases.name m p
            (100. *. (p -. m) /. m))
    Cases.elsewhere;
  [%expect
    {| fastvit 49 x 512, 16 x 32: measured 40109063 predicted 40311179 +0.5% |}]
