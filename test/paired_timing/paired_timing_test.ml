open Paired_timing

(* A run whose pair [i] has the gain [g i] percent: the old form takes 1 ms. *)
let run ?(matches = fun _ -> true) g =
  List.init 18 (fun i ->
      {
        old_time = 1e-3;
        new_time = 1e-3 *. exp (-.g i /. 100.);
        output_matches = matches i;
      })

let show name pairs =
  match decide pairs with
  | Ok r -> Fmt.pr "%s: %a@." name pp_report r
  | Error Non_positive_time -> Fmt.pr "%s: a time is not positive@." name
  | Error Too_few_pairs -> Fmt.pr "%s: too few pairs@." name
  | Error Too_many_pairs -> Fmt.pr "%s: too many pairs@." name

let%expect_test "the rule accepts, rejects or leaves a change inconclusive" =
  (* gains spread around a centre, the spread symmetric over the kept pairs *)
  let around centre i = centre +. (float_of_int ((i * 7 mod 15) - 7) *. 0.3) in
  show "gain of 5%" (run (around 5.));
  show "loss of 3%" (run (around (-3.)));
  show "no change" (run (around 0.));
  show "late shape, about -1%" (run (around (-1.)));
  [%expect
    {|
    gain of 5%: median +5.0%, interval [+3.8, +6.2]: accept
    loss of 3%: median -3.0%, interval [-4.2, -1.8]: regression
    no change: median +0.0%, interval [-1.2, +1.2]: inconclusive
    late shape, about -1%: median -1.0%, interval [-2.2, +0.2]: inconclusive |}]

let%expect_test "warm-up pairs do not count, and a differing output refuses" =
  let wild i = if i < 3 then -50. else 5. in
  show "warm-up ignored" (run wild);
  show "output differs in one pair"
    (run (fun _ -> 5.) ~matches:(fun i -> i <> 9));
  show "a differing warm-up pair also refuses"
    (run (fun _ -> 5.) ~matches:(fun i -> i <> 1));
  [%expect
    {|
    warm-up ignored: median +5.0%, interval [+5.0, +5.0]: accept
    output differs in one pair: median +5.0%, interval [+5.0, +5.0]: refused: output differs
    a differing warm-up pair also refuses: median +5.0%, interval [+5.0, +5.0]: refused: output differs |}]

let%expect_test "a run with the wrong number of pairs or a bad time is an error"
    =
  show "short" (List.filteri (fun i _ -> i < 17) (run (fun _ -> 5.)));
  show "long" (run (fun _ -> 5.) @ run (fun _ -> 5.));
  show "zero time"
    (List.mapi
       (fun i p -> if i = 5 then { p with new_time = 0. } else p)
       (run (fun _ -> 5.)));
  [%expect
    {|
    short: too few pairs
    long: too many pairs
    zero time: a time is not positive |}]

(* ---- verdicts ------------------------------------------------------------- *)

let report g =
  match
    decide
      (List.init 18 (fun _ ->
           {
             old_time = 1e-3;
             new_time = 1e-3 *. exp (-.g /. 100.);
             output_matches = true;
           }))
  with
  | Ok r -> r
  | Error _ -> assert false

let key ?(candidate = "fused conv+relu6") ?(version = "v1") () =
  {
    Verdict.target = "aarch64 host";
    numerics = "fp32_ordered";
    candidate;
    baseline = "stored";
    version;
  }

let verdicts =
  [
    { Verdict.key = key (); report = report 5. };
    { Verdict.key = key ~candidate:"fused, 7x7" (); report = report (-3.) };
    { Verdict.key = key ~candidate:"fused, near zero" (); report = report 0. };
  ]

let choose ?reason k =
  Fmt.pr "%a@." Verdict.pp_choice (Verdict.choose verdicts k ?reason ())

let%expect_test "a verdict decides only the candidate it measured" =
  choose (key ());
  choose (key ~candidate:"fused, 7x7" ());
  choose (key ~candidate:"fused, near zero" ());
  choose (key ~candidate:"fused, near zero" ()) ~reason:Verdict.Capacity;
  choose (key ~candidate:"fused, 7x7" ()) ~reason:Verdict.Capacity;
  (* a candidate with no verdict, another home or block size, another version *)
  choose (key ~candidate:"three-node segment" ());
  choose (key ~candidate:"fused conv+relu6, home ram" ());
  choose (key ~version:"v2" ());
  [%expect
    {|
    candidate, accepted at +5.0%
    baseline kept: regression at -3.0%
    baseline kept: inconclusive
    candidate, for capacity (speed +0.0%)
    candidate, for capacity (speed -3.0%)
    baseline kept: no verdict
    baseline kept: no verdict
    baseline kept: no verdict |}]

let%expect_test "a new version invalidates the verdicts, and coverage is exact"
    =
  let live = Verdict.current verdicts ~version:"v2" in
  Fmt.pr "verdicts for v2: %d@." (List.length live);
  let covered ks =
    match Verdict.covered verdicts ks with
    | Ok () -> Fmt.pr "covered@."
    | Error k -> Fmt.pr "no verdict for %s@." k.Verdict.candidate
  in
  covered [ key (); key ~candidate:"fused, 7x7" () ];
  covered [ key (); key ~candidate:"three-node segment" () ];
  [%expect
    {|
    verdicts for v2: 0
    covered
    no verdict for three-node segment |}]

(* ---- the matched-policy golden -------------------------------------------- *)

let%expect_test "the golden is bitwise, matched to its policy, and can fail" =
  let values = [| 1.5; -0.25; 3.0e-5; Float.nan |] in
  let g = { Golden.policy = "fp32_ordered"; values } in
  let show name r =
    match r with
    | Ok () -> Fmt.pr "%s: equal@." name
    | Error m -> Fmt.pr "%s: %a@." name Golden.pp_mismatch m
  in
  show "same outputs"
    (Golden.compare g ~policy:"fp32_ordered" (Array.copy values));
  show "one bit off"
    (Golden.compare g ~policy:"fp32_ordered" (Golden.with_defect values));
  show "another policy" (Golden.compare g ~policy:"fp32_relaxed" values);
  show "shorter" (Golden.compare g ~policy:"fp32_ordered" [| 1.5 |]);
  show "negative zero is not zero"
    (Golden.compare
       { g with Golden.values = [| 0. |] }
       ~policy:"fp32_ordered" [| -0. |]);
  [%expect
    {|
    same outputs: equal
    one bit off: element 0: 3ff8000000000001, expected 3ff8000000000000
    another policy: compared under fp32_relaxed, the golden was made under fp32_ordered
    shorter: length 1, expected 4
    negative zero is not zero: element 0: 8000000000000000, expected 0000000000000000 |}]
