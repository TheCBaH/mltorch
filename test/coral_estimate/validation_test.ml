(* Accuracy limits are assertions, independent of the diagnostic snapshots.
   Gaps have names and reasons: adding a case cannot silently enlarge the
   validated domain, and promoting an expect output cannot relax a limit. *)

module Cases = Coral_estimate_cases
module F = Fit_test
module CM = Coral_estimate.Coral_model

let refused =
  [ "softmax over C 16x16"; "softmax over C 32x32"; "softmax over C 64x64" ]

let gaps =
  [
    ("layer norm over C 16x64", "ordered lane fold priced as scalar extracts");
    ("rms norm over C 16x64", "ordered lane fold priced as scalar extracts");
    ( "layer decomposed, bare",
      "data-dependent control flow counted heuristically" );
  ]

(* Initial envelopes for the existing held-out data, in percent. They state
   the current validated domain, not a claim that a 30% model is adequate for
   choosing between two kernels separated by a few percent. *)
let limits =
  [
    ("exp 64x64", 2.);
    ("softmax over W 16x16", 5.);
    ("linear 32x96 k48", 2.);
    ("sdpa bf16 41x61 d32", 10.);
    ("layer, bare", 12.);
    ("layer, all flags", 3.);
    ("layer, pre layer norm, rope", 30.);
    ("layer, post rms", 10.);
  ]

let case name = List.find (fun (c : Cases.t) -> c.name = name) Cases.all

let error rates (c : Cases.t) =
  match F.cycles c.name with
  | None -> Fmt.failwith "%s: missing measurement" c.name
  | Some ys ->
      let measured = List.fold_left ( +. ) 0. ys in
      let predicted =
        List.fold_left ( +. ) 0.
          (List.map (CM.Rates.predict rates) (F.features ~rates c))
      in
      if not (Float.is_finite predicted && predicted > 0. && measured > 0.) then
        Fmt.failwith "%s: invalid prediction or measurement" c.name;
      100. *. (predicted -. measured) /. measured

let check rates =
  List.iter
    (fun (name, limit) ->
      let c = case name in
      if c.role <> Cases.Role.Check then
        Fmt.failwith "%s: accuracy gate must be held out of fitting" name;
      if
        List.exists
          (fun (x : Coral_estimate.Coral_census.t) -> x.unknown > 0)
          (F.censuses c)
      then Fmt.failwith "%s: accuracy gate requires known control flow" name;
      let miss = error rates c in
      if Float.abs miss > limit then
        Fmt.failwith "%s: error %+.2f%% exceeds %.1f%%" name miss limit)
    limits

let rounded_error rates c = (Float.round (10. *. error rates c) /. 10.) +. 0.

let%test_unit "every case has measurements or an explicit RTL refusal" =
  let lines =
    In_channel.with_open_bin "../coral_estimate_rtl/calibrate_test.ml"
      In_channel.input_all
    |> String.split_on_char '\n' |> List.map String.trim
  in
  List.iter
    (fun (c : Cases.t) ->
      match (F.cycles c.name, List.mem c.name refused) with
      | Some ys, false ->
          if
            ys = []
            || List.exists (fun y -> not (Float.is_finite y && y > 0.)) ys
          then Fmt.failwith "%s: empty or invalid measurements" c.name;
          let prefix = c.name ^ " (" in
          let header = List.find (String.starts_with ~prefix) lines in
          let declared =
            Scanf.sscanf
              (String.sub header (String.length c.name)
                 (String.length header - String.length c.name))
              " (%d invocations)" Fun.id
          in
          if
            List.length ys <> declared || List.length (F.censuses c) <> declared
          then Fmt.failwith "%s: incomplete invocation measurements" c.name
      | None, true ->
          let refusal =
            c.name
            ^ ": refused: invocation 0 (n0): itcm: 11352 bytes needed, 8192 \
               available"
          in
          if not (List.mem refusal lines) then
            Fmt.failwith "%s: expected refusal changed or disappeared" c.name
      | Some _, true -> Fmt.failwith "%s: remove resolved refusal" c.name
      | None, false -> Fmt.failwith "%s: missing calibration data" c.name)
    Cases.all

let%test_unit "every held-out case has an explicit validation status" =
  let expected =
    List.map fst limits @ List.map fst gaps
    @ List.filter (fun n -> (case n).role = Cases.Role.Check) refused
  in
  let actual = List.map (fun (c : Cases.t) -> c.name) Cases.check in
  assert (List.sort compare expected = List.sort compare actual)

let%test_unit "committed rates satisfy held-out accuracy limits" =
  check Coral_estimate.Coral_rates.vme

let%test_unit "the held-out gate rejects a lost transcendental cost" =
  let rates =
    List.remove_assoc "class.vec_exp" Coral_estimate.Coral_rates.vme
  in
  let rejected =
    try
      check rates;
      false
    with Failure _ -> true
  in
  assert rejected

let%expect_test "held-out accuracy domain and known gaps" =
  let rates = Coral_estimate.Coral_rates.vme in
  List.iter
    (fun (name, limit) ->
      Fmt.pr "%s: %+.1f%%, limit %.1f%%@." name
        (rounded_error rates (case name))
        limit)
    limits;
  List.iter
    (fun (name, reason) ->
      Fmt.pr "%s: outside accuracy gate (%s), error %+.1f%%@." name reason
        (rounded_error rates (case name)))
    gaps;
  List.iter
    (fun name -> Fmt.pr "%s: RTL refused (ITCM capacity)@." name)
    refused;
  [%expect
    {|
    exp 64x64: -0.4%, limit 2.0%
    softmax over W 16x16: +1.9%, limit 5.0%
    linear 32x96 k48: +0.7%, limit 2.0%
    sdpa bf16 41x61 d32: -9.6%, limit 10.0%
    layer, bare: -9.8%, limit 12.0%
    layer, all flags: -1.0%, limit 3.0%
    layer, pre layer norm, rope: +25.2%, limit 30.0%
    layer, post rms: +7.5%, limit 10.0%
    layer norm over C 16x64: outside accuracy gate (ordered lane fold priced as scalar extracts), error +590.6%
    rms norm over C 16x64: outside accuracy gate (ordered lane fold priced as scalar extracts), error +507.2%
    layer decomposed, bare: outside accuracy gate (data-dependent control flow counted heuristically), error -22.2%
    softmax over C 16x16: RTL refused (ITCM capacity)
    softmax over C 32x32: RTL refused (ITCM capacity)
    softmax over C 64x64: RTL refused (ITCM capacity) |}]
