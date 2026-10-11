open Inference_compare_schema
module Core = Inference_compare_core
module K = Coord_fixture
module R = Inference_compare_engines.Synthetic_runner
module C = Chain_fixture
module P = Plan_fixtures
module L = Inference_compare_codecs.Logical_value

let u1u2 = K.make C.u1u2

let right_deviates =
  R.{ (default_config "right") with offsets = [ ("n1_r", 1L) ] }

let pp_list l = String.concat " | " l

let report name (run : K.run) =
  Format.printf "%s:@.  %s@.  comparisons: %s@." name (K.completion run.result)
    (pp_list (K.verdicts run.result));
  match run.result.failure with
  | Some f ->
      Format.printf "  failure: %s in phase %s@."
        (match Failure.encode_code f.code with
        | Jsont.String (s, _) -> s
        | _ -> "?")
        (match Phase.encode f.phase with Jsont.String (s, _) -> s | _ -> "?")
  | None -> ()

let%expect_test "reset divergence at every cluster, both donors" =
  let go donor =
    let run = K.execute u1u2 (K.setup ~right:right_deviates ~donor ()) in
    report (Printf.sprintf "donor %s" (Side.to_string donor)) run;
    Format.printf "  final y = %s@."
      (pp_list
         (List.map
            (fun (n, xs) ->
              n ^ "=" ^ Format.asprintf "%a" Codec_fixtures.pp_ints xs)
            (K.result_ints run.ctx run.result)));
    Format.printf "  left engine saw:  %s@.  right engine saw: %s@."
      (pp_list (K.observed run.left))
      (pp_list (K.observed run.right));
    List.iter (fun (_, v) -> L.drop v) run.result.finals
  in
  go Left;
  go Right;
  [%expect
    {|
    donor left:
      execution=complete numerical=disagreement coverage=complete
      comparisons: cmp_h=mismatch | cmp_y=pass
      final y = out0=[200; 300; 400]
      left engine saw:  x=[1; 2; 3] | h=[2; 3; 4]
      right engine saw: x_r=[1; 2; 3] | h_r=[2; 3; 4]
    donor right:
      execution=complete numerical=disagreement coverage=complete
      comparisons: cmp_h=mismatch | cmp_y=pass
      final y = out0=[300; 400; 500]
      left engine saw:  x=[1; 2; 3] | h=[3; 4; 5]
      right engine saw: x_r=[1; 2; 3] | h_r=[3; 4; 5] |}]

let%expect_test "events: both outputs compared before any value is committed" =
  let run = K.execute u1u2 (K.setup ~right:right_deviates ()) in
  Format.printf "%s@." (pp_list (List.map K.event_name run.result.events));
  let seqs = List.map (fun (e : Event.t) -> e.sequence) run.result.events in
  Format.printf "sequence monotone from zero: %b@."
    (seqs = List.init (List.length seqs) Int64.of_int);
  List.iter (fun (_, v) -> L.drop v) run.result.finals;
  [%expect
    {|
    U1:inputs_verified | U1:graph_completed(left) | U1:graph_completed(right) | U1:outputs_compared | U1:frontier_committed | U2:inputs_verified | U2:graph_completed(left) | U2:graph_completed(right) | U2:outputs_compared | U2:frontier_committed
    sequence monotone from zero: true |}]

let%expect_test
    "lineage: committed value descends from the donor's computed output" =
  let run = K.execute u1u2 (K.setup ~right:right_deviates ~donor:Right ()) in
  let origin inst = List.assoc inst run.result.lineage in
  (match run.result.commits with
  | (u, [ c ]) :: _ ->
      Format.printf "%a committed %a from donor %s@." Id.Unit.pp u
        Id.Canonical_value.pp c.key (Side.to_string c.donor);
      let rec describe depth inst =
        match origin inst with
        | Computed { side; unit; tensor } ->
            Format.printf "%sraw output of %a on the %s side (%a)@."
              (String.make depth ' ') Id.Unit.pp unit (Side.to_string side)
              Id.Tensor_local.pp tensor.local_id
        | Converted { parents; recipe } ->
            Format.printf "%sconverted via %a@." (String.make depth ' ')
              Id.Recipe.pp recipe;
            List.iter (describe (depth + 2)) parents
        | Initial { key; _ } ->
            Format.printf "%sinitial %a@." (String.make depth ' ')
              Id.Canonical_value.pp key
      in
      describe 2 c.converted_instance
  | _ -> Format.printf "no commits@.");
  List.iter (fun (_, v) -> L.drop v) run.result.finals;
  [%expect
    {|
    U1 committed h from donor right
      converted via or_U1_right_h
        converted via or_U1_right_h
          raw output of U1 on the right side (h_r) |}]

let%expect_test "an identical pair agrees completely and cleans up" =
  let run = K.execute u1u2 (K.setup ()) in
  report "identical engines" run;
  Format.printf "  prepared open %d/%d, values live %d/%d@."
    run.left.prepared_open run.right.prepared_open run.left.values_live
    run.right.values_live;
  Format.printf "  ledger holds only the final outputs: %d blob(s)@."
    (Inference_compare_codecs.Owned_blob.Ledger.live_blobs run.ctx.ledger);
  List.iter (fun (_, v) -> L.drop v) run.result.finals;
  Format.printf "  ledger balanced after the caller drops them: %b@."
    (Inference_compare_codecs.Owned_blob.Ledger.balanced run.ctx.ledger);
  [%expect
    {|
    identical engines:
      execution=complete numerical=agreement coverage=complete
      comparisons: cmp_h=pass | cmp_y=pass
      prepared open 0/0, values live 0/0
      ledger holds only the final outputs: 1 blob(s)
      ledger balanced after the caller drops them: true |}]

let%expect_test "input corruption blocks both executions" =
  let corrupt =
    R.{ (default_config "right") with corrupt_imports = [ (0, 5L) ] }
  in
  let run = K.execute u1u2 (K.setup ~right:corrupt ()) in
  report "corrupted right import" run;
  Format.printf "  events: %s@."
    (pp_list (List.map K.event_name run.result.events));
  Format.printf
    "  runs: left %d, right %d; nodes executed: %d/%d; published units: %d@."
    run.left.runs run.right.runs
    (List.length run.left.executed_nodes)
    (List.length run.right.executed_nodes)
    (List.length run.result.executed);
  Format.printf
    "  cleanup: prepared open %d/%d, values live %d/%d, ledger balanced %b@."
    run.left.prepared_open run.right.prepared_open run.left.values_live
    run.right.values_live
    (Inference_compare_codecs.Owned_blob.Ledger.balanced run.ctx.ledger);
  [%expect
    {|
    corrupted right import:
      execution=failed numerical=not_established coverage=complete
      comparisons:
      failure: input_transfer_failed in phase input_check
      events: U1:failed:input_transfer_failed
      runs: left 0, right 0; nodes executed: 0/0; published units: 0
      cleanup: prepared open 0/0, values live 0/0, ledger balanced true |}]

let%expect_test "an execution failure never borrows the other side's answer" =
  let two = K.make C.u1u2 in
  let show name left right =
    let run = K.execute two (K.setup ~left ~right ()) in
    Format.printf "%s: %s; events %s; published %d; finals %d; balanced %b@."
      name (K.completion run.result)
      (pp_list (List.map K.event_name run.result.events))
      (List.length run.result.executed)
      (List.length run.result.finals)
      (Inference_compare_codecs.Owned_blob.Ledger.balanced run.ctx.ledger)
  in
  show "left fails"
    R.{ (default_config "left") with fail_nodes = [ "n1" ] }
    (R.default_config "right");
  show "right fails after left completed" (R.default_config "left")
    R.{ (default_config "right") with fail_nodes = [ "n1_r" ] };
  show "second unit fails" (R.default_config "left")
    R.{ (default_config "right") with fail_nodes = [ "n2_r" ] };
  [%expect
    {|
    left fails: execution=failed numerical=not_established coverage=complete; events U1:inputs_verified | U1:failed:execution_failed; published 0; finals 0; balanced true
    right fails after left completed: execution=failed numerical=not_established coverage=complete; events U1:inputs_verified | U1:graph_completed(left) | U1:failed:execution_failed; published 0; finals 0; balanced true
    second unit fails: execution=failed numerical=not_established coverage=complete; events U1:inputs_verified | U1:graph_completed(left) | U1:graph_completed(right) | U1:outputs_compared | U1:frontier_committed | U2:inputs_verified | U2:graph_completed(left) | U2:failed:execution_failed; published 1; finals 0; balanced true |}]

let%expect_test "an evidence failure publishes nothing" =
  let calls = ref 0 in
  let refuse_commit (e : Event.t) =
    match e.payload with
    | Frontier_committed _ ->
        incr calls;
        Error "disk full"
    | _ -> Ok ()
  in
  let run = K.execute u1u2 (K.setup ~emit:refuse_commit ()) in
  report "sink refuses the commit event" run;
  Format.printf "  published units: %d, second unit started: %b@."
    (List.length run.result.executed)
    (run.left.runs > 1);
  Format.printf "  events kept: %s@."
    (pp_list (List.map K.event_name run.result.events));
  Format.printf "  balanced: %b@."
    (Inference_compare_codecs.Owned_blob.Ledger.balanced run.ctx.ledger);
  [%expect
    {|
    sink refuses the commit event:
      execution=failed numerical=not_established coverage=complete
      comparisons: cmp_h=pass
      failure: evidence_failed in phase commit
      published units: 0, second unit started: false
      events kept: U1:inputs_verified | U1:graph_completed(left) | U1:graph_completed(right) | U1:outputs_compared | U1:failed:evidence_failed
      balanced: true |}]

(* U = {n1: x+1 -> a ; n2: 2x -> b}; V = a + b -> y. The right engine
   deviates at n1 only, so the outputs a differ and b agree. *)
let pair =
  K.make
    (C.synthetic
       [
         C.node "n1" "addc:1" [ "x" ] [ "a" ];
         C.node "n2" "mulc:2" [ "x" ] [ "b" ];
         C.node "n3" "add" [ "a"; "b" ] [ "y" ];
       ]
       ~sources:[ "x" ] ~finals:[ "y" ]
       ~units:[ ("U", [ "n1"; "n2" ]); ("V", [ "n3" ]) ])

let%expect_test
    "donor overrides choose per output; both next engines see the bundle" =
  let right = R.{ (default_config "right") with offsets = [ ("n1_r", 1L) ] } in
  let ov key donor : Donor_override.t =
    { unit = P.uid "U"; output = P.ckey key; donor }
  in
  let show name s =
    let run = K.execute pair s in
    Format.printf "%s: %s | V left saw %s | V right saw %s | y=%s@." name
      (pp_list (K.verdicts run.result))
      (List.nth (K.observed run.left) 1)
      (List.nth (K.observed run.right) 1)
      (match K.result_ints run.ctx run.result with
      | [ (_, xs) ] -> Format.asprintf "%a" Codec_fixtures.pp_ints xs
      | _ -> "-");
    List.iter (fun (_, v) -> L.drop v) run.result.finals
  in
  show "all left" (K.setup ~right ());
  show "all right" (K.setup ~right ~donor:Right ());
  show "a from right, b default left"
    (K.setup ~right ~overrides:[ ov "a" Right ] ());
  show "b from right (they agree), a default left"
    (K.setup ~right ~overrides:[ ov "b" Right ] ());
  [%expect
    {|
    all left: cmp_a=mismatch | cmp_b=pass | cmp_y=pass | V left saw a=[2; 3; 4] b=[2; 4; 6] | V right saw a_r=[2; 3; 4] b_r=[2; 4; 6] | y=[4; 7; 10]
    all right: cmp_a=mismatch | cmp_b=pass | cmp_y=pass | V left saw a=[3; 4; 5] b=[2; 4; 6] | V right saw a_r=[3; 4; 5] b_r=[2; 4; 6] | y=[5; 8; 11]
    a from right, b default left: cmp_a=mismatch | cmp_b=pass | cmp_y=pass | V left saw a=[3; 4; 5] b=[2; 4; 6] | V right saw a_r=[3; 4; 5] b_r=[2; 4; 6] | y=[5; 8; 11]
    b from right (they agree), a default left: cmp_a=mismatch | cmp_b=pass | cmp_y=pass | V left saw a=[2; 3; 4] b=[2; 4; 6] | V right saw a_r=[2; 3; 4] b_r=[2; 4; 6] | y=[4; 7; 10] |}]

let%expect_test
    "unknown, duplicate and unused overrides are refused before anything runs" =
  let right = right_deviates in
  let ov unit key donor : Donor_override.t =
    { unit = P.uid unit; output = P.ckey key; donor }
  in
  let show name overrides =
    let run = K.execute u1u2 (K.setup ~right ~overrides ()) in
    Format.printf "%s: %s; engine runs %d/%d; events %s@." name
      (K.completion run.result) run.left.runs run.right.runs
      (pp_list (List.map K.event_name run.result.events))
  in
  show "unknown output" [ ov "U1" "zz" Right ];
  show "wrong unit" [ ov "U2" "h" Right ];
  show "duplicate" [ ov "U1" "h" Right; ov "U1" "h" Left ];
  [%expect
    {|
    unknown output: execution=refused numerical=not_established coverage=complete; engine runs 0/0; events U1:failed:unknown_override
    wrong unit: execution=refused numerical=not_established coverage=complete; engine runs 0/0; events U2:failed:unknown_override
    duplicate: execution=refused numerical=not_established coverage=complete; engine runs 0/0; events U1:failed:unknown_override |}]

(* y = (x + w) * w + x : a constant used in two units and a residual operand. *)
let residual =
  K.make
    (C.synthetic
       [
         C.node "n1" "add" [ "x"; "w" ] [ "h" ];
         C.node "n2" "mul" [ "h"; "w" ] [ "m" ];
         C.node "n3" "add" [ "m"; "x" ] [ "y" ];
       ]
       ~sources:[ "x"; "w" ] ~finals:[ "y" ]
       ~units:[ ("A", [ "n1" ]); ("B", [ "n2" ]); ("D", [ "n3" ]) ])

let%expect_test "constants and residual branches reach both engines unchanged" =
  let run = K.execute residual (K.setup ()) in
  report "residual network" run;
  (* the elements only: the two engines name their tensors differently *)
  let strip s =
    String.concat " "
      (List.map
         (fun w ->
           match String.index_opt w '=' with
           | Some i -> String.sub w (i + 1) (String.length w - i - 1)
           | None -> w)
         (String.split_on_char ' ' s))
  in
  List.iteri
    (fun i (l, r) ->
      Format.printf "  run %d: left %s | right %s | same=%b@." i (strip l)
        (strip r)
        (strip l = strip r))
    (List.combine (K.observed run.left) (K.observed run.right));
  Format.printf "  y = %s (expected (x+w)*w+x for x=[1;2;3], w=[10;20;30])@."
    (match K.result_ints run.ctx run.result with
    | [ (_, xs) ] -> Format.asprintf "%a" Codec_fixtures.pp_ints xs
    | _ -> "-");
  List.iter (fun (_, v) -> L.drop v) run.result.finals;
  [%expect
    {|
    residual network:
      execution=complete numerical=agreement coverage=complete
      comparisons: cmp_h=pass | cmp_m=pass | cmp_y=pass
      run 0: left [1; 2; 3] [10; 20; 30] | right [1; 2; 3] [10; 20; 30] | same=true
      run 1: left [11; 22; 33] [10; 20; 30] | right [11; 22; 33] [10; 20; 30] | same=true
      run 2: left [110; 440; 990] [1; 2; 3] | right [110; 440; 990] [1; 2; 3] | same=true
      y = [111; 442; 993] (expected (x+w)*w+x for x=[1;2;3], w=[10;20;30]) |}]

let%expect_test "reused engine output storage cannot change committed values" =
  let reuse = R.{ (default_config "left") with reuse_output_storage = true } in
  let baseline = K.execute residual (K.setup ()) in
  let reused =
    K.execute residual
      (K.setup ~left:reuse ~right:{ reuse with name = "right" } ())
  in
  Format.printf "same final outputs with and without reuse: %b@."
    (K.result_ints baseline.ctx baseline.result
    = K.result_ints reused.ctx reused.result);
  Format.printf "reused run: %s@." (K.completion reused.result);
  List.iter (fun (_, v) -> L.drop v) baseline.result.finals;
  List.iter (fun (_, v) -> L.drop v) reused.result.finals;
  [%expect
    {|
    same final outputs with and without reuse: true
    reused run: execution=complete numerical=agreement coverage=complete |}]

let%expect_test "preparation failures close everything and run nothing" =
  let run =
    K.execute residual
      (K.setup
         ~right:R.{ (default_config "right") with refuse_ops = [ "mul" ] }
         ())
  in
  report "right refuses the second unit's operator" run;
  Format.printf
    "  runs %d/%d, prepared open %d/%d (units A and D were prepared first)@."
    run.left.runs run.right.runs run.left.prepared_open run.right.prepared_open;
  Format.printf "  prepares left %d, right %d@." run.left.prepares
    run.right.prepares;
  (match run.result.failure with
  | Some f ->
      Format.printf "  failure on side %s, unit %s@."
        (match f.side with Some s -> Side.to_string s | None -> "-")
        (match f.unit with Some u -> Id.Unit.to_string u | None -> "-")
  | None -> ());
  [%expect
    {|
    right refuses the second unit's operator:
      execution=refused numerical=not_established coverage=complete
      comparisons:
      failure: preparation_refused in phase preparation
      runs 0/0, prepared open 0/0 (units A and D were prepared first)
      prepares left 2, right 1
      failure on side right, unit B |}]

let%expect_test "engine identity must match the pin" =
  let lr, _ = R.create (R.default_config "left") in
  let module M = (val lr) in
  let pinned = M.identity () in
  let ok =
    K.execute u1u2 (K.setup ~expected_identities:(Some pinned, None) ())
  in
  List.iter (fun (_, v) -> L.drop v) ok.result.finals;
  Format.printf "pinned and unchanged: %s@." (K.completion ok.result);
  let drift =
    K.execute u1u2
      (K.setup
         ~left:R.{ (default_config "left") with offsets = [ ("n1_r", 1L) ] }
         ~expected_identities:(Some pinned, None) ())
  in
  Format.printf "configuration drifted: %s; runs %d@."
    (K.completion drift.result)
    drift.left.runs;
  [%expect
    {|
    pinned and unchanged: execution=complete numerical=agreement coverage=complete
    configuration drifted: execution=refused numerical=not_established coverage=complete; runs 0 |}]

let%expect_test "pinned initial payloads are verified" =
  let ictx = Codec_fixtures.context () in
  let wrong =
    [ (P.ckey "x", Codec_fixtures.integers ictx C.i64_spec [ 9L; 9L; 9L ]) ]
  in
  let run = K.execute u1u2 (K.setup ~initial_override:wrong ()) in
  report "initial value differs from its pinned digest" run;
  Format.printf "  runs %d/%d@." run.left.runs run.right.runs;
  let missing = K.execute u1u2 (K.setup ~initial_override:[] ()) in
  report "initial value absent" missing;
  [%expect
    {|
    initial value differs from its pinned digest:
      execution=refused numerical=not_established coverage=complete
      comparisons:
      failure: conflicting_initial_value in phase planning
      runs 0/0
    initial value absent:
      execution=refused numerical=not_established coverage=complete
      comparisons:
      failure: missing_common_input in phase planning |}]

let%expect_test "a live-byte limit fails the unit without partial publication" =
  let run = K.execute residual (K.setup ~limit:200L ()) in
  report "tight ledger" run;
  Format.printf "  published %d unit(s); balanced %b@."
    (List.length run.result.executed)
    (Inference_compare_codecs.Owned_blob.Ledger.balanced run.ctx.ledger);
  [%expect
    {|
    tight ledger:
      execution=failed numerical=not_established coverage=complete
      comparisons:
      failure: resource_exhausted in phase input_check
      published 0 unit(s); balanced true |}]

let%expect_test "two runs from fresh state agree" =
  let a = K.execute residual (K.setup ~right:right_deviates ()) in
  let b = K.execute residual (K.setup ~right:right_deviates ()) in
  Format.printf "same verdicts: %b, same finals: %b, same event kinds: %b@."
    (K.verdicts a.result = K.verdicts b.result)
    (K.result_ints a.ctx a.result = K.result_ints b.ctx b.result)
    (List.map K.event_name a.result.events
    = List.map K.event_name b.result.events);
  List.iter (fun (_, v) -> L.drop v) a.result.finals;
  List.iter (fun (_, v) -> L.drop v) b.result.finals;
  [%expect {| same verdicts: true, same finals: true, same event kinds: true |}]

let%expect_test "values are released at their last use" =
  (* At each unit's Inputs_verified event the ledger holds the frontier plus
     that unit's own temporaries: two inputs and two readbacks per side (8
     blobs). So the frontier is the count minus 8. *)
  let counts = ref [] in
  let sink (e : Event.t) =
    (match e.payload with
    | Inputs_verified _ -> counts := (!K.probe () - 8) :: !counts
    | _ -> ());
    Ok ()
  in
  let run = K.execute residual (K.setup ~emit:sink ()) in
  Format.printf "frontier blobs when each unit starts: %s@."
    (String.concat ", " (List.rev_map string_of_int !counts));
  (* A: x, w.  B: x, w, h.  D: x, m -- w and h were finished with B. *)
  List.iter (fun (_, v) -> L.drop v) run.result.finals;
  [%expect {| frontier blobs when each unit starts: 2, 3, 2 |}]

let%expect_test "a failed import releases every handle already taken" =
  (* Unit A has two inputs; the right engine's second import fails after its
     first succeeded and after the left engine imported both. *)
  let right = R.{ (default_config "right") with fail_imports = [ 1 ] } in
  let run = K.execute residual (K.setup ~right ()) in
  report "second right import fails" run;
  Format.printf
    "  runs %d/%d; imports left %d right %d; values live %d/%d; prepared open \
     %d/%d; balanced %b@."
    run.left.runs run.right.runs run.left.imports run.right.imports
    run.left.values_live run.right.values_live run.left.prepared_open
    run.right.prepared_open
    (Inference_compare_codecs.Owned_blob.Ledger.balanced run.ctx.ledger);
  [%expect
    {|
    second right import fails:
      execution=failed numerical=not_established coverage=complete
      comparisons:
      failure: input_transfer_failed in phase import
      runs 0/0; imports left 2 right 1; values live 0/0; prepared open 0/0; balanced true |}]

let%expect_test "compile-cache keys track graph and engine identity" =
  let ex u =
    List.find
      (fun (e : Inference_compare_extract.Extraction.t) ->
        Id.Unit.to_string e.unit = u && e.side = Left)
      u1u2.extractions
  in
  let id cfg =
    let r, _ = R.create cfg in
    let module M = (val r) in
    M.identity ()
  in
  let a = id (R.default_config "e")
  and b = id R.{ (default_config "e") with offsets = [ ("n1", 1L) ] } in
  let key = Core.Cache_key.make in
  Format.printf "stable: %b@."
    (key (ex "U1") a = key (ex "U1") (id (R.default_config "e")));
  Format.printf "other unit differs: %b@." (key (ex "U1") a <> key (ex "U2") a);
  Format.printf "other engine configuration differs: %b@."
    (key (ex "U1") a <> key (ex "U1") b);
  [%expect
    {|
    stable: true
    other unit differs: true
    other engine configuration differs: true |}]

let describe_snapshot (s : Core.Coordinator.snapshot) =
  Printf.sprintf "%s %s inputs=%d raw=%d candidates=%d mismatching=%b"
    (Id.Unit.to_string s.unit)
    (match s.stage with
    | Before_commit -> "before_commit"
    | After_failure -> "after_failure")
    (List.length s.common_inputs)
    (List.length s.raw) (List.length s.candidates) s.mismatching

let%expect_test
    "the retention hook sees every unit's payloads before anything is published"
    =
  let seen = ref [] in
  let retain (s : Core.Coordinator.snapshot) =
    seen := describe_snapshot s :: !seen;
    Ok ()
  in
  let run = K.execute u1u2 (K.setup ~right:right_deviates ~retain ()) in
  List.iter print_endline (List.rev !seen);
  Format.printf "published: %d@." (List.length run.result.executed);
  List.iter (fun (_, v) -> L.drop v) run.result.finals;
  [%expect
    {|
    U1 before_commit inputs=1 raw=2 candidates=2 mismatching=true
    U2 before_commit inputs=1 raw=2 candidates=2 mismatching=false
    published: 2 |}]

let%expect_test
    "a retention that cannot be met stops the unit; failures retain their \
     inputs" =
  let seen = ref [] in
  let refuse (s : Core.Coordinator.snapshot) =
    seen := describe_snapshot s :: !seen;
    match s.stage with
    | Before_commit -> Error "retention budget exhausted"
    | After_failure -> Ok ()
  in
  let run = K.execute u1u2 (K.setup ~retain:refuse ()) in
  report "required retention refused" run;
  List.iter print_endline (List.rev !seen);
  Format.printf "published: %d, ledger balanced: %b@."
    (List.length run.result.executed)
    (Inference_compare_codecs.Owned_blob.Ledger.balanced run.ctx.ledger);
  seen := [];
  let failing =
    K.execute u1u2
      (K.setup
         ~left:R.{ (default_config "left") with fail_nodes = [ "n1" ] }
         ~retain:(fun s ->
           seen := describe_snapshot s :: !seen;
           Ok ())
         ())
  in
  ignore failing;
  List.iter print_endline (List.rev !seen);
  [%expect
    {|
    required retention refused:
      execution=failed numerical=not_established coverage=complete
      comparisons:
      failure: evidence_failed in phase commit
    U1 before_commit inputs=1 raw=2 candidates=2 mismatching=false
    U1 after_failure inputs=1 raw=2 candidates=2 mismatching=false
    published: 0, ledger balanced: true
    U1 after_failure inputs=1 raw=0 candidates=0 mismatching=false |}]

let%expect_test "one unit replays from retained common inputs" =
  let ictx = Codec_fixtures.context () in
  let h = Codec_fixtures.integers ictx C.i64_spec [ 2L; 3L; 4L ] in
  let run =
    K.execute u1u2
      (K.setup
         ~mode:(Single_unit { unit = P.uid "U2"; inputs = [ (P.ckey "h", h) ] })
         ())
  in
  report "replay of U2" run;
  Format.printf
    "  executed: %s; finals: %d; engine prepares %d/%d, runs %d/%d@."
    (String.concat "," (List.map Id.Unit.to_string run.result.executed))
    (List.length run.result.finals)
    run.left.prepares run.right.prepares run.left.runs run.right.runs;
  Format.printf "  U2 saw: %s@." (pp_list (K.observed run.left));
  let missing =
    K.execute u1u2
      (K.setup ~mode:(Single_unit { unit = P.uid "U2"; inputs = [] }) ())
  in
  report "replay without its input" missing;
  Format.printf "  runs %d/%d@." missing.left.runs missing.right.runs;
  [%expect
    {|
    replay of U2:
      execution=complete numerical=agreement coverage=complete
      comparisons: cmp_y=pass
      executed: U2; finals: 0; engine prepares 1/1, runs 1/1
      U2 saw: h=[2; 3; 4]
    replay without its input:
      execution=refused numerical=not_established coverage=complete
      comparisons:
      failure: missing_common_input in phase planning
      runs 0/0 |}]
