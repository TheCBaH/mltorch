(* The outcome-label vocabulary: [Verdict.label], [Verdict.labels],
   [Outcome.label] and the closed [Coverage] constructors.

   These exist so ONE vocabulary serves both summaries — [Map_verify.Tally] and
   [Pass.Outcome_counts] — and so a decoder rebuilding counts from wire labels
   can tell a well-formed label from an arbitrary string. Two properties carry
   that, and both are checked here rather than asserted:

     every constructor's label is in [labels]     (nothing escapes the set)
     every entry in [labels] is some label        (nothing in it is unreachable)

   The first is what breaks when a verdict is added and [labels] is not
   updated. The second is what breaks when a label is reworded. *)

open Map_verify

(* Payload samples. [label] drops every payload by contract, so these only have
   to EXIST — which is the point of listing all of them: the contract is what
   makes the label set finite, and a payload that could reach the label would
   show up here as a golden that varies. *)
let member side : Member.Erased.t = { id = Tensor_id.of_int 0; side }
let coord = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0

let cell : Ground_expr.Cell.t =
  { coord; origin = Ground_expr.Origin.Dst (Tensor_id.of_int 0) }

let every_verdict : Verdict.t list =
  [
    Proved Strength.Constants;
    Proved Strength.Structural;
    Refuted
      (Refutation.Shape
         {
           lhs = member `Src;
           lhs_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1;
           rhs = member `Dst;
           rhs_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2;
         });
    Refuted
      (Refutation.Value
         {
           coord;
           lhs = member `Src;
           rhs = member `Dst;
           valuation = Ground_expr.Valuation.empty;
         });
    Tested (Strength.Agrees 1e-6);
    Tested (Strength.Disagrees Ground_expr.Valuation.empty);
    Unproved (Unproved.Eval (`Unknown_edge (Tensor_id.of_int 0)));
    Unproved
      (Unproved.Exhausted { coord; lhs = member `Src; rhs = member `Dst });
    Unproved (Unproved.Max_ground_nodes 1L);
    Unproved (Unproved.Max_nodes 1);
    Unproved Unproved.Max_rounds;
    Unproved (Unproved.Max_clusters 1);
    Unproved (Unproved.Out_of_bounds cell);
    Unproved (Unproved.Too_large 1);
    Unproved (Unproved.Unbound_constant cell);
    Unproved
      (Unproved.Unsupported_format { blocked = cell; member = member `Dst });
    Unproved (Unproved.Unsupported_relation Correspondence.Unverifiable);
    Vacuous;
  ]

let%expect_test "every verdict's label is in Verdict.labels" =
  List.iter
    (fun v ->
      let l = Verdict.label v in
      if not (List.mem l Verdict.labels) then Printf.printf "ESCAPED: %s\n" l)
    every_verdict;
  Printf.printf "verdicts=%d labels=%d\n"
    (List.length every_verdict)
    (List.length Verdict.labels);
  [%expect {| verdicts=18 labels=18 |}]

let%expect_test "every entry in Verdict.labels is reachable" =
  let produced = List.map Verdict.label every_verdict in
  List.iter
    (fun l ->
      if not (List.mem l produced) then Printf.printf "UNREACHABLE: %s\n" l)
    Verdict.labels;
  print_endline "all reachable";
  [%expect {| all reachable |}]

let%expect_test "Verdict.labels is the canonical order, printed once" =
  List.iter print_endline Verdict.labels;
  [%expect
    {|
    proved (for these constants)
    proved (structural)
    refuted (shape)
    refuted (counterexample)
    tested (coefficients agree)
    tested (disagrees)
    unproved (grounding failed)
    unproved (frontier exhausted)
    unproved (over max_ground_nodes)
    unproved (over max_nodes)
    unproved (over max_rounds)
    unproved (global verification budget exhausted)
    unproved (out of bounds)
    unproved (too large)
    unproved (unbound constant)
    unproved (format blocks collapse)
    unproved (unsupported relation)
    vacuous
    |}]

(* --- Outcome.label: the sampled suffix, and only for [Sampled] --- *)

let outcome coverage verdict : Outcome.t = { coverage; verdict }

let%expect_test "Outcome.label suffixes only a sampled coverage" =
  let v = Verdict.Proved Strength.Structural in
  let sampled n =
    Err.or_raise
      ~pp_error:(fun ppf (`Invalid_coverage n) ->
        Fmt.pf ppf "invalid coverage %d" n)
      (Coverage.sampled n)
  in
  print_endline (Outcome.label (outcome Coverage.exhaustive v));
  print_endline (Outcome.label (outcome Coverage.not_applicable v));
  print_endline (Outcome.label (outcome (sampled 1) v));
  print_endline (Outcome.label (outcome (sampled 4096) v));
  [%expect
    {|
    proved (structural)
    proved (structural)
    proved (structural) [sampled 1]
    proved (structural) [sampled 4096]
    |}]

let%expect_test "Outcome.label is the key Tally counts by" =
  (* The property the factoring exists for: [Tally.of_entries] must bucket by
     exactly [Outcome.label], so a second summary built on that function agrees
     with this one by construction rather than by having copied the format. *)
  let entry coverage verdict : Entry.t =
    {
      cluster =
        {
          src = Tensor_id.Set.empty;
          dst = Tensor_id.Set.empty;
          label = Correspondence.Identical;
        };
      group = [];
      outcome = outcome coverage verdict;
    }
  in
  let sampled n =
    Err.or_raise
      ~pp_error:(fun ppf (`Invalid_coverage n) ->
        Fmt.pf ppf "invalid coverage %d" n)
      (Coverage.sampled n)
  in
  let entries =
    [
      entry Coverage.exhaustive (Verdict.Proved Strength.Structural);
      entry Coverage.exhaustive (Verdict.Proved Strength.Structural);
      entry (sampled 8) (Verdict.Proved Strength.Structural);
      entry Coverage.not_applicable Verdict.Vacuous;
    ]
  in
  List.iter
    (fun (label, n) -> Printf.printf "%-40s %d\n" label n)
    (Tally.bindings (Tally.of_entries entries));
  [%expect
    {|
    proved (structural)                      2
    proved (structural) [sampled 8]          1
    vacuous                                  1
    |}]

(* --- the closed Coverage constructors --- *)

let%expect_test "Coverage.sampled rejects a count no producer can emit" =
  (* [sampled_coords] returns [max 1 (min n numel)], so this rejects nothing the
     verifier produces. What it closes is the PUBLIC domain: [Entry.t] and
     [Report.t] are public records, so before this any caller could put
     [Sampled 0] into one and reach a decoder that refuses the label it
     generates.

     That [Coverage.Sampled 0] is now unwritable is checked by the compiler, not
     here — a test containing that expression would not build, which is why the
     .mli is the evidence and this file carries the constructor's behaviour. *)
  let show n =
    match Coverage.sampled n with
    | Ok c -> Format.printf "%d -> %a@." n Coverage.pp c
    | Error e ->
        let (`Invalid_coverage bad) = Err.Error.kind e in
        Printf.printf "%d -> rejected (%d)\n" n bad
  in
  show 1;
  show 4096;
  show 0;
  show (-1);
  [%expect
    {|
    1 -> sampled 1
    4096 -> sampled 4096
    0 -> rejected (0)
    -1 -> rejected (-1)
    |}]
