(* [Pass.Outcome_counts] and [Pass.Audit_summary]: the counting vocabulary the
   whole-pipeline summary is built from.

   Three things are under test and they fail in different ways:

   - TOTALITY over [Map_verify.Outcome.t]. The counts must agree with
     [Map_verify.Tally] label for label, since both key on [Outcome.label] — if
     they ever diverge, one of the two summaries is lying about the same report.
   - the [of_bindings] GRAMMAR, which is what stops a decoder trusting the wire.
     Its sampled bound is [1 .. Int64.max_int] — a strict superset of BOTH
     backends' [int] domains — and is checked lexically, never converted, so
     this file runs under node as well as natively.
   - the OVERFLOW checks, which can only be reached by seeding a bucket at
     [Int64.max_int]. Nothing but [of_bindings] can build that, which is why a
     type crossing a wire needs a way back in. *)

let pp_bindings ppf t =
  Fmt.pf ppf "@[<v>%a@]"
    (Fmt.list (fun ppf (l, n) -> Fmt.pf ppf "  %-40s %Ld" l n))
    (Pass.Outcome_counts.bindings t)

let pp_accepted ppf t =
  Fmt.pf ppf "accepted %d" (List.length (Pass.Outcome_counts.bindings t))

let pp_count_error ppf (`Count_overflow c) = Pass.Count_overflow.pp ppf c

(* [count_error] and [invalid] are different rows, so they compose with
   different error printers — which is the point of them being different rows.
   Composed through [Core.Pretty.core_result] rather than matched by hand, per
   CLAUDE.md. *)
let pp_counts = Core.Pretty.core_result ~ok:pp_bindings ~error:pp_count_error

let pp_decoded =
  Core.Pretty.core_result ~ok:pp_bindings ~error:Pass.Outcome_counts.pp_invalid

let pp_accept =
  Core.Pretty.core_result ~ok:pp_accepted ~error:Pass.Outcome_counts.pp_invalid

let show pp label r = Format.printf "@[<v>%s:@,%a@]@." label pp r
let show_inline pp label r = Format.printf "%s: %a@." label pp r

(* --- totality over every outcome, and agreement with Tally --- *)

let sampled n =
  Core.or_raise
    (fun ppf (`Invalid_coverage n) -> Fmt.pf ppf "invalid coverage %d" n)
    (Map_verify.Coverage.sampled n)

let entry coverage verdict : Map_verify.Entry.t =
  {
    cluster =
      {
        src = Graph_ir.Tensor_id.Set.empty;
        dst = Graph_ir.Tensor_id.Set.empty;
        label = Correspondence.Identical;
      };
    group = [];
    outcome = { coverage; verdict };
  }

let member side : Map_verify.Member.Erased.t =
  { id = Graph_ir.Tensor_id.of_int 0; side }

let coord = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0

let cell : Ground_expr.Cell.t =
  { coord; origin = Ground_expr.Origin.Dst (Graph_ir.Tensor_id.of_int 0) }

(* One entry of EVERY outcome, so "total" is exercised rather than asserted. A
   sampled proof sits beside an exhaustive one deliberately: they must land in
   different buckets. *)
let full_report : Map_verify.Report.t =
  {
    entries =
      [
        entry Map_verify.Coverage.exhaustive
          (Map_verify.Verdict.Proved Map_verify.Strength.Structural);
        entry (sampled 7)
          (Map_verify.Verdict.Proved Map_verify.Strength.Structural);
        entry Map_verify.Coverage.exhaustive
          (Map_verify.Verdict.Proved Map_verify.Strength.Constants);
        entry Map_verify.Coverage.not_applicable
          (Map_verify.Verdict.Refuted
             (Map_verify.Refutation.Shape
                {
                  lhs = member `Src;
                  lhs_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1;
                  rhs = member `Dst;
                  rhs_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2;
                }));
        entry Map_verify.Coverage.exhaustive
          (Map_verify.Verdict.Refuted
             (Map_verify.Refutation.Value
                {
                  coord;
                  lhs = member `Src;
                  rhs = member `Dst;
                  valuation = Ground_expr.Valuation.empty;
                }));
        entry Map_verify.Coverage.exhaustive
          (Map_verify.Verdict.Tested (Map_verify.Strength.Agrees 1e-6));
        entry Map_verify.Coverage.exhaustive
          (Map_verify.Verdict.Tested
             (Map_verify.Strength.Disagrees Ground_expr.Valuation.empty));
        entry Map_verify.Coverage.not_applicable
          (Map_verify.Verdict.Unproved (Map_verify.Unproved.Too_large 1));
        entry Map_verify.Coverage.not_applicable
          (Map_verify.Verdict.Unproved Map_verify.Unproved.Max_rounds);
        entry Map_verify.Coverage.not_applicable
          (Map_verify.Verdict.Unproved (Map_verify.Unproved.Out_of_bounds cell));
        entry Map_verify.Coverage.not_applicable Map_verify.Verdict.Vacuous;
      ];
  }

let%expect_test
    "of_report is total over outcomes, and sampled gets its own bucket" =
  show pp_counts "counts" (Pass.Outcome_counts.of_report full_report);
  [%expect
    {|
    counts:
      proved (for these constants)             1
      proved (structural)                      1
      proved (structural) [sampled 7]          1
      refuted (shape)                          1
      refuted (counterexample)                 1
      tested (coefficients agree)              1
      tested (disagrees)                       1
      unproved (over max_rounds)               1
      unproved (out of bounds)                 1
      unproved (too large)                     1
      vacuous                                  1
    |}]

let%expect_test "Outcome_counts and Map_verify.Tally agree label for label" =
  (* Both key on [Outcome.label]. If this ever diverges, one of the two
     summaries is describing the same report differently — which is the drift
     factoring that function out was meant to make impossible. *)
  let tally =
    List.sort compare
      (List.map
         (fun (l, n) -> (l, Int64.of_int n))
         (Map_verify.Tally.bindings
            (Map_verify.Tally.of_entries full_report.entries)))
  in
  let pp_agreement ppf t =
    let counts = List.sort compare (Pass.Outcome_counts.bindings t) in
    Fmt.pf ppf "agree=%b entries=%d buckets=%d" (tally = counts)
      (List.length full_report.entries)
      (List.length counts)
  in
  show_inline
    (Core.Pretty.core_result ~ok:pp_agreement ~error:pp_count_error)
    "tally"
    (Pass.Outcome_counts.of_report full_report);
  [%expect {| tally: agree=true entries=11 buckets=11 |}]

(* --- the of_bindings grammar --- *)

let%expect_test "of_bindings canonicalizes, numerically in the sample count" =
  (* [sampled 9] before [sampled 10]: forbidding leading zeros is what makes
     (digit count, digit string) a numeric order without ever converting a
     digit run to an integer. *)
  show pp_decoded "canonical"
    (Pass.Outcome_counts.of_bindings
       [
         ("vacuous", 1L);
         ("proved (structural) [sampled 10]", 2L);
         ("proved (structural) [sampled 9]", 3L);
         ("proved (structural)", 4L);
         ("refuted (shape)", 5L);
       ]);
  [%expect
    {|
    canonical:
      proved (structural)                      4
      proved (structural) [sampled 9]          3
      proved (structural) [sampled 10]         2
      refuted (shape)                          5
      vacuous                                  1
    |}]

let%expect_test "bindings after of_bindings is the identity on canonical input"
    =
  let input =
    [
      ("proved (structural)", 4L);
      ("proved (structural) [sampled 9]", 3L);
      ("proved (structural) [sampled 10]", 2L);
      ("vacuous", 1L);
    ]
  in
  let pp_identity ppf t =
    Fmt.pf ppf "identity=%b" (Pass.Outcome_counts.bindings t = input)
  in
  show_inline
    (Core.Pretty.core_result ~ok:pp_identity
       ~error:Pass.Outcome_counts.pp_invalid)
    "canonical input"
    (Pass.Outcome_counts.of_bindings input);
  [%expect {| canonical input: identity=true |}]

let%expect_test "of_bindings: the sampled bound, lexically and on both backends"
    =
  (* The four values that separate a parser narrowing through [int] from one
     that does not. 2^31 and 2^31-1 straddle the jsoo [int] domain; the last two
     straddle [Int64.max_int], which is the stated bound and a strict superset
     of both backends. *)
  let ok label =
    show_inline pp_accept label
      (Pass.Outcome_counts.of_bindings [ (label, 1L) ])
  in
  ok "proved (structural) [sampled 1]";
  ok "proved (structural) [sampled 2147483647]";
  ok "proved (structural) [sampled 2147483648]";
  ok "proved (structural) [sampled 9223372036854775807]";
  ok "proved (structural) [sampled 9223372036854775808]";
  ok "proved (structural) [sampled 99999999999999999999]";
  [%expect
    {|
    proved (structural) [sampled 1]: accepted 1
    proved (structural) [sampled 2147483647]: accepted 1
    proved (structural) [sampled 2147483648]: accepted 1
    proved (structural) [sampled 9223372036854775807]: accepted 1
    proved (structural) [sampled 9223372036854775808]: invalid outcome counts: not an outcome label ("proved (structural) [sampled 9223372036854775808]")
    proved (structural) [sampled 99999999999999999999]: invalid outcome counts: not an outcome label ("proved (structural) [sampled 99999999999999999999]")
    |}]

let%expect_test "of_bindings: every rejection kind" =
  show_inline pp_accept "unknown base"
    (Pass.Outcome_counts.of_bindings [ ("proved (magic)", 1L) ]);
  show_inline pp_accept "sampled 0"
    (Pass.Outcome_counts.of_bindings
       [ ("proved (structural) [sampled 0]", 1L) ]);
  show_inline pp_accept "leading zero"
    (Pass.Outcome_counts.of_bindings
       [ ("proved (structural) [sampled 07]", 1L) ]);
  show_inline pp_accept "negative count"
    (Pass.Outcome_counts.of_bindings [ ("vacuous", -1L) ]);
  show_inline pp_accept "duplicate label"
    (Pass.Outcome_counts.of_bindings [ ("vacuous", 1L); ("vacuous", 2L) ]);
  show_inline pp_accept "too long"
    (Pass.Outcome_counts.of_bindings [ (String.make 200 'x', 1L) ]);
  [%expect
    {|
    unknown base: invalid outcome counts: not an outcome label ("proved (magic)")
    sampled 0: invalid outcome counts: not an outcome label ("proved (structural) [sampled 0]")
    leading zero: invalid outcome counts: not an outcome label ("proved (structural) [sampled 07]")
    negative count: invalid outcome counts: negative count ("vacuous")
    duplicate label: invalid outcome counts: duplicate label ("vacuous")
    too long: invalid outcome counts: label too long ("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    |}]

(* --- the wire --- *)

let%expect_test "jsont round-trips through of_bindings, int64 as a string" =
  let t =
    Core.or_raise Pass.Outcome_counts.pp_invalid
      (Pass.Outcome_counts.of_bindings
         [
           ("proved (structural)", 2147483648L);
           ("proved (structural) [sampled 9223372036854775807]", 1L);
           ("vacuous", 9007199254740993L);
         ])
  in
  (* Jsont's own results are [Stdlib.result], so they compose through
     [Core.Pretty.result]; bound together first so there is ONE printer rather
     than a nest of matches. *)
  let roundtrip =
    Result.bind
      (Jsont_bytesrw.encode_string ~format:Jsont.Indent
         Pass.Outcome_counts.jsont t) (fun json ->
        Result.map
          (fun back -> (json, back))
          (Jsont_bytesrw.decode_string Pass.Outcome_counts.jsont json))
  in
  let pp_ok ppf (json, back) =
    Fmt.pf ppf "%s@,roundtrip=%b" json
      (Pass.Outcome_counts.bindings back = Pass.Outcome_counts.bindings t)
  in
  Format.printf "@[<v>%a@]@."
    (Core.Pretty.result ~ok:pp_ok ~error:Fmt.string)
    roundtrip;
  [%expect
    {|
    [
      {
        "label": "proved (structural)",
        "count": "2147483648"
      },
      {
        "label": "proved (structural) [sampled 9223372036854775807]",
        "count": "1"
      },
      {
        "label": "vacuous",
        "count": "9007199254740993"
      }
    ]
    roundtrip=true
    |}]

let%expect_test "jsont decoding refuses a malformed binding" =
  (* The decoder goes through [of_bindings], so the wire cannot introduce a
     label the producer could not have emitted. [Jsont] has no typed error
     channel, so this surfaces as a message — which is exactly what the named
     bridge in the implementation is for. *)
  Format.printf "%a@."
    (Core.Pretty.result ~ok:pp_accepted ~error:Fmt.string)
    (Jsont_bytesrw.decode_string Pass.Outcome_counts.jsont
       {|[{"label":"proved (magic)","count":"1"}]|});
  [%expect
    {| invalid outcome counts: not an outcome label ("proved (magic)") |}]

(* --- the overflow checks, reachable only through of_bindings --- *)

let seeded label n =
  Core.or_raise Pass.Outcome_counts.pp_invalid
    (Pass.Outcome_counts.of_bindings [ (label, n) ])

let%expect_test "add overflows its own bucket, and names it" =
  let full = seeded "vacuous" Int64.max_int in
  show_inline pp_counts "add"
    (Pass.Outcome_counts.add full
       {
         coverage = Map_verify.Coverage.not_applicable;
         verdict = Map_verify.Verdict.Vacuous;
       });
  [%expect {| add: outcome bucket "vacuous" overflowed |}]

let%expect_test "merge overflows its own bucket, and names it" =
  let full = seeded "vacuous" Int64.max_int in
  let one = seeded "vacuous" 1L in
  show_inline pp_counts "merge" (Pass.Outcome_counts.merge full one);
  [%expect {| merge: outcome bucket "vacuous" overflowed |}]

let%expect_test "Audit_summary counts reports and clusters independently" =
  (* Two counters, two checks. [omitted_reports] overflowing is a DIFFERENT
     failure from a bucket overflowing, and the second is only reachable when
     the first does not fire — so a single "int64 boundary" case would have
     proved one and left the other untested. *)
  let pp_summary ppf (s : Pass.Audit_summary.t) =
    Fmt.pf ppf "omitted=%Ld buckets=%d" s.omitted_reports
      (List.length (Pass.Outcome_counts.bindings s.counts))
  in
  let pp = Core.Pretty.core_result ~ok:pp_summary ~error:pp_count_error in
  show_inline pp "first"
    (Pass.Audit_summary.add Pass.Audit_summary.empty full_report);
  let open Core.Syntax in
  show_inline pp "twice"
    (let* one = Pass.Audit_summary.add Pass.Audit_summary.empty full_report in
     Pass.Audit_summary.add one full_report);
  show_inline pp "reports at max"
    (Pass.Audit_summary.add
       {
         Pass.Audit_summary.omitted_reports = Int64.max_int;
         counts = Pass.Outcome_counts.empty;
       }
       full_report);
  show_inline pp "bucket at max"
    (Pass.Audit_summary.add
       {
         Pass.Audit_summary.omitted_reports = 0L;
         counts = seeded "vacuous" Int64.max_int;
       }
       full_report);
  [%expect
    {|
    first: omitted=1 buckets=11
    twice: omitted=2 buckets=11
    reports at max: audit reports overflowed
    bucket at max: outcome bucket "vacuous" overflowed
    |}]
