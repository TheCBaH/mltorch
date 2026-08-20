(* [Infer_report] -- the ranking comparison, the strict verdict, the loop that
   drives them, and the command line.

   Everything here is synthetic. That is deliberate: the release fixtures agree
   with their own results.json by construction, so the mismatch path -- the one
   `make jsoo.pt2.run --strict` exists to police -- is unreachable from a real
   model run. *)

open Infer_report

(* Class index -> printed label. Deliberately not synset-shaped: these goldens
   are about ordering and verdicts, not about the ImageNet label files. *)
let label idx = Printf.sprintf "c%d" idx

let pred rank class_index probability =
  { Prediction.rank; class_index; label = label class_index; probability }

(* A reference lookup over an association list, standing in for the decoded
   results.json map. *)
let reference_of entries sample = List.assoc_opt sample entries

let pp_failure ppf e =
  Fmt.pf ppf "@[<h>error: %a@]" (pp_error (Fmt.any "<eval>")) e

let show result =
  Fmt.pr "%a@."
    (Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:pp_failure)
    result

(* --- compare_ranking --- *)

let%expect_test "compare_ranking: equal sequences match" =
  let local = [ (3, 0.5); (1, 0.3) ] in
  Fmt.pr "%a@."
    (Fmt.option ~none:(Fmt.any "match") (fun ppf m ->
         Fmt.pf ppf "mismatch %s" m.Mismatch.sample))
    (compare_ranking ~sample:"images/a.pt" ~local ~reference:[ 3; 1 ]);
  [%expect {| match |}]

(* Three ways to disagree: a different index, a different order, and a different
   length. The length case matters because the reference block is printed from
   the reference list, not from the local one. *)
let%expect_test "compare_ranking: every kind of disagreement" =
  let local = [ (3, 0.5); (1, 0.3) ] in
  List.iter
    (fun reference ->
      Fmt.pr "%a@."
        (Fmt.option ~none:(Fmt.any "match") (fun ppf m ->
             Fmt.pf ppf "local [%a] reference [%a]"
               Fmt.(list ~sep:(any ";") int)
               m.Mismatch.local
               Fmt.(list ~sep:(any ";") int)
               m.Mismatch.reference))
        (compare_ranking ~sample:"images/a.pt" ~local ~reference))
    [ [ 3; 2 ]; [ 1; 3 ]; [ 3 ]; [ 3; 1; 7 ]; [] ];
  [%expect
    {|
    local [3;1] reference [3;2]
    local [3;1] reference [1;3]
    local [3;1] reference [3]
    local [3;1] reference [3;1;7]
    local [3;1] reference [] |}]

(* --- strict_verdict --- *)

let%expect_test "strict_verdict: first in sample order" =
  let m sample = { Mismatch.sample; local = [ 1 ]; reference = [ 2 ] } in
  List.iter
    (fun ms ->
      Fmt.pr "%a@."
        (Fmt.result ~ok:(Fmt.any "ok") ~error:(fun ppf m ->
             Fmt.pf ppf "%s" m.Mismatch.sample))
        (strict_verdict ms))
    [ []; [ m "images/b.pt" ]; [ m "images/b.pt"; m "images/c.pt" ] ];
  [%expect {|
    ok
    images/b.pt
    images/b.pt |}]

(* --- report --- *)

let options ?(mode = Natural) ?(strict = false) () = { Options.mode; strict }

(* Two samples whose local ranking is fed in directly, so the loop is the only
   thing under test. *)
let two_samples ~local_b =
  let entries =
    [
      ("images/a.pt", [ pred 1 7 0.9; pred 2 4 0.1 ]);
      ("images/b.pt", [ pred 1 5 0.8; pred 2 2 0.2 ]);
    ]
  in
  let local = function
    | "images/a.pt" -> [ (7, 0.9); (4, 0.1) ]
    | _ -> local_b
  in
  ( reference_of entries,
    (fun sample -> Err.return (local sample)),
    [ "images/a.pt"; "images/b.pt" ] )

let%expect_test "report: everything matches" =
  let reference, infer, samples = two_samples ~local_b:[ (5, 0.8); (2, 0.2) ] in
  List.iter
    (fun strict ->
      show (report ~label ~reference ~samples ~infer (options ~strict ())))
    [ false; true ];
  [%expect
    {|
    === images/a.pt ===
    1: c7 (0.9, ref 0.9)
    2: c4 (0.1, ref 0.1)
    === images/b.pt ===
    1: c5 (0.8, ref 0.8)
    2: c2 (0.2, ref 0.2)
    ok
    === images/a.pt ===
    1: c7 (0.9, ref 0.9)
    2: c4 (0.1, ref 0.1)
    === images/b.pt ===
    1: c5 (0.8, ref 0.8)
    2: c2 (0.2, ref 0.2)
    ok |}]

(* Cram mode drops the probabilities from a matching entry entirely. *)
let%expect_test "report: cram mode drops probabilities on a match" =
  let reference, infer, samples = two_samples ~local_b:[ (5, 0.8); (2, 0.2) ] in
  show (report ~label ~reference ~samples ~infer (options ~mode:Cram ()));
  [%expect
    {|
    === images/a.pt ===
    1: c7
    2: c4
    === images/b.pt ===
    1: c5
    2: c2
    ok |}]

(* The default: a mismatch is REPORTED but does not fail the run. This is what
   the four interp_*_cram.t goldens depend on. *)
let%expect_test "report: a mismatch is exit-0 without --strict" =
  let reference, infer, samples = two_samples ~local_b:[ (9, 0.7); (2, 0.3) ] in
  show (report ~label ~reference ~samples ~infer (options ()));
  [%expect
    {|
    === images/a.pt ===
    1: c7 (0.9, ref 0.9)
    2: c4 (0.1, ref 0.1)
    === images/b.pt ===
    local:
    1: c9 (0.7)
    2: c2 (0.3)
    reference:
    1: c5 (0.8)
    2: c2 (0.2)
    ok |}]

(* Same input under --strict: the SAME report is printed in full, and only then
   does the run fail. A verdict that truncated the report would be much less
   useful, so the ordering is part of the contract. *)
let%expect_test "report: --strict fails after reporting every sample" =
  let reference, infer, samples = two_samples ~local_b:[ (9, 0.7); (2, 0.3) ] in
  show (report ~label ~reference ~samples ~infer (options ~strict:true ()));
  [%expect
    {|
    === images/a.pt ===
    1: c7 (0.9, ref 0.9)
    2: c4 (0.1, ref 0.1)
    === images/b.pt ===
    local:
    1: c9 (0.7)
    2: c2 (0.3)
    reference:
    1: c5 (0.8)
    2: c2 (0.2)
    error: images/b.pt: top-2 [9, 2] does not match reference [5, 2] |}]

(* Two mismatches: the verdict names the FIRST in sample order, not the last
   one seen. *)
let%expect_test "report: --strict reports the first mismatch" =
  let entries =
    [
      ("images/a.pt", [ pred 1 7 0.9 ]);
      ("images/b.pt", [ pred 1 5 0.8 ]);
      ("images/c.pt", [ pred 1 3 0.6 ]);
    ]
  in
  show
    (report ~label ~reference:(reference_of entries)
       ~samples:[ "images/a.pt"; "images/b.pt"; "images/c.pt" ]
       ~infer:(fun _ -> Err.return [ (1, 1.0) ])
       (options ~mode:Cram ~strict:true ()));
  [%expect
    {|
    === images/a.pt ===
    local:
    1: c1 (1)
    reference:
    1: c7 (0.9)
    === images/b.pt ===
    local:
    1: c1 (1)
    reference:
    1: c5 (0.8)
    === images/c.pt ===
    local:
    1: c1 (1)
    reference:
    1: c3 (0.6)
    error: images/a.pt: top-1 [1] does not match reference [7] |}]

let%expect_test "report: a sample with no reference" =
  show
    (report ~label
       ~reference:(fun _ -> None)
       ~samples:[ "images/a.pt" ]
       ~infer:(fun _ -> Err.return [ (1, 1.0) ])
       (options ()));
  [%expect {| error: no reference for "images/a.pt" |}]

(* Both kinds of callback failure propagate UNCHANGED. [report] must not
   reclassify: its caller's closure loads the .pt file too, so a blanket [`Eval]
   here would report a corrupt archive member as an evaluator fault. The
   [`Missing_weight] case is a bare Pt2_archive.error row and must still print
   as one. *)
let%expect_test "report: callback errors are propagated, not reclassified" =
  let at failure =
    show
      (report ~label
         ~reference:(fun _ -> Some [ pred 1 1 1.0 ])
         ~samples:[ "images/a.pt" ]
         ~infer:(fun _ -> Err.fail failure)
         (options ()))
  in
  at (`Eval ());
  at (`Missing_weight "conv.weight");
  at (`Results_decode "unexpected end of input");
  [%expect
    {|
    error: <eval>
    error: no weight named "conv.weight"
    error: results.json: unexpected end of input |}]

(* --- parse_argv --- *)

let paths = [| "prog"; "m.pt2"; "inputs.pt"; "expected.json"; "outputs.pt" |]

let show_argv argv =
  Fmt.pr "%a@."
    (Fmt.result
       ~ok:(fun ppf ((p : Paths.t), (o : Options.t)) ->
         Fmt.pf ppf "%s %s %s %s%s%s" p.pt2
           (Option.value p.inputs ~default:"?")
           (Option.value p.expected ~default:"?")
           (Option.value p.outputs ~default:"?")
           (match o.mode with Natural -> "" | Cram -> " cram")
           (if o.strict then " strict" else ""))
       ~error:(fun ppf _ -> Fmt.pf ppf "usage"))
    (parse_argv argv)

let%expect_test "parse_argv: tensor-map paths and the flag grammar" =
  List.iter
    (fun extra -> show_argv (Array.append paths extra))
    [
      [||];
      [| "--cram" |];
      [| "--strict" |];
      [| "--cram"; "--strict" |];
      [| "--strict"; "--cram" |];
      [| "--cram"; "--cram" |];
    ];
  [%expect
    {|
    m.pt2 inputs.pt expected.json outputs.pt
    m.pt2 inputs.pt expected.json outputs.pt cram
    m.pt2 inputs.pt expected.json outputs.pt strict
    m.pt2 inputs.pt expected.json outputs.pt cram strict
    m.pt2 inputs.pt expected.json outputs.pt cram strict
    m.pt2 inputs.pt expected.json outputs.pt cram |}]

(* An extra positional is a distinct branch from a missing one, and the one that
   would otherwise be silently dropped -- which is how a caller ends up thinking
   it passed --strict when it did not. *)
let%expect_test "parse_argv: rejections" =
  List.iter show_argv
    [
      [||];
      [| "prog" |];
      [| "prog"; "m.pt2"; "inputs.pt"; "expected.json" |];
      Array.append paths [| "extra.json" |];
      Array.append paths [| "--cram"; "extra.json" |];
      Array.append paths [| "--verbose" |];
      [|
        "prog"; "m.pt2"; "--cram"; "inputs.pt"; "expected.json"; "outputs.pt";
      |];
    ];
  [%expect
    {|
    usage
    usage
    usage
    usage
    usage
    usage
    usage |}]
