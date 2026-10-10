open Transformers_metadata.Json_util
module R = Transformers_reports
module J = Jsont.Json

let get result =
  Err.or_raise ~pp_error:Transformers_metadata.Fault.pp_error result

let cohort_bytes = Pt2_fixture_test.World.cohort_json ()
let cohort = get (Pt2_fixture.Cohort.of_string cohort_bytes)
let entry = List.hd cohort.entries

let row =
  obj
    [
      ("artifact_id", J.string entry.artifact_id);
      ("gate", J.string "core");
      ("reason", J.string "published-case gate");
      ("policy", R.Run.Policy.json (List.hd R.Run.Policy.all));
    ]

let selection =
  obj
    [
      ("schema_version", J.int 1);
      ( "cohort_sha256",
        J.string (Transformers_metadata.Source.hash cohort_bytes) );
      ("requirements", J.list [ row ]);
    ]

let set key value json =
  obj ((key, value) :: List.remove_assoc key (get (members json)))

let%expect_test
    "policy choices bind the complete exact cohort and explicit route" =
  let accepted = get (R.Policy_selection.of_json ~cohort_bytes selection) in
  Fmt.pr "required=%d@." (List.length (R.Policy_selection.core accepted));
  List.iter
    (fun (label, json) ->
      Fmt.pr "%s refused=%b@." label
        (Result.is_error
           (Err.payload (R.Policy_selection.of_json ~cohort_bytes json))))
    [
      ( "stale cohort",
        set "cohort_sha256" (J.string (String.make 64 '0')) selection );
      ("duplicate", set "requirements" (J.list [ row; row ]) selection);
      ("partial", set "requirements" (J.list []) selection);
      ( "unknown policy",
        set "requirements"
          (J.list
             [
               set "policy"
                 (set "dots" (J.string "fast") (get (member "policy" row)))
                 row;
             ])
          selection );
    ];
  [%expect
    {|
    required=1
    stale cohort refused=true
    duplicate refused=true
    partial refused=true
    unknown policy refused=true |}]
