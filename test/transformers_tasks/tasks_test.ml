open Transformers_metadata.Json_util
open World
open Pt2_fixture_unix_test.Support
module J = Jsont.Json

let rec update keys f value =
  match keys with
  | [] -> f value
  | key :: rest ->
      let old = get (member key value) in
      set key (update rest f old) value

let change key value = update key (fun _ -> value)
let each f value = J.list (List.map f (get (array value)))

let%expect_test
    "pinned task extraction, every tensor and all original diagnostic cases \
     reopen offline" =
  with_dir (fun dir ->
      let fixture = build () in
      let calls = ref [] in
      let config = online fixture dir calls in
      let bundle = get (T.Reference.ensure config cohort fixture.request) in
      List.iter
        (fun row -> get (T.Tensors.verify_case bundle row))
        (get (member "cases" bundle.manifest >>= array));
      let report = get (T.Diagnostic.run bundle) in
      let cases = get (member "cases" report >>= array) in
      let comparisons = List.concat_map (fun case -> get (array case)) cases in
      let outputs =
        List.concat_map
          (fun pair ->
            get (path [ "report"; "cases" ] pair >>= array)
            |> List.concat_map (fun case ->
                get (member "outputs" case >>= array)))
          comparisons
      in
      Fmt.pr "downloads=%d cases=%d comparisons=%d outputs=%d@."
        (List.length !calls) (List.length cases) (List.length comparisons)
        (List.length outputs);
      let offline = U.Bundle.config config.cache in
      let reopened = get (T.Reference.ensure offline cohort fixture.request) in
      let repeated = get (T.Diagnostic.run reopened) in
      Fmt.pr "offline identical=%b acceptance=%s@."
        (bytes report = bytes repeated)
        (get (field "consumer_acceptance" report)));
  [%expect
    {|
    downloads=6 cases=2 comparisons=6 outputs=12
    offline identical=true acceptance=unchanged; component replay decides |}]

let%expect_test
    "provenance, schema, case and recipe byte mismatches fail before comparison"
    =
  let mutations =
    [
      ("schema", change [ "schema_version" ] (J.int 2));
      ( "generator",
        change [ "generator"; "commit" ] (J.string (String.make 40 'b')) );
      ( "lock",
        change [ "generator"; "lock_sha256" ] (J.string (String.make 64 '3')) );
      ( "environment",
        change [ "generator"; "versions"; "transformers" ] (J.string "wrong") );
      ( "consumer claim",
        change [ "generator"; "consumer_execution" ] (J.string "passed") );
      ( "recipe asset",
        change
          [ "recipe"; "assets"; "config.json"; "sha256" ]
          (J.string (String.make 64 '0')) );
      ( "reference graph",
        update [ "references" ]
          (each (change [ "graph"; "sha256" ] (J.string (String.make 64 '0'))))
      );
      ( "case coverage",
        change [ "expected_cases" ] (J.list [ J.string "reference-00-case-00" ])
      );
    ]
  in
  List.iter
    (fun (label, mutate) ->
      with_dir (fun dir ->
          let fixture = build ~mutate () in
          let result =
            T.Reference.ensure
              (online fixture dir (ref []))
              cohort fixture.request
          in
          Printf.printf "%s refused=%b\n" label
            (Result.is_error (Err.payload result))))
    mutations;
  [%expect
    {|
    schema refused=true
    generator refused=true
    lock refused=true
    environment refused=true
    consumer claim refused=true
    recipe asset refused=true
    reference graph refused=true
    case coverage refused=true |}]

let%expect_test
    "tensor bytes, shape, dtype, missing K/V and dishonest comparison counts \
     are not trusted" =
  let case00 f =
    update [ "cases" ] (fun rows ->
        match get (array rows) with
        | first :: rest -> J.list (f first :: rest)
        | [] -> assert false)
  in
  let mutations =
    [
      ( "tensor digest",
        case00
          (update
             [ "files"; "cases/reference-00-case-00/eager.pt"; "tensors" ]
             (each (change [ "sha256" ] (J.string (String.make 64 '0'))))) );
      ( "shape",
        case00
          (update
             [ "files"; "cases/reference-00-case-00/eager.pt"; "tensors" ]
             (each (change [ "shape" ] (J.list [ J.int 3; J.int 2 ])))) );
      ( "dtype",
        case00
          (update
             [ "files"; "cases/reference-00-case-00/eager.pt"; "tensors" ]
             (each (change [ "dtype" ] (J.string "int64")))) );
      ( "missing output",
        case00
          (update [ "files"; "cases/reference-00-case-00/eager.pt"; "tensors" ]
             (fun rows -> J.list [ List.hd (get (array rows)) ])) );
      ( "claimed count",
        case00
          (update
             [ "comparisons"; "eager_vs_published"; "outputs" ]
             (each (change [ "over_tolerance" ] (J.int 1)))) );
      ( "claimed maximum",
        case00
          (update
             [ "comparisons"; "eager_vs_published"; "outputs" ]
             (each (change [ "max_absolute_error" ] (J.number 1.)))) );
      ( "claimed equality",
        case00
          (update
             [ "comparisons"; "eager_vs_published"; "outputs" ]
             (each (change [ "bitwise" ] (J.bool false)))) );
      ( "wrong original case",
        case00 (change [ "artifact_id" ] (J.string "wrong")) );
    ]
  in
  List.iter
    (fun (label, mutate) ->
      with_dir (fun dir ->
          let fixture = build ~mutate () in
          let bundle =
            get
              (T.Reference.ensure
                 (online fixture dir (ref []))
                 cohort fixture.request)
          in
          let result = T.Diagnostic.run bundle in
          Printf.printf "%s refused=%b\n" label
            (Result.is_error (Err.payload result))))
    mutations;
  [%expect
    {|
    tensor digest refused=true
    shape refused=true
    dtype refused=true
    missing output refused=true
    claimed count refused=true
    claimed maximum refused=true
    claimed equality refused=true
    wrong original case refused=true |}]

let%expect_test "offline corruption and links fail exact inventory checks" =
  with_dir (fun dir ->
      let fixture = build () in
      let config = online fixture dir (ref []) in
      let bundle = get (T.Reference.ensure config cohort fixture.request) in
      let offline = U.Bundle.config config.cache in
      let path = Filename.concat bundle.dir "environment.json" in
      write path "corrupt";
      let corrupt = T.Reference.ensure offline cohort fixture.request in
      Printf.printf "corrupt refused=%b\n"
        (Result.is_error (Err.payload corrupt));
      write path (bytes generator);
      let outside = Filename.concat dir "outside.json" in
      Unix.rename path outside;
      Unix.symlink outside path;
      let linked = T.Reference.ensure offline cohort fixture.request in
      Printf.printf "linked refused=%b\n" (Result.is_error (Err.payload linked)));
  [%expect {|
    corrupt refused=true
    linked refused=true |}]

let%expect_test
    "signed-zero equality is a producer value claim, raw byte identity stays \
     separate" =
  let logical values =
    F.Logical.of_bytes ~dtype:Pt2_checkpoint_map.Dtype.F32 ~shape:[ 1L ]
      (R.f32s values)
    |> Err.map_error (fun e -> `Logical_tensor ("fixture", e))
    |> get
  in
  let positive = logical [ 0. ] in
  let negative = logical [ -0. ] in
  let equal = get (T.Diagnostic.equal_values positive negative) in
  Fmt.pr "value equal=%b bytes equal=%b@." equal
    (Pt2_sha256.Digest.equal
       (Pt2_sha256.bigstring positive.data)
       (Pt2_sha256.bigstring negative.data));
  [%expect {| value equal=true bytes equal=false |}]

let%expect_test "a verified producer mismatch remains a failed comparison" =
  let logical value =
    F.Logical.of_bytes ~dtype:Pt2_checkpoint_map.Dtype.F32 ~shape:[ 1L ]
      (R.f32s [ value ])
    |> Err.map_error (fun e -> `Logical_tensor ("fixture", e))
    |> get
  in
  let comparisons =
    get
      (T.Diagnostic.compare ~atol:0. ~rtol:0.
         [ ("logits", logical 1.) ]
         [ ("logits", logical 2.) ])
  in
  let claim =
    obj
      [
        ( "pair",
          obj
            [
              ("status", J.string "mismatch");
              ( "outputs",
                J.list
                  [
                    obj
                      [
                        ("name", J.string "logits");
                        ("elements", J.int 1);
                        ("over_tolerance", J.int 1);
                        ("max_absolute_error", J.number 1.);
                        ("bitwise", J.bool false);
                      ];
                  ] );
            ] );
      ]
  in
  get (T.Diagnostic.declared_comparison ~name:"pair" claim comparisons);
  Fmt.pr "claim verified; comparison passed=%b@."
    (List.for_all F.Compare.passed comparisons);
  [%expect {| claim verified; comparison passed=false |}]

let%expect_test
    "task runs retain reports and refuse changing execution identity" =
  with_dir (fun dir ->
      let context = obj [ ("source", J.string "before") ] in
      let run = get (T.Runtime.create dir context) in
      let report =
        obj [ ("status", J.string "failed"); ("error", J.string "offline") ]
      in
      get (T.Runtime.add run "fixture" report);
      let collision =
        try
          ignore (T.Runtime.add run "fixture" report);
          false
        with Unix.Unix_error (Unix.EEXIST, _, _) -> true
      in
      let result =
        T.Runtime.finish run (obj [ ("source", J.string "after") ])
      in
      let completion =
        get
          (Transformers_metadata.Json_util.read
             (Filename.concat run.dir "completion.json"))
      in
      Fmt.pr "exclusive=%b identity refused=%b unchanged=%b reports=%d@."
        collision
        (Result.is_error (Err.payload result))
        (get (member "context_unchanged" completion >>= bool))
        (List.length (get (member "reports" completion >>= array)));
      Unix.chmod run.dir 0o755);
  [%expect {| exclusive=true identity refused=true unchanged=false reports=1 |}]
