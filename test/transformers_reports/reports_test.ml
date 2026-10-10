open Transformers_metadata.Json_util
open Pt2_fixture_unix_test.Support
module R = Transformers_reports
module Release = Pt2_fixture_replay_test.Release
module J = Jsont.Json

let get result =
  match Err.payload result with
  | Ok value -> value
  | Error _ -> failwith "fixture operation failed"

let identity =
  obj
    [
      ("consumer", obj [ ("commit", J.string (String.make 40 'a')) ]);
      ("workspace_sha256", J.string "synthetic-fixture");
    ]

let rec replace keys value json =
  match keys with
  | [] -> value
  | key :: rest ->
      let rows = get (members json) in
      obj
        (List.map
           (fun (name, old) ->
             (name, if name = key then replace rest value old else old))
           rows)

let change_cases f json =
  replace [ "cases" ] (J.list (f (get (member "cases" json >>= array)))) json

let rec writable dir =
  Unix.chmod dir 0o755;
  Array.iter
    (fun name ->
      let path = Filename.concat dir name in
      if Sys.is_directory path then writable path else Unix.chmod path 0o644)
    (Sys.readdir dir)

let fixture f =
  with_dir (fun dir ->
      let rel = Release.build () in
      let transport : U.Transport.t =
       fun ~url ~dest ->
        match List.assoc_opt url rel.served with
        | Some bytes ->
            write dest bytes;
            Ok ()
        | None -> Error "404"
      in
      let config =
        U.Bundle.config ~transport (cache (Filename.concat dir "cache"))
      in
      let entry = List.hd rel.cohort.entries in
      let expected = get (R.Expectation.load config rel.cohort entry) in
      let opened = get (U.Fixture.open_ config rel.cohort entry) in
      let replay policy =
        let dots =
          if policy.R.Run.Policy.dots = "exact" then Direct.Binary64
          else Direct.Binary32_sequential
        in
        let casts =
          if policy.casts = "checked" then Direct.Checked else Direct.Saturating
        in
        let report =
          get
            (Pt2_fixture_replay.replay ~dots ~casts
               ~consumer:(String.make 40 'a') opened)
        in
        get (parse (Pt2_fixture.Report.to_string report))
      in
      let root = Filename.concat dir "runs" in
      let create ?(identity = identity) ?(expected = expected) policy report =
        let run =
          get (R.Run.create ~root ~identity ~policy ~expected:[ expected ])
        in
        get (R.Run.add run report);
        run
      in
      let finish run = get (R.Run.finish run ~identity) in
      let matrix () =
        R.Matrix.generate ~identity
          ~expected:[ (entry.artifact_id, expected) ]
          [ root ]
      in
      Fun.protect
        ~finally:(fun () -> if Sys.file_exists root then writable root)
        (fun () -> f ~expected ~replay ~create ~finish ~matrix ~root))

let summary label matrix =
  let rows = get (member "rows" matrix >>= array) in
  let statuses = List.map (fun row -> get (field "status" row)) rows in
  let historical = get (member "historical" matrix >>= array) in
  let invalid = get (member "invalid" matrix >>= array) in
  Fmt.pr "%s: passed=%b [%s], historical=%d invalid=%d@." label
    (get (member "passed" matrix >>= bool))
    (String.concat ", " statuses)
    (List.length historical) (List.length invalid)

let%expect_test
    "complete independent policies pass, duplicates conflict instead of \
     replacing a failure" =
  fixture (fun ~expected:_ ~replay ~create ~finish ~matrix ~root:_ ->
      List.iter
        (fun policy -> finish (create policy (replay policy)))
        R.Run.Policy.all;
      summary "complete" (matrix ());
      let policy = List.hd R.Run.Policy.all in
      let failed =
        replay policy
        |> change_cases
             (List.map (fun row ->
                  replace [ "passed" ] (J.bool false) row
                  |> replace [ "error" ]
                       (J.string "graph refused before outputs")
                  |> replace [ "inputs_digest_verified" ] (J.bool false)
                  |> replace [ "outputs" ] (J.list [])))
        |> replace [ "status" ] (J.string "failed")
      in
      finish (create policy failed);
      let result = matrix () in
      summary "duplicate" result;
      Fmt.pr "error retained=%b, digest retained=%b@."
        (List.exists
           (fun row ->
             let runs = get (member "runs" row >>= array) in
             List.exists
               (fun run ->
                 List.mem "case-00: graph refused before outputs"
                   (get (member "details" run >>= R.Validate.names)))
               runs)
           (get (member "rows" result >>= array)))
        (List.exists
           (fun row ->
             get (field "status" row) = "conflict"
             &&
             let runs = get (member "runs" row >>= array) in
             List.exists
               (fun run ->
                 List.mem "case-00: input digest failed"
                   (get (member "details" run >>= R.Validate.names)))
               runs)
           (get (member "rows" result >>= array))));
  [%expect
    {|
    complete: passed=true [passed, passed, passed, passed], historical=0 invalid=0
    duplicate: passed=false [conflict, passed, passed, passed], historical=0 invalid=0
    error retained=true, digest retained=true |}]

let%expect_test
    "partial, stale, tolerant, policy and output mutations cannot pass" =
  let mutations =
    [
      ("partial cases", fun j -> change_cases (fun rows -> [ List.hd rows ]) j);
      ( "stale graph",
        replace [ "pins"; "graph" ] (J.string (String.make 64 '0')) );
      ("relaxed tolerance", replace [ "tolerances"; "atol" ] (J.number 1.));
      ("wrong backend", replace [ "backend" ] (J.string "kernel"));
      ( "partial outputs",
        change_cases
          (List.map (fun row ->
               let outputs = get (member "outputs" row >>= array) in
               replace [ "outputs" ] (J.list [ List.hd outputs ]) row)) );
      ( "bad coverage",
        change_cases
          (List.map (fun row ->
               let outputs = get (member "outputs" row >>= array) in
               replace [ "outputs" ]
                 (J.list
                    (List.map (replace [ "elements" ] (J.string "0")) outputs))
                 row)) );
      ( "dishonest digest",
        change_cases
          (List.map (replace [ "inputs_digest_verified" ] (J.bool false))) );
    ]
  in
  List.iter
    (fun (label, mutate) ->
      fixture (fun ~expected:_ ~replay ~create ~finish ~matrix ~root:_ ->
          let policy = List.hd R.Run.Policy.all in
          List.iter
            (fun p -> finish (create p (replay p)))
            (List.tl R.Run.Policy.all);
          finish (create policy (mutate (replay policy)));
          summary label (matrix ())))
    mutations;
  [%expect
    {|
    partial cases: passed=false [not run, passed, passed, passed], historical=0 invalid=1
    stale graph: passed=false [not run, passed, passed, passed], historical=0 invalid=1
    relaxed tolerance: passed=false [not run, passed, passed, passed], historical=0 invalid=1
    wrong backend: passed=false [not run, passed, passed, passed], historical=0 invalid=1
    partial outputs: passed=false [not run, passed, passed, passed], historical=0 invalid=1
    bad coverage: passed=false [not run, passed, passed, passed], historical=0 invalid=1
    dishonest digest: passed=false [not run, passed, passed, passed], historical=0 invalid=1 |}]

let%expect_test "mixed consumers and legacy passes remain historical" =
  fixture (fun ~expected:_ ~replay ~create ~finish:_ ~matrix ~root ->
      let policy = List.hd R.Run.Policy.all in
      let old_identity =
        replace [ "consumer"; "commit" ]
          (J.string (String.make 40 'b'))
          identity
      in
      let report =
        replace [ "consumer" ] (J.string (String.make 40 'b')) (replay policy)
      in
      let run = create ~identity:old_identity policy report in
      get (R.Run.finish run ~identity:old_identity);
      write (Filename.concat root "old.replay.json") (get (text report));
      summary "mixed" (matrix ()));
  [%expect
    {|
    mixed: passed=false [not run, not run, not run, not run], historical=2 invalid=0 |}]

let%expect_test
    "self-consistent stale run pins and unknown requested cases are rejected" =
  fixture (fun ~expected ~replay ~create ~finish ~matrix ~root:_ ->
      let policy = List.hd R.Run.Policy.all in
      let sha = J.string (String.make 64 '0') in
      let stale = replace [ "graph_sha256" ] sha expected in
      let report = replace [ "pins"; "graph" ] sha (replay policy) in
      finish (create ~expected:stale policy report);
      summary "stale manifest" (matrix ()));
  fixture (fun ~expected ~replay ~create ~finish ~matrix ~root:_ ->
      let policy = List.hd R.Run.Policy.all in
      let partial =
        replace [ "cases" ] (R.Identity.strings [ "case-00" ]) expected
      in
      let report =
        change_cases (fun rows -> [ List.hd rows ]) (replay policy)
      in
      finish (create ~expected:partial policy report);
      summary "partial manifest" (matrix ()));
  [%expect
    {|
    stale manifest: passed=false [not run, not run, not run, not run], historical=0 invalid=1
    partial manifest: passed=false [not run, not run, not run, not run], historical=0 invalid=1 |}]

let%expect_test
    "incomplete and corrupted files fail; immutable writers refuse overwrite" =
  fixture (fun ~expected:_ ~replay ~create ~finish ~matrix ~root:_ ->
      let policy = List.hd R.Run.Policy.all in
      let report = replay policy in
      let run = create policy report in
      summary "interrupted" (matrix ());
      finish run;
      let overwrite =
        try
          ignore (R.Run.finish run ~identity);
          false
        with Unix.Unix_error (Unix.EEXIST, _, _) -> true
      in
      Fmt.pr "overwrite refused=%b@." overwrite;
      let path =
        Filename.concat run.dir
          (Transformers_metadata.Source.hash Release.artifact_id
          ^ ".replay.json")
      in
      Unix.chmod path 0o644;
      write path "{broken";
      summary "corrupt" (matrix ()));
  [%expect
    {|
    interrupted: passed=false [incomplete, not run, not run, not run], historical=0 invalid=0
    overwrite refused=true
    corrupt: passed=false [not run, not run, not run, not run], historical=0 invalid=1 |}]

let%expect_test "acquisition errors remain durable without outputs" =
  fixture (fun ~expected ~replay:_ ~create ~finish ~matrix ~root:_ ->
      let policy = List.hd R.Run.Policy.all in
      let report =
        get
          (R.Run.acquisition_error ~consumer:(String.make 40 'a') ~policy
             ~expected "checkpoint digest failed")
      in
      finish (create policy report);
      summary "acquisition" (matrix ()));
  [%expect
    {|
    acquisition: passed=false [failed, not run, not run, not run], historical=0 invalid=0 |}]

let%expect_test
    "same changed-file count and assume-unchanged cannot hide source bytes" =
  with_dir (fun dir ->
      let git args = get (Transformers_metadata.Producer.command dir args) in
      ignore (git [ "init"; "-q" ]);
      ignore (git [ "config"; "user.name"; "fixture" ]);
      ignore (git [ "config"; "user.email"; "fixture@example.invalid" ]);
      let file = Filename.concat dir "code.ml" in
      write file "let x = 0\n";
      ignore (git [ "add"; "code.ml" ]);
      ignore (git [ "commit"; "-qm"; "fixture" ]);
      write file "let x = 1\n";
      let a = get (R.Identity.snapshot dir [ "." ]) in
      let status_a = git [ "status"; "--porcelain" ] in
      write file "let x = 2\n";
      let b = get (R.Identity.snapshot dir [ "." ]) in
      let status_b = git [ "status"; "--porcelain" ] in
      ignore (git [ "update-index"; "--assume-unchanged"; "code.ml" ]);
      let hidden = get (R.Identity.snapshot dir [ "." ]) in
      write file "let x = 3\n";
      let hidden2 = get (R.Identity.snapshot dir [ "." ]) in
      Fmt.pr "same status=%b different bytes=%b hidden change=%b@."
        (status_a = status_b)
        (get (text a) <> get (text b))
        (get (text hidden) <> get (text hidden2)));
  [%expect {| same status=true different bytes=true hidden change=true |}]
