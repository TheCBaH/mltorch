open Err.Syntax
open Transformers_metadata.Json_util
module J = Jsont.Json
module Source = Transformers_metadata.Source

module Observation = struct
  type t = {
    artifact : string;
    backend : string;
    details : string list;
    report : Jsont.json option;
    run : string;
    status : string;
  }

  let json t =
    obj
      [
        ("artifact_id", J.string t.artifact);
        ("backend", J.string t.backend);
        ("details", Identity.strings t.details);
        ("run", J.string t.run);
        ("status", J.string t.status);
        ("report", Option.value ~default:(J.null ()) t.report);
      ]
end

module Scan = struct
  type t = {
    current : Observation.t list;
    historical : Jsont.json list;
    invalid : string list;
  }

  let empty = { current = []; historical = []; invalid = [] }
end

let error e = Fmt.str "%a" Transformers_metadata.Fault.pp_error e

let regular path =
  let* () =
    Validate.require
      ((Unix.lstat path).Unix.st_kind = Unix.S_REG)
      ("non-regular report: " ^ path)
  in
  Pt2_fixture_unix.Fetch.read path

let same_identity a b =
  let* a = text a in
  let+ b = text b in
  a = b

let run ~identity ~expected dir =
  let* bytes = regular (Filename.concat dir "run.json") in
  let manifest_sha256 = Source.hash bytes in
  let* manifest = parse bytes in
  let* () = schema 1 manifest in
  let* run_id = field "run_id" manifest in
  let* () =
    Validate.require (run_id = Filename.basename dir) "run directory identity"
  in
  let* have_identity = member "identity" manifest in
  let* current = same_identity have_identity identity in
  let* policy = member "policy" manifest >>= Run.Policy.of_json in
  let backend = Run.Policy.backend policy in
  let* declared = field "backend" manifest in
  let* () = Validate.require (declared = backend) "manifest backend" in
  let* selected = member "expected" manifest >>= array in
  let* ids = Err.List.map (field "artifact_id") selected in
  let* () = unique ids in
  let* () = Validate.require (ids <> []) "empty run selection" in
  let completion_path = Filename.concat dir "completion.json" in
  let* completion =
    if Sys.file_exists completion_path then
      let* bytes = regular completion_path in
      let+ json = parse bytes in
      Some json
    else Ok None
  in
  let* files =
    match completion with
    | None -> Ok []
    | Some json ->
        let* () = schema 1 json in
        let* id = field "run_id" json in
        let* sha = field "run_manifest_sha256" json in
        let* unchanged = member "identity_unchanged" json >>= bool in
        let* () =
          Validate.require
            (id = run_id && sha = manifest_sha256 && unchanged)
            "completion identity/digest"
        in
        let* reports = member "reports" json >>= array in
        let* files =
          Err.List.map
            (fun row ->
              let+ id = field "artifact_id" row in
              (id, row))
            reports
        in
        let* () = unique (List.map fst files) in
        let+ () =
          Validate.require
            (List.sort String.compare ids
            = List.sort String.compare (List.map fst files))
            "completed run has missing/extra artifact reports"
        in
        files
  in
  let allowed =
    [ "run.json"; "completion.json" ]
    @ List.map (fun id -> Source.hash id ^ ".replay.json") ids
  in
  let* () =
    Err.List.iter
      (fun name ->
        Validate.require (List.mem name allowed)
          ("unexpected run member: " ^ name))
      (Array.to_list (Sys.readdir dir))
  in
  let* observations =
    Err.List.map
      (fun exp ->
        let* artifact = field "artifact_id" exp in
        let* () =
          if current then
            match List.assoc_opt artifact expected with
            | None -> invalid ("artifact outside current cohort: " ^ artifact)
            | Some want ->
                let* ms = members exp in
                if List.mem_assoc "metadata_error" ms then
                  let keys =
                    List.map fst
                      (match Err.payload (members exp) with
                      | Ok rows -> List.remove_assoc "metadata_error" rows
                      | Error _ -> [])
                  in
                  let* want = members want in
                  equal ~identity:artifact ~field:"entry pins"
                    (obj (List.remove_assoc "metadata_error" ms))
                    (obj (List.filter (fun (key, _) -> List.mem key keys) want))
                else
                  equal ~identity:artifact
                    ~field:"verified contract and full pins" exp want
          else Ok ()
        in
        let name = Source.hash artifact ^ ".replay.json" in
        let path = Filename.concat dir name in
        if (not (Sys.file_exists path)) && completion = None then
          Ok
            {
              Observation.artifact;
              backend;
              details = [ "report not written" ];
              report = None;
              run = run_id;
              status = "incomplete";
            }
        else
          let* bytes = regular path in
          let* () =
            match List.assoc_opt artifact files with
            | None -> Ok ()
            | Some row ->
                let* pinned_name = field "file" row in
                let* sha = field "sha256" row in
                Validate.require
                  (pinned_name = name && sha = Source.hash bytes)
                  "report file digest/name"
          in
          let* report = parse bytes in
          let* status, details =
            Validate.report ~manifest ~manifest_sha256 ~expected:exp report
          in
          Ok
            {
              Observation.artifact;
              backend;
              details;
              report = Some report;
              run = run_id;
              status = (if completion = None then "incomplete" else status);
            })
      selected
  in
  if current then Ok { Scan.empty with current = observations }
  else
    Ok
      {
        Scan.empty with
        historical =
          [
            obj
              [
                ( "reason",
                  J.string "consumer/source/cohort/environment identity differs"
                );
                ("manifest", manifest);
                ("observations", J.list (List.map Observation.json observations));
              ];
          ];
      }

let append a b =
  {
    Scan.current = a.Scan.current @ b.Scan.current;
    historical = a.historical @ b.historical;
    invalid = a.invalid @ b.invalid;
  }

let scan ~identity ~expected roots =
  let inspect path =
    try
      match Err.payload (run ~identity ~expected path) with
      | Ok result -> result
      | Error e -> { Scan.empty with invalid = [ path ^ ": " ^ error e ] }
    with Unix.Unix_error (e, fn, arg) ->
      {
        Scan.empty with
        invalid = [ path ^ ": " ^ fn ^ " " ^ arg ^ ": " ^ Unix.error_message e ];
      }
  in
  let legacy path =
    match Err.payload (regular path >>= parse) with
    | Ok json ->
        {
          Scan.empty with
          historical =
            [
              obj
                [
                  ("file", J.string path);
                  ( "reason",
                    J.string "legacy report has no immutable run manifest" );
                  ("report", json);
                ];
            ];
        }
    | Error e -> { Scan.empty with invalid = [ path ^ ": " ^ error e ] }
  in
  let root path =
    if not (Sys.file_exists path) then Scan.empty
    else if Sys.file_exists (Filename.concat path "run.json") then inspect path
    else
      Array.to_list (Sys.readdir path)
      |> List.sort String.compare
      |> List.fold_left
           (fun acc name ->
             let p = Filename.concat path name in
             let stat = Unix.lstat p in
             let result =
               if stat.Unix.st_kind = Unix.S_LNK then
                 { Scan.empty with invalid = [ "report symlink: " ^ p ] }
               else if stat.st_kind = Unix.S_DIR then inspect p
               else if Filename.check_suffix name ".replay.json" then legacy p
               else Scan.empty
             in
             append acc result)
           Scan.empty
  in
  List.fold_left (fun acc path -> append acc (root path)) Scan.empty roots

let generate ~identity ~expected roots =
  let scans = scan ~identity ~expected roots in
  let rows =
    List.concat_map
      (fun (artifact, _) ->
        List.map
          (fun policy ->
            let backend = Run.Policy.backend policy in
            let runs =
              List.filter
                (fun o ->
                  o.Observation.artifact = artifact && o.backend = backend)
                scans.current
            in
            let status =
              match runs with
              | [] -> "not run"
              | [ o ] -> o.status
              | _ -> "conflict"
            in
            obj
              [
                ("artifact_id", J.string artifact);
                ("backend", J.string backend);
                ("status", J.string status);
                ("runs", J.list (List.map Observation.json runs));
              ])
          Run.Policy.all)
      expected
  in
  let passed =
    scans.invalid = [] && rows <> []
    && List.for_all
         (fun row -> Err.payload (field "status" row) = Ok "passed")
         rows
  in
  obj
    [
      ("schema_version", J.int 1);
      ("identity", identity);
      ("passed", J.bool passed);
      ("expected_policies", J.list (List.map Run.Policy.json Run.Policy.all));
      ("rows", J.list rows);
      ("historical", J.list scans.historical);
      ("invalid", Identity.strings scans.invalid);
    ]

let markdown matrix =
  let escape s =
    String.concat "\\|" (String.split_on_char '|' s)
    |> String.split_on_char '\n' |> String.concat " "
  in
  let* rows = member "rows" matrix >>= array in
  let* lines =
    Err.List.map
      (fun row ->
        let* id = field "artifact_id" row in
        let* backend = field "backend" row in
        let* status = field "status" row in
        let* runs = member "runs" row >>= array in
        let* details =
          Err.List.map
            (fun run ->
              let* id = field "run" run in
              let+ details = member "details" run >>= Validate.names in
              id
              ^ if details = [] then "" else ": " ^ String.concat "; " details)
            runs
        in
        Ok
          ("| "
          ^ String.concat " | "
              (List.map escape
                 [ id; backend; status; String.concat "; " details ])
          ^ " |"))
      rows
  in
  let* historical = member "historical" matrix >>= array in
  let* invalid = member "invalid" matrix >>= Validate.names in
  let* passed = member "passed" matrix >>= bool in
  Ok
    ("Current matrix: "
    ^ (if passed then "passed" else "incomplete or failing")
    ^ ". All four explicit policies are shown; numerical gate selection is \
       separate.\n\n"
    ^ "| Artifact | Policy / route | Status | Evidence / failure |\n\
       |---|---|---|---|\n" ^ String.concat "\n" lines
    ^ "\n\nHistorical reports: "
    ^ string_of_int (List.length historical)
    ^ " (excluded from current rows; identities and reports retained in JSON).\n"
    ^
    if invalid = [] then ""
    else
      "\nInvalid runs:\n\n"
      ^ String.concat "\n" (List.map (fun s -> "- " ^ escape s) invalid)
      ^ "\n")
