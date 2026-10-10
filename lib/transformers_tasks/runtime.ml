open Err.Syntax
open Transformers_metadata.Json_util
module J = Jsont.Json
module R = Transformers_reports
module Source = Transformers_metadata.Source

let context ~cohort ~selection ~executable =
  let* identity =
    R.Identity.capture ~consumer_root:"."
      ~source:"modules/devcontainer.transformers" ~cohort
  in
  let* bytes = Pt2_fixture_unix.Fetch.read selection in
  let* request = parse bytes in
  let* executable = R.Identity.file "." executable in
  Ok
    (obj
       [
         ("identity", identity);
         ("selection", request);
         ("selection_sha256", J.string (Source.hash bytes));
         ("executable", snd executable);
         ( "route",
           J.string
             "OCaml verification and Pt2_fixture.Compare.tensor over producer \
              data" );
         ("consumer_model_execution", J.bool false);
       ])

type t = {
  context : Jsont.json;
  dir : string;
  id : string;
  sha256 : string;
  mutable reports : Jsont.json list;
}

let create root context =
  Pt2_fixture_unix.Cache.mkdir_p root;
  let dir = Filename.temp_file ~temp_dir:root "task-run-" "" in
  Unix.unlink dir;
  Unix.mkdir dir 0o755;
  let id = Filename.basename dir in
  let manifest =
    obj
      [
        ("schema_version", J.int 1);
        ("run_id", J.string id);
        ("context", context);
      ]
  in
  let* bytes = R.Run.json_bytes manifest in
  R.Run.exclusive (Filename.concat dir "run.json") bytes;
  Ok { context; dir; id; sha256 = Source.hash bytes; reports = [] }

let add t fixture_id result =
  let* fields = members result in
  let result =
    obj
      (fields
      @ [
          ("run_id", J.string t.id); ("run_manifest_sha256", J.string t.sha256);
        ])
  in
  let* bytes = R.Run.json_bytes result in
  let file = Source.hash fixture_id ^ ".json" in
  R.Run.exclusive (Filename.concat t.dir file) bytes;
  t.reports <-
    obj
      [
        ("fixture_id", J.string fixture_id);
        ("file", J.string file);
        ("sha256", J.string (Source.hash bytes));
        ("size", J.int (String.length bytes));
      ]
    :: t.reports;
  Ok ()

let finish t final_context =
  let* a = text t.context in
  let* b = text final_context in
  let unchanged = a = b in
  let* bytes =
    R.Run.json_bytes
      (obj
         [
           ("schema_version", J.int 1);
           ("run_id", J.string t.id);
           ("run_manifest_sha256", J.string t.sha256);
           ("context_unchanged", J.bool unchanged);
           ("reports", J.list (List.rev t.reports));
         ])
  in
  R.Run.exclusive (Filename.concat t.dir "completion.json") bytes;
  Unix.chmod t.dir 0o555;
  Spec.require unchanged "workspace/executable changed during task verification"
