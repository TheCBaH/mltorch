open Err.Syntax
open Transformers_metadata.Json_util
module J = Jsont.Json
module Source = Transformers_metadata.Source

module Policy = struct
  type t = { casts : string; dots : string }

  let all =
    [
      { casts = "checked"; dots = "exact" };
      { casts = "checked"; dots = "binary32-sequential" };
      { casts = "saturating"; dots = "exact" };
      { casts = "saturating"; dots = "binary32-sequential" };
    ]

  let backend t =
    "native-direct"
    ^ (if t.dots = "exact" then "" else "+binary32-sequential-dots")
    ^ if t.casts = "checked" then "" else "+saturating-casts"

  let json t =
    obj
      [
        ("casts", J.string t.casts);
        ("dots", J.string t.dots);
        ("route", J.string "Native_interp.run_named");
        ("normalization", J.string "empty-cache-opt-in");
      ]

  let of_json value =
    let* casts = field "casts" value in
    let* dots = field "dots" value in
    let t = { casts; dots } in
    let* () = if List.mem t all then Ok () else invalid "execution policy" in
    let+ () =
      equal ~identity:"policy" ~field:"effective route" value (json t)
    in
    t
end

type t = {
  dir : string;
  id : string;
  manifest : Jsont.json;
  sha256 : string;
  mutable reports : (string * Jsont.json) list;
}

let exclusive file bytes =
  let fd =
    Unix.openfile file [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_EXCL ] 0o444
  in
  let oc = Unix.out_channel_of_descr fd in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () ->
      output_string oc bytes;
      flush oc;
      Unix.fsync fd)

let json_bytes json =
  let+ bytes = text json in
  bytes ^ "\n"

let create ~root ~identity ~policy ~expected =
  Pt2_fixture_unix.Cache.mkdir_p root;
  let tmp = Filename.temp_file ~temp_dir:root "run-" "" in
  Unix.unlink tmp;
  Unix.mkdir tmp 0o755;
  let id = Filename.basename tmp in
  let manifest =
    obj
      [
        ("schema_version", J.int 1);
        ("run_id", J.string id);
        ("identity", identity);
        ("policy", Policy.json policy);
        ("backend", J.string (Policy.backend policy));
        ("expected", J.list expected);
      ]
  in
  let* bytes = json_bytes manifest in
  let sha256 = Source.hash bytes in
  exclusive (Filename.concat tmp "run.json") bytes;
  Ok { dir = tmp; id; manifest; sha256; reports = [] }

let envelope t json =
  let* fields = members json in
  let* execution = member "policy" t.manifest in
  Ok
    (obj
       (fields
       @ [
           ("run_id", J.string t.id);
           ("run_manifest_sha256", J.string t.sha256);
           ("execution", execution);
         ]))

let add t json =
  let* id = field "artifact_id" json in
  let* () = unique (id :: List.map fst t.reports) in
  let* json = envelope t json in
  let* bytes = json_bytes json in
  let name = Source.hash id ^ ".replay.json" in
  exclusive (Filename.concat t.dir name) bytes;
  t.reports <-
    ( id,
      obj
        [
          ("artifact_id", J.string id);
          ("file", J.string name);
          ("sha256", J.string (Source.hash bytes));
        ] )
    :: t.reports;
  Ok ()

let finish t ~identity =
  let* original = member "identity" t.manifest in
  let* unchanged =
    let* a = text original in
    let+ b = text identity in
    a = b
  in
  let completion =
    obj
      [
        ("schema_version", J.int 1);
        ("run_id", J.string t.id);
        ("run_manifest_sha256", J.string t.sha256);
        ("identity_unchanged", J.bool unchanged);
        ("reports", J.list (List.map snd (List.rev t.reports)));
      ]
  in
  let* bytes = json_bytes completion in
  exclusive (Filename.concat t.dir "completion.json") bytes;
  Unix.chmod t.dir 0o555;
  if unchanged then Ok () else invalid "workspace changed during replay"

let acquisition_error ~consumer ~policy ~expected error =
  let* artifact_id = member "artifact_id" expected in
  let+ pins = Expectation.report_pins expected in
  obj
    [
      ("schema_version", J.int 3);
      ("artifact_id", artifact_id);
      ("status", J.string "failed");
      ("backend", J.string (Policy.backend policy));
      ("consumer", J.string consumer);
      ("acquisition_error", J.string error);
      ("pins", pins);
      ("cases", J.list []);
      ("normalizations", J.list []);
      ("refusal", J.null ());
      ("scope", J.null ());
    ]
