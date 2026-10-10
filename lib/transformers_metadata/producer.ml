open Err.Syntax
open Json_util
module J = Jsont.Json

type t = {
  candidates : Jsont.json list;
  catalogue : Jsont.json list;
  commit : string;
  metadata : Jsont.json;
  root : string;
  task_models : Jsont.json list;
}

let command cwd args =
  let argv = Array.of_list ("git" :: "-C" :: cwd :: args) in
  let ic = Unix.open_process_args_in "git" argv in
  let text = In_channel.input_all ic in
  match Unix.close_process_in ic with
  | Unix.WEXITED 0 -> Ok text
  | _ -> invalid ("git failed: " ^ String.concat " " args)

let checked_text root commit relative =
  let* file = Source.regular root relative in
  let* actual = Pt2_fixture_unix.Fetch.read file in
  let* expected = command root [ "show"; commit ^ ":" ^ relative ] in
  let* () =
    same ~identity:relative ~field:"committed bytes" (Source.hash actual)
      (Source.hash expected)
  in
  Ok actual

let load ~consumer_root ~root =
  let* index =
    command consumer_root
      [ "ls-files"; "--stage"; "--"; "modules/devcontainer.transformers" ]
  in
  let* commit =
    match String.split_on_char ' ' (String.trim index) with
    | [ "160000"; pin; stage ] when String.starts_with ~prefix:"0\t" stage ->
        Ok pin
    | _ -> invalid "consumer index has no unconflicted producer gitlink"
  in
  if not (Sys.file_exists root) then
    missing
      "producer checkout; run git submodule update --init \
       modules/devcontainer.transformers"
  else
    let root = Unix.realpath root in
    let* top = command root [ "rev-parse"; "--show-toplevel" ] in
    let* () =
      same ~identity:root ~field:"initialized checkout root" (String.trim top)
        root
    in
    let* have = command root [ "rev-parse"; "HEAD" ] in
    let* () = same ~identity:root ~field:"HEAD" (String.trim have) commit in
    let* dirty =
      command root [ "status"; "--porcelain"; "--untracked-files=all" ]
    in
    let* () =
      same ~identity:root ~field:"checkout changes" (String.trim dirty) ""
    in
    let* bytes = checked_text root commit "catalogue.json" in
    let* catalogue = parse bytes in
    let* () = schema 1 catalogue in
    let* rows = member "artifacts" catalogue >>= array in
    let* () =
      if rows <> [] && List.length rows <= 1000 then Ok ()
      else invalid "source artifact count"
    in
    let* artifacts = Err.List.map (Source.artifact root) rows in
    let* () = unique (List.map (fun a -> a.Source.Artifact.id) artifacts) in
    let* () = unique (List.map (fun a -> a.Source.Artifact.flat) artifacts) in
    let* () = unique (List.map (fun a -> a.Source.Artifact.path) artifacts) in
    let* task_bytes = checked_text root commit "task-assets.json" in
    let* tasks = parse task_bytes in
    let* () = schema 1 tasks in
    let* task_models = member "models" tasks >>= array in
    let* ids = Err.List.map (field "model_id") task_models in
    let* () = unique ids in
    let* candidate_bytes = checked_text root commit "model-candidates.yaml" in
    let* candidate_document = parse candidate_bytes in
    let* () = schema 1 candidate_document in
    let* candidates = member "models" candidate_document >>= array in
    let* ids = Err.List.map (field "id") candidates in
    let* () = unique ids in
    let* () =
      Err.List.iter
        (fun row ->
          let* id = field "id" row in
          let* ms = members row in
          let* config_id =
            match List.assoc_opt "config_model_id" ms with
            | Some j -> string j
            | None -> Ok id
          in
          let* bytes =
            checked_text root commit ("configs/reference/" ^ config_id ^ ".json")
          in
          let* expected =
            path [ "reference"; "config_sha256" ] row >>= string
          in
          same ~identity:id ~field:"source reference config" (Source.hash bytes)
            expected)
        candidates
    in
    let files =
      [
        "docs/checkpoint-map-v2.md";
        "src/hf_pt2_tools/recipes.py";
        "src/hf_pt2_tools/registry.py";
        "src/hf_pt2_tools/generation.py";
        "src/hf_pt2_tools/encoders.py";
        "uv.lock";
      ]
    in
    let* task_files =
      command root
        [
          "ls-tree";
          "-r";
          "--name-only";
          commit;
          "--";
          "task-recipes.json";
          "task-fixture-request.json";
          "task-fixtures.pin.json";
          "schemas/task-index.schema.json";
          "schemas/task-manifest.schema.json";
          "schemas/task-contract.schema.json";
          "docs/task-fixtures-v1.md";
        ]
    in
    let files =
      files @ List.filter (( <> ) "") (String.split_on_char '\n' task_files)
    in
    let* metadata_files =
      Err.List.map
        (fun file ->
          let+ bytes = checked_text root commit file in
          (file, J.string (Source.hash bytes)))
        files
    in
    Ok
      {
        candidates;
        catalogue = rows;
        commit;
        root;
        task_models;
        metadata =
          obj
            [
              ("catalogue_sha256", J.string (Source.hash bytes));
              ("task_assets_sha256", J.string (Source.hash task_bytes));
              ("candidates_sha256", J.string (Source.hash candidate_bytes));
              ("files", obj metadata_files);
              ("catalogue_schema", J.int 1);
              ("task_assets_schema", J.int 1);
              ( "checkpoint_map_schema",
                J.int Pt2_checkpoint_map.Document.supported_schema_version );
            ];
      }

let candidate t model =
  let rec find = function
    | [] -> missing ("source candidate " ^ model)
    | row :: rest ->
        let* id = field "id" row in
        if id = model then Ok row else find rest
  in
  find t.candidates

let shape_kind s =
  match s with
  | "static" | "dynamic" -> Ok s
  | _ when String.starts_with ~prefix:"static-h" s ->
      let digits = String.sub s 8 (String.length s - 8) in
      if
        digits <> ""
        && String.for_all (function '0' .. '9' -> true | _ -> false) digits
      then
        match Int64.of_string_opt digits with
        | Some n when n > 0L && n <= 1_000_000L -> Ok "static-history"
        | _ -> invalid ("static history bound: " ^ s)
      else invalid ("shape policy: " ^ s)
  | _ -> invalid ("shape policy: " ^ s)

let source_artifact t id =
  let* parts = Source.safe id in
  match parts with
  | [
   model; category; "reference"; component; dtype; policy; shape; checkpoint;
  ] ->
      let* () =
        if String.starts_with ~prefix:"ckpt-" checkpoint then Ok ()
        else invalid ("checkpoint artifact ID: " ^ id)
      in
      let* candidate = candidate t model in
      let* candidate_category = field "category" candidate in
      let* () =
        same ~identity:id ~field:"source candidate category" candidate_category
          category
      in
      let source_id =
        String.concat "/"
          [ model; category; "tiny"; component; dtype; policy; shape ]
      in
      let* want_shape = shape_kind shape in
      let* matching =
        filter
          (fun row ->
            let* m = field "model_id" row in
            let* c = field "component" row in
            let* d = field "dtype" row in
            let* p = field "policy" row in
            let* s = field "shape_policy" row in
            let+ kind = shape_kind s in
            m = model && c = component && d = dtype && p = policy
            && kind = want_shape)
          t.catalogue
      in
      let* exact =
        filter
          (fun row ->
            let+ have = field "artifact_id" row in
            have = source_id)
          matching
      in
      let chosen =
        match (exact, matching) with
        | r :: _, _ | [], r :: _ -> Some r
        | [], [] -> None
      in
      let* () =
        match chosen with
        | Some _ -> Ok ()
        | None
          when component = "forward" && dtype = "fp32" && policy = "dynamo"
               && shape = "static" ->
            Ok ()
        | None -> missing ("source component recipe " ^ source_id)
      in
      Ok (candidate, chosen)
  | _ -> invalid ("reference artifact ID: " ^ id)

let task t model =
  let rec find = function
    | [] -> missing ("source task metadata for " ^ model)
    | row :: rest ->
        let* id = field "model_id" row in
        if id = model then Ok row else find rest
  in
  find t.task_models
