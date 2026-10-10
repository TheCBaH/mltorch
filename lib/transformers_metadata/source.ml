(* Offline catalogue validation and admission summary. The shell boundary checks
   the gitlink/checkout before this command reads any producer metadata. *)
open Err.Syntax
module J = Jsont.Json

let ( >>= ) = Err.bind
let bad detail = Err.fail (`Source detail)
let read path = In_channel.with_open_bin path In_channel.input_all
let hash bytes = Pt2_sha256.(Digest.to_hex (string bytes))
let obj fields = J.object' (List.map (fun (k, v) -> J.mem (J.name k) v) fields)

let members = function
  | Jsont.Object (ms, _) -> Ok (List.map (fun ((k, _), v) -> (k, v)) ms)
  | _ -> bad "expected object"

let member k j =
  let* ms = members j in
  Err.of_option (`Source ("missing member: " ^ k)) (List.assoc_opt k ms)

let string = function Jsont.String (s, _) -> Ok s | _ -> bad "expected string"
let array = function Jsont.Array (xs, _) -> Ok xs | _ -> bad "expected array"

let integer = function
  | Jsont.Number (n, _)
    when Float.is_finite n && n >= 0. && n <= 100_000_000. && Float.floor n = n
    ->
      Ok (Int64.of_float n)
  | _ -> bad "expected nonnegative integer <= 100000000"

let unique what names =
  let rec check = function
    | a :: (b :: _ as rest) ->
        if a = b then bad ("duplicate " ^ what ^ ": " ^ a) else check rest
    | _ -> Ok ()
  in
  check (List.sort String.compare names)

let rec keys = function
  | Jsont.Object (ms, _) ->
      let* () = unique "JSON member" (List.map (fun ((k, _), _) -> k) ms) in
      Err.List.iter (fun (_, v) -> keys v) ms
  | Jsont.Array (xs, _) -> Err.List.iter keys xs
  | _ -> Ok ()

let parse bytes =
  let* j =
    Err.import ~pos:__POS__
      (fun e -> `Source e)
      (Jsont_bytesrw.decode_string Jsont.json bytes)
  in
  let* () = keys j in
  Ok j

let encode path j =
  let* text =
    Err.import ~pos:__POS__
      (fun e -> `Source e)
      (Jsont_bytesrw.encode_string ~format:Jsont.Indent Jsont.json j)
  in
  Out_channel.with_open_bin path (fun oc ->
      output_string oc text;
      output_char oc '\n');
  Ok ()

let safe path =
  let parts = String.split_on_char '/' path in
  if
    List.exists
      (fun p ->
        p = "" || p = "." || p = ".." || String.contains p '\\'
        || String.contains p '\000')
      parts
  then bad ("unsafe relative path: " ^ path)
  else Ok parts

(* Check every path component: realpath alone would allow an in-tree symlink
   to substitute unlisted bytes or a directory outside the catalogue. *)
let regular source path =
  let* parts = safe path in
  let rec walk base = function
    | [] -> bad ("empty file path: " ^ path)
    | name :: rest ->
        let full = Filename.concat base name in
        let stat = Unix.lstat full in
        let expected = if rest = [] then Unix.S_REG else Unix.S_DIR in
        if stat.Unix.st_kind <> expected then bad ("non-regular path: " ^ full)
        else if rest = [] then Ok full
        else walk full rest
  in
  walk source parts

module Artifact = struct
  type t = { id : string; flat : string; path : string; files : int }
end

let artifact source row =
  let* id = member "artifact_id" row >>= string in
  let* parts = safe id in
  let* path = member "path" row >>= string in
  let* _ = safe path in
  let* files = member "files" row >>= members in
  let required =
    [
      "captures.json";
      "cases.json";
      "contract.json";
      "data/constants/model_constants_config.json";
      "data/weights/model_weights_config.json";
      "models/model.json";
      "models/op_facts.json";
    ]
  in
  let* () =
    Err.List.iter
      (fun name ->
        if List.mem_assoc name files then Ok ()
        else bad (id ^ ": missing file " ^ name))
      required
  in
  let* () =
    Err.List.iter
      (fun (name, pin) ->
        let* _ = safe name in
        let* file = regular source (path ^ "/" ^ name) in
        let* size = member "size" pin >>= integer in
        let* digest = member "sha256" pin >>= string in
        if Int64.of_int (Unix.stat file).Unix.st_size <> size then
          bad ("size differs: " ^ file)
        else if hash (read file) <> digest then bad ("sha256 differs: " ^ file)
        else Ok ())
      files
  in
  let* graph = member "graph_sha256" row >>= string in
  let* graph_pin =
    member "sha256" (List.assoc "models/model.json" files) >>= string
  in
  if graph <> graph_pin then bad (id ^ ": graph pin differs from files")
  else
    Ok
      {
        Artifact.id;
        flat = String.concat "--" parts;
        path;
        files = List.length files;
      }

let inventory source flat output =
  let bytes = read (Filename.concat source "catalogue.json") in
  let* catalogue = parse bytes in
  let* schema = member "schema_version" catalogue >>= integer in
  let* () = if schema = 1L then Ok () else bad "unsupported catalogue schema" in
  let* rows = member "artifacts" catalogue >>= array in
  let* () =
    if rows = [] || List.length rows > 1000 then bad "invalid artifact count"
    else Ok ()
  in
  let* artifacts = Err.List.map (artifact source) rows in
  let* () =
    unique "artifact ID" (List.map (fun a -> a.Artifact.id) artifacts)
  in
  let* () =
    unique "flattened artifact ID"
      (List.map (fun a -> a.Artifact.flat) artifacts)
  in
  let* () =
    unique "artifact path" (List.map (fun a -> a.Artifact.path) artifacts)
  in
  (* No writes before the complete inventory has been checked. *)
  List.iter
    (fun a ->
      Unix.symlink
        (Filename.concat source a.Artifact.path)
        (Filename.concat flat a.Artifact.flat))
    artifacts;
  encode output
    (obj
       [
         ("catalogue_sha256", J.string (hash bytes));
         ("schema_version", J.int 1);
         ( "verified_files",
           J.int (List.fold_left (fun n a -> n + a.Artifact.files) 0 artifacts)
         );
         ( "artifacts",
           J.list
             (List.map
                (fun a ->
                  obj
                    [
                      ("artifact_id", J.string a.Artifact.id);
                      ("model", J.string a.Artifact.flat);
                      ("path", J.string a.Artifact.path);
                    ])
                artifacts) );
       ])

let summary inventory_path admission_path output pin consumer changes =
  let* inventory = parse (read inventory_path) in
  let* expected = member "artifacts" inventory >>= array in
  let* names = Err.List.map (fun j -> member "model" j >>= string) expected in
  let* rows =
    Err.List.map parse
      (String.split_on_char '\n' (read admission_path)
      |> List.filter (( <> ) ""))
  in
  let* actual = Err.List.map (fun j -> member "model" j >>= string) rows in
  let* () =
    if List.sort String.compare names = List.sort String.compare actual then
      Ok ()
    else bad "admission coverage differs from inventory"
  in
  let count key =
    List.fold_left
      (fun n j ->
        match member key j with Ok (Jsont.Bool (true, _)) -> n + 1 | _ -> n)
      0 rows
  in
  let* catalogue = member "catalogue_sha256" inventory in
  let* files = member "verified_files" inventory in
  encode output
    (obj
       [
         ("catalogue_sha256", catalogue);
         ("verified_files", files);
         ("consumer_commit", J.string consumer);
         ("consumer_workspace_changes", J.string changes);
         ("producer_commit", J.string pin);
         ("source_gitlink", J.string pin);
         ("rows", J.int (List.length rows));
         ("native_builds", J.int (count "native_builds"));
         ("native4d_converts", J.int (count "native4d_converts"));
         ("kernel_converts", J.int (count "kernel_converts"));
       ])
