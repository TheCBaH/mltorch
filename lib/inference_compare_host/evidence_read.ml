open Inference_compare_schema
module W = Wire
module H = Inference_compare_codecs.Payload_hasher

type error =
  | Incomplete of string  (** No completion record: not a result. *)
  | Io of File_io.error
  | Corrupt of string
  | Digest_mismatch of string

let pp_error ppf = function
  | Incomplete d ->
      Fmt.pf ppf "%s has no completion record: the run was interrupted" d
  | Io e -> File_io.pp_error ppf e
  | Corrupt m -> Fmt.pf ppf "corrupt evidence: %s" m
  | Digest_mismatch f -> Fmt.pf ppf "%s does not match its pinned digest" f

type t = { dir : string; manifest : W.json; completion : Completion.t }

let ( let* ) = Result.bind
let io r = Result.map_error (fun e -> Io e) r
let corrupt r = Result.map_error (fun m -> Corrupt m) r

let pin : (string * Digest.Payload.t) W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* name = W.Obj.required o "name" W.string in
      let* digest = W.Obj.required o "digest" Wire_digest.payload in
      Ok (name, digest))

let parse text =
  match Wire_text.parse text with
  | Ok j -> Ok j
  | Error _ -> Error (Corrupt "not valid JSON")

let open_ ~hashers dir =
  let completion_path = Filename.concat dir "completion.json" in
  if not (File_io.exists completion_path) then Error (Incomplete dir)
  else
    let* text = io (File_io.read completion_path) in
    let* cj = parse text in
    let* files = corrupt (Json_access.decode_member "files" (W.list pin) cj) in
    let* completion =
      corrupt (Json_access.decode_member "completion" Completion.decode cj)
    in
    let* () =
      List.fold_left
        (fun acc (name, expected) ->
          let* () = acc in
          let* body = io (File_io.read (Filename.concat dir name)) in
          match
            H.digest_string hashers (Digest.Payload.algorithm expected) body
          with
          | Ok d when Digest.Payload.equal d expected -> Ok ()
          | Ok _ -> Error (Digest_mismatch name)
          | Error _ -> Error (Corrupt ("unknown digest algorithm for " ^ name)))
        (Ok ()) files
    in
    let* mtext = io (File_io.read (Filename.concat dir "manifest.json")) in
    let* manifest = parse mtext in
    Ok { dir; manifest; completion }

let events (t : t) =
  let* text = io (File_io.read (Filename.concat t.dir "events.jsonl")) in
  let lines = List.filter (fun l -> l <> "") (String.split_on_char '\n' text) in
  List.fold_left
    (fun acc line ->
      let* acc = acc in
      match Wire_text.decode_string Event.decode line with
      | Ok e -> Ok (acc @ [ e ])
      | Error e ->
          Error
            (Corrupt (Format.asprintf "event: %a" W.pp_error (Err.Error.kind e))))
    (Ok []) lines

let retained (t : t) =
  let* text = io (File_io.read (Filename.concat t.dir "retained.json")) in
  match Wire_text.decode_string Retained.decode_index text with
  | Ok i -> Ok i
  | Error e ->
      Error
        (Corrupt
           (Format.asprintf "retained.json: %a" W.pp_error (Err.Error.kind e)))

let payload ~hashers (t : t) (d : Digest.Payload.t) =
  let path =
    Filename.concat t.dir (Filename.concat "payloads" (H.content_path d))
  in
  let* body = io (File_io.read path) in
  match H.digest_string hashers (Digest.Payload.algorithm d) body with
  | Ok d' when Digest.Payload.equal d' d -> Ok body
  | _ -> Error (Digest_mismatch (H.content_path d))
