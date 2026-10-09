open Schema_runtime
open Err.Syntax

module Pin = struct
  type t = {
    name : string;
    sha256 : Pt2_sha256.Digest.t;
    size : int64;
    url : string;
  }
end

module Derived = struct
  type t = {
    file : string;
    repo_id : string;
    revision : string;
    sha256 : Pt2_sha256.Digest.t;
    tool : string;
  }
end

module Upstream = struct
  type t = { repo_id : string; revision : string }
end

module Source = struct
  type provenance = Converted of Derived.t | Upstream of Upstream.t
  type t = { pin : Pin.t; provenance : provenance }
end

module Origin = struct
  type convert = Cast of { from : Dtype.t; to_ : Dtype.t } | Identity

  module Checkpoint = struct
    type t = {
      convert : convert;
      file : string;
      key : string;
      tied_aliases : string list;
    }
  end

  type t =
    | Checkpoint of Checkpoint.t
    | Empty
    | Fill of string
    | Inline of string
    | Pack of string
end

(* Overflow-checked: the division recovers the factor only if the multiply did
   not wrap. [None] past [int64] -- and past 2^62, so the caller's own further
   multiply by a small width has headroom to be checked the same way. *)
let checked_mul a b =
  if Int64.equal a 0L || Int64.equal b 0L then Some 0L
  else if Int64.compare a 0L < 0 || Int64.compare b 0L < 0 then None
  else
    let r = Int64.mul a b in
    if Int64.compare r 0L < 0 || not (Int64.equal (Int64.div r b) a) then None
    else Some r

let element_count_opt shape =
  List.fold_left
    (fun acc d -> match acc with None -> None | Some a -> checked_mul a d)
    (Some 1L) shape

module Entry = struct
  type t = {
    dtype : Dtype.t;
    origin : Origin.t;
    sha256 : Pt2_sha256.Digest.t;
    shape : int64 list;
  }

  let element_count t = Option.get (element_count_opt t.shape)

  let byte_count t =
    Option.get
      (checked_mul (element_count t) (Int64.of_int (Dtype.byte_width t.dtype)))
end

type t = {
  artifact_id : string;
  checkpoint_files : Source.t list;
  graph_owned : Pin.t option;
  graph_sha256 : Pt2_sha256.Digest.t;
  model_id : string;
  tensors : Entry.t String_map.t;
}

let supported_schema_version = 2

(* The wire shapes, exactly as the JSON schema states them. Only [of_string]
   turns them into the typed records above, so a decode failure and a semantic
   failure are different errors. *)
module Wire = struct
  type pin = {
    name : string;
    sha256 : string;
    size : int64;
    url : string;
    repo_id : string option;
    revision : string option;
    derived_from : derived option;
  }

  and derived = {
    d_repo_id : string;
    d_revision : string;
    d_file : string;
    d_sha256 : string;
    d_tool : string;
  }

  let derived_jsont =
    Jsont.Object.map ~kind:"derived_from"
      (fun d_repo_id d_revision d_file d_sha256 d_tool ->
        { d_repo_id; d_revision; d_file; d_sha256; d_tool })
    |> Jsont.Object.mem "repo_id" Jsont.string
    |> Jsont.Object.mem "revision" Jsont.string
    |> Jsont.Object.mem "file" Jsont.string
    |> Jsont.Object.mem "sha256" Jsont.string
    |> Jsont.Object.mem "tool" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  (* The schema does not close a pinned file: a producer may add a field. *)
  let pin_jsont =
    Jsont.Object.map ~kind:"pinned file"
      (fun name sha256 size url repo_id revision derived_from ->
        { name; sha256; size; url; repo_id; revision; derived_from })
    |> Jsont.Object.mem "name" Jsont.string
    |> Jsont.Object.mem "sha256" Jsont.string
    |> Jsont.Object.mem "size" Jsont.int64
    |> Jsont.Object.mem "url" Jsont.string
    |> Jsont.Object.opt_mem "repo_id" Jsont.string
    |> Jsont.Object.opt_mem "revision" Jsont.string
    |> Jsont.Object.opt_mem "derived_from" derived_jsont
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  type convert = { op : string; from : string option; to_ : string option }

  let convert_jsont =
    Jsont.Object.map ~kind:"convert" (fun op from to_ -> { op; from; to_ })
    |> Jsont.Object.mem "op" Jsont.string
    |> Jsont.Object.opt_mem "from" Jsont.string
    |> Jsont.Object.opt_mem "to" Jsont.string
    |> Jsont.Object.error_unknown |> Jsont.Object.finish

  type origin =
    | Checkpoint of {
        file : string;
        key : string;
        convert : convert;
        tied_aliases : string list;
      }
    | Generated of { op : string; element_hex : string option }
    | Inline of string
    | Pack of string

  let origin_jsont =
    let checkpoint =
      Jsont.Object.map ~kind:"checkpoint origin"
        (fun file key convert tied_aliases ->
          Checkpoint { file; key; convert; tied_aliases })
      |> Jsont.Object.mem "file" Jsont.string
      |> Jsont.Object.mem "key" Jsont.string
      |> Jsont.Object.mem "convert" convert_jsont
      |> Jsont.Object.mem "tied_aliases" (Jsont.list Jsont.string)
      |> Jsont.Object.error_unknown |> Jsont.Object.finish
    in
    let generated =
      Jsont.Object.map ~kind:"generated origin" (fun op element_hex ->
          Generated { op; element_hex })
      |> Jsont.Object.mem "op" Jsont.string
      |> Jsont.Object.opt_mem "element_hex" Jsont.string
      |> Jsont.Object.error_unknown |> Jsont.Object.finish
    in
    let inline =
      Jsont.Object.map ~kind:"inline origin" (fun data -> Inline data)
      |> Jsont.Object.mem "data_base64" Jsont.string
      |> Jsont.Object.error_unknown |> Jsont.Object.finish
    in
    let pack =
      Jsont.Object.map ~kind:"pack origin" (fun key -> Pack key)
      |> Jsont.Object.mem "key" Jsont.string
      |> Jsont.Object.error_unknown |> Jsont.Object.finish
    in
    let case tag obj =
      Jsont.Object.Case.make (Jsont.Object.Case.map ~dec:Fun.id tag obj)
    in
    Jsont.Object.map ~kind:"origin" Fun.id
    |> Jsont.Object.case_mem "kind" Jsont.string
         [
           case "checkpoint" checkpoint;
           case "generated" generated;
           case "inline" inline;
           case "pack" pack;
         ]
    |> Jsont.Object.error_unknown |> Jsont.Object.finish

  type tensor = {
    dtype : string;
    shape : int64 list;
    sha256 : string;
    origin : origin;
  }

  let tensor_jsont =
    Jsont.Object.map ~kind:"tensor" (fun dtype shape sha256 origin ->
        { dtype; shape; sha256; origin })
    |> Jsont.Object.mem "dtype" Jsont.string
    |> Jsont.Object.mem "shape" (Jsont.list Jsont.int64)
    |> Jsont.Object.mem "sha256" Jsont.string
    |> Jsont.Object.mem "origin" origin_jsont
    |> Jsont.Object.error_unknown |> Jsont.Object.finish

  (* Members kept in document order with duplicates, which a [String_map]
     would silently collapse. *)
  let tensors_jsont =
    let mems =
      Jsont.Object.Mems.map
        ~dec_empty:(fun () -> [])
        ~dec_add:(fun _ name v acc -> (name, v) :: acc)
        ~dec_finish:(fun _ acc -> List.rev acc)
        tensor_jsont
    in
    Jsont.Object.map ~kind:"tensors" Fun.id
    |> Jsont.Object.keep_unknown mems
    |> Jsont.Object.finish

  type sources = { files : pin list; graph_owned : pin option }

  let sources_jsont =
    let checkpoint =
      Jsont.Object.map ~kind:"checkpoint sources" Fun.id
      |> Jsont.Object.mem "files" (Jsont.list pin_jsont)
      |> Jsont.Object.error_unknown |> Jsont.Object.finish
    in
    Jsont.Object.map ~kind:"sources" (fun files graph_owned ->
        { files; graph_owned })
    |> Jsont.Object.mem "checkpoint" checkpoint
    |> Jsont.Object.opt_mem "graph_owned" pin_jsont
    |> Jsont.Object.error_unknown |> Jsont.Object.finish

  type document = {
    schema_version : int;
    artifact_id : string;
    graph_sha256 : string;
    model_id : string;
    sources : sources;
    tensors : (string * tensor) list;
    unmapped : string list;
  }

  let document_jsont =
    Jsont.Object.map ~kind:"checkpoint map"
      (fun
        schema_version
        artifact_id
        graph_sha256
        model_id
        sources
        tensors
        unmapped
      ->
        {
          schema_version;
          artifact_id;
          graph_sha256;
          model_id;
          sources;
          tensors;
          unmapped;
        })
    |> Jsont.Object.mem "schema_version" Jsont.int
    |> Jsont.Object.mem "artifact_id" Jsont.string
    |> Jsont.Object.mem "graph_sha256" Jsont.string
    |> Jsont.Object.mem "model_id" Jsont.string
    |> Jsont.Object.mem "sources" sources_jsont
    |> Jsont.Object.mem "tensors" tensors_jsont
    |> Jsont.Object.mem "unmapped" (Jsont.list Jsont.string)
    |> Jsont.Object.error_unknown |> Jsont.Object.finish
end

let digest_of s =
  match Pt2_sha256.Digest.of_hex s with
  | Some d -> Err.return d
  | None -> Err.fail (`Bad_digest s)

let is_hex40 s =
  String.length s = 40
  && String.for_all (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false) s

let valid_file_name n =
  n <> "" && n <> "." && n <> ".."
  && String.length n <= 255
  && String.for_all (fun c -> c <> '/' && c <> '\\' && c <> '\000') n

let has_prefix ~prefix s =
  String.length s >= String.length prefix
  && String.equal (String.sub s 0 (String.length prefix)) prefix

let pin_of_fields ~name ~sha256 ~size ~url =
  let* sha256 = digest_of sha256 in
  if not (valid_file_name name) then Err.fail (`Bad_pin (Fault.Name, name))
  else if Int64.compare size 1L < 0 then
    Err.fail (`Bad_pin (Fault.Size, Int64.to_string size))
  else if not (has_prefix ~prefix:"https://" url) then
    Err.fail (`Bad_pin (Fault.Url, url))
  else Err.return { Pin.name; sha256; size; url }

let pin_of (w : Wire.pin) =
  pin_of_fields ~name:w.name ~sha256:w.sha256 ~size:w.size ~url:w.url

let source_of (w : Wire.pin) =
  let* pin = pin_of w in
  let revision r =
    if is_hex40 r then Err.return r else Err.fail (`Bad_pin (Fault.Revision, r))
  in
  match (w.repo_id, w.revision, w.derived_from) with
  | Some repo_id, Some rev, None ->
      let* revision = revision rev in
      Err.return
        {
          Source.pin;
          provenance = Source.Upstream { Upstream.repo_id; revision };
        }
  | None, None, Some d ->
      let* revision = revision d.d_revision in
      let* sha256 = digest_of d.d_sha256 in
      Err.return
        {
          Source.pin;
          provenance =
            Source.Converted
              {
                Derived.file = d.d_file;
                repo_id = d.d_repo_id;
                revision;
                sha256;
                tool = d.d_tool;
              };
        }
  | _ ->
      (* Neither or both provenance forms: the schema's [oneOf] fails. *)
      Err.fail (`Bad_pin (Fault.Provenance, w.name))

let over_limit what ~limit ~actual =
  Err.fail (`Over_limit { Fault.Over_limit.what; limit; actual })

let name_set_unique domain names =
  let seen = Hashtbl.create 64 in
  Err.List.iter
    (fun n ->
      if Hashtbl.mem seen n then Err.fail (`Duplicate (domain, n))
      else begin
        Hashtbl.add seen n ();
        Err.return ()
      end)
    names

let convert_of ~target ~(dtype : Dtype.t) (c : Wire.convert) =
  match (c.op, c.from, c.to_) with
  | "none", None, None -> Err.return Origin.Identity
  | "cast", Some from, Some to_ -> (
      match (Dtype.of_code from, Dtype.of_code to_) with
      | Some from, Some to_ ->
          if Dtype.equal from to_ || not (Dtype.equal to_ dtype) then
            Err.fail (`Cast_inconsistent { Fault.Cast.target; from; to_ })
          else Err.return (Origin.Cast { from; to_ })
      | None, _ -> Err.fail (`Unknown_dtype from)
      | _, None -> Err.fail (`Unknown_dtype to_))
  | _ -> Err.fail (`Conversion_malformed target)

let entry_of ~limits ~files ~has_pack target (w : Wire.tensor) =
  let* dtype =
    match Dtype.of_code w.dtype with
    | Some d -> Err.return d
    | None -> Err.fail (`Unknown_dtype w.dtype)
  in
  let* sha256 = digest_of w.sha256 in
  let rank = List.length w.shape in
  let* () =
    if rank > limits.Limits.max_rank then
      over_limit Limits.Rank
        ~limit:(Int64.of_int limits.max_rank)
        ~actual:(Int64.of_int rank)
    else Err.return ()
  in
  let* () =
    if List.exists (fun d -> Int64.compare d 0L < 0) w.shape then
      Err.fail (`Negative_extent target)
    else Err.return ()
  in
  let width = Int64.of_int (Dtype.byte_width dtype) in
  let* elements, bytes =
    match element_count_opt w.shape with
    | None ->
        over_limit Limits.Tensor_bytes ~limit:limits.max_tensor_bytes
          ~actual:Int64.max_int
    | Some n -> (
        match checked_mul n width with
        | Some b when Int64.compare b limits.max_tensor_bytes <= 0 ->
            Err.return (n, b)
        | Some b ->
            over_limit Limits.Tensor_bytes ~limit:limits.max_tensor_bytes
              ~actual:b
        | None ->
            over_limit Limits.Tensor_bytes ~limit:limits.max_tensor_bytes
              ~actual:Int64.max_int)
  in
  let* origin =
    match w.origin with
    | Wire.Checkpoint { file; key; convert; tied_aliases } ->
        if not (List.mem file files) then
          Err.fail (`Unknown_source_file (target, file))
        else
          let* convert = convert_of ~target ~dtype convert in
          Err.return
            (Origin.Checkpoint
               { Origin.Checkpoint.convert; file; key; tied_aliases })
    | Wire.Generated { op = "empty"; element_hex = None } ->
        if Int64.equal elements 0L then Err.return Origin.Empty
        else Err.fail (`Empty_not_empty target)
    | Wire.Generated { op = "fill"; element_hex = Some hex } -> (
        match Codec.hex_decode hex with
        | None -> Err.fail (`Hex_invalid target)
        | Some element ->
            if String.length element <> Dtype.byte_width dtype then
              Err.fail
                (`Element_width
                   {
                     Fault.Size_clash.target;
                     expected = width;
                     actual = Int64.of_int (String.length element);
                   })
            else Err.return (Origin.Fill element))
    | Wire.Generated _ -> Err.fail (`Generated_malformed target)
    | Wire.Inline data -> (
        (* Bounded from the text's length before any byte is decoded, then
           decoded once and compared with what the shape needs. *)
        match Codec.base64_decoded_length data with
        | None -> Err.fail (`Base64_invalid target)
        | Some n when n > limits.max_inline_bytes ->
            over_limit Limits.Inline_bytes
              ~limit:(Int64.of_int limits.max_inline_bytes)
              ~actual:(Int64.of_int n)
        | Some _ -> (
            match Codec.base64_decode data with
            | None -> Err.fail (`Base64_invalid target)
            | Some raw ->
                let actual = Int64.of_int (String.length raw) in
                if Int64.equal actual bytes then Err.return (Origin.Inline raw)
                else
                  Err.fail
                    (`Inline_size
                       { Fault.Size_clash.target; expected = bytes; actual })))
    | Wire.Pack key ->
        if has_pack then Err.return (Origin.Pack key)
        else Err.fail (`Pack_without_source target)
  in
  Err.return { Entry.dtype; origin; sha256; shape = w.shape }

let of_string ?(limits = Limits.default) text =
  let* () =
    if String.length text > limits.max_document_bytes then
      over_limit Limits.Document_bytes
        ~limit:(Int64.of_int limits.max_document_bytes)
        ~actual:(Int64.of_int (String.length text))
    else Err.return ()
  in
  let* w =
    Jsont_bytesrw.decode_string Wire.document_jsont text
    |> Err.import ~pos:__POS__ (fun e -> `Map_json_decode e)
  in
  let* () =
    if w.schema_version <> supported_schema_version then
      Err.fail (`Schema_version w.schema_version)
    else Err.return ()
  in
  let* () =
    match w.unmapped with
    | [] -> Err.return ()
    | names -> Err.fail (`Unmapped names)
  in
  let* graph_sha256 = digest_of w.graph_sha256 in
  let nfiles = List.length w.sources.files in
  let* () =
    if nfiles > limits.max_checkpoint_files then
      over_limit Limits.Checkpoint_files
        ~limit:(Int64.of_int limits.max_checkpoint_files)
        ~actual:(Int64.of_int nfiles)
    else Err.return ()
  in
  let* checkpoint_files = Err.List.map source_of w.sources.files in
  let* graph_owned =
    match w.sources.graph_owned with
    | None -> Err.return None
    | Some p ->
        let+ pin = pin_of p in
        Some pin
  in
  let names =
    List.map (fun (s : Source.t) -> s.pin.name) checkpoint_files
    @ match graph_owned with Some p -> [ p.Pin.name ] | None -> []
  in
  let* () = name_set_unique Fault.Checkpoint_file names in
  let ncaptures = List.length w.tensors in
  let* () =
    if ncaptures > limits.max_captures then
      over_limit Limits.Captures
        ~limit:(Int64.of_int limits.max_captures)
        ~actual:(Int64.of_int ncaptures)
    else Err.return ()
  in
  let* () = name_set_unique Fault.Capture (List.map fst w.tensors) in
  let files = List.map (fun (s : Source.t) -> s.pin.name) checkpoint_files in
  let has_pack = Option.is_some graph_owned in
  let* entries =
    Err.List.map
      (fun (target, tensor) ->
        let+ entry = entry_of ~limits ~files ~has_pack target tensor in
        (target, entry))
      w.tensors
  in
  Err.return
    {
      artifact_id = w.artifact_id;
      checkpoint_files;
      graph_owned;
      graph_sha256;
      model_id = w.model_id;
      tensors = String_map.of_seq (List.to_seq entries);
    }

let find_source t name =
  List.find_opt
    (fun (s : Source.t) -> String.equal s.pin.name name)
    t.checkpoint_files
