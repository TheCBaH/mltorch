open Schema_runtime
open Pytorch_weights_config
open Err.Syntax

module Checkpoint_map = struct
  module Source = struct
    type t = {
      filename : string;
      repo_id : string;
      revision : string;
      sha256 : string;
      size : int64;
      url : string;
    }

    let jsont =
      Jsont.Object.map ~kind:"Source"
        (fun filename repo_id revision sha256 size url ->
          { filename; repo_id; revision; sha256; size; url })
      |> Jsont.Object.mem "filename" Jsont.string
      |> Jsont.Object.mem "repo_id" Jsont.string
      |> Jsont.Object.mem "revision" Jsont.string
      |> Jsont.Object.mem "sha256" Jsont.string
      |> Jsont.Object.mem "size" Jsont.int64
      |> Jsont.Object.mem "url" Jsont.string
      |> Jsont.Object.finish
  end

  module Entry = struct
    type t = { dtype : string; key : string; shape : int list }

    let jsont =
      Jsont.Object.map ~kind:"Entry" (fun dtype key shape ->
          { dtype; key; shape })
      |> Jsont.Object.mem "dtype" Jsont.string
      |> Jsont.Object.mem "key" Jsont.string
      |> Jsont.Object.mem "shape" (Jsont.list Jsont.int)
      |> Jsont.Object.finish
  end

  type t = {
    schema_version : int;
    source : Source.t;
    tensors : Entry.t String_map.t;
    unmapped : string list;
  }

  let jsont =
    Jsont.Object.map ~kind:"Checkpoint_map"
      (fun schema_version source tensors unmapped ->
        { schema_version; source; tensors; unmapped })
    |> Jsont.Object.mem "schema_version" Jsont.int
    |> Jsont.Object.mem "source" Source.jsont
    |> Jsont.Object.mem "tensors" (Jsont.Object.as_string_map Entry.jsont)
    |> Jsont.Object.mem "unmapped" (Jsont.list Jsont.string)
    |> Jsont.Object.finish
end

module Mismatch = struct
  type t =
    | Checkpoint_dtype of { checkpoint : Safetensors.Dtype.t; map : string }
    | Checkpoint_shape of { checkpoint : int64 list; map : int list }
    | Graph_dtype of { graph : Pt2_dtype.t; map : string }
    | Graph_layout
    | Graph_shape of { graph : int list; map : int list }
end

module Source_mismatch = struct
  type t =
    | Sha256 of { actual : string option; expected : string }
    | Size of { actual : int64; expected : int64 }
end

type error =
  [ `Map_decode of string
  | `Mismatch of string * Mismatch.t
  | `Missing_in_checkpoint of string * string
  | `Schema_version of int
  | `Source_mismatch of Source_mismatch.t
  | `Unmapped_constants of string list
  | `Unmapped_tensor of string
  | Pt2_tensor.error ]

let pp_ints = Fmt.brackets (Fmt.list ~sep:Fmt.semi Fmt.int)
let pp_int64s = Fmt.brackets (Fmt.list ~sep:Fmt.semi Fmt.int64)

let pp_mismatch ppf : Mismatch.t -> unit = function
  | Checkpoint_dtype { checkpoint; map } ->
      Fmt.pf ppf "checkpoint dtype %s, map says %s"
        (Safetensors.Dtype.to_string checkpoint)
        map
  | Checkpoint_shape { checkpoint; map } ->
      Fmt.pf ppf "checkpoint shape %a, map says %a" pp_int64s checkpoint pp_ints
        map
  | Graph_dtype { graph; map } ->
      Fmt.pf ppf "graph dtype %s, map says %s" (Pt2_dtype.to_string graph) map
  | Graph_layout ->
      Fmt.string ppf "graph tensor is not a dense row-major buffer at offset 0"
  | Graph_shape { graph; map } ->
      Fmt.pf ppf "graph shape %a, map says %a" pp_ints graph pp_ints map

let pp_error ppf : error -> unit = function
  | `Map_decode msg -> Fmt.pf ppf "failed to decode safetensors.json: %s" msg
  | `Mismatch (name, m) -> Fmt.pf ppf "tensor %S: %a" name pp_mismatch m
  | `Missing_in_checkpoint (name, key) ->
      Fmt.pf ppf "tensor %S: checkpoint has no tensor %S" name key
  | `Schema_version v ->
      Fmt.pf ppf "unsupported safetensors.json schema_version %d" v
  | `Source_mismatch (Sha256 { actual; expected }) ->
      Fmt.pf ppf "checkpoint sha256 is %a, safetensors.json pins %s"
        Fmt.(option ~none:(any "unknown") string)
        actual expected
  | `Source_mismatch (Size { actual; expected }) ->
      Fmt.pf ppf "checkpoint is %Ld bytes, safetensors.json pins %Ld" actual
        expected
  | `Unmapped_constants names ->
      Fmt.pf ppf "the checkpoint lacks %d captured tensor(s): %a"
        (List.length names)
        Fmt.(list ~sep:(any ", ") string)
        names
  | `Unmapped_tensor name ->
      Fmt.pf ppf "captured tensor %S is not in safetensors.json" name
  | #Pt2_tensor.error as e -> Pt2_tensor.pp_error ppf e

let check_source (source : Checkpoint_map.Source.t) ~etag ~size =
  if etag <> Some source.sha256 then
    Err.fail
      (`Source_mismatch
         (Source_mismatch.Sha256 { actual = etag; expected = source.sha256 }))
  else if not (Int64.equal size source.size) then
    Err.fail
      (`Source_mismatch
         (Source_mismatch.Size { actual = size; expected = source.size }))
  else Err.return ()

let supported_schema_version = 1

let map_of_string json =
  let* map =
    Jsont_bytesrw.decode_string Checkpoint_map.jsont json
    |> Err.import ~pos:__POS__ (fun e -> `Map_decode e)
  in
  if map.Checkpoint_map.schema_version <> supported_schema_version then
    Err.fail (`Schema_version map.schema_version)
  else Err.return map

(* The safetensors dtype of each Pt2_dtype; [None] where the checkpoint format
   has no equal. A bare mismatch is reported rather than a conversion tried. *)
let safetensors_dtype_of_pt2 : Pt2_dtype.t -> Safetensors.Dtype.t = function
  | Bool -> Bool
  | Float32 -> F32
  | Float64 -> F64
  | Int16 -> I16
  | Int32 -> I32
  | Int64 -> I64
  | Int8 -> I8
  | UInt8 -> U8

(* What the map and the graph's config must agree on, without a checkpoint:
   dtype, shape, and that the graph's tensor is a dense row-major buffer, which
   is all a checkpoint tensor can be. *)
let check_graph_tensor name (meta : Pytorch_types.TensorMeta.t)
    (entry : Checkpoint_map.Entry.t) : (unit, [> error ]) Err.t =
  let open Mismatch in
  let mismatch m = Err.fail (`Mismatch (name, m)) in
  let* graph =
    Pt2_tensor.of_meta meta ~data:Pt2_storage.empty
    |> Err.map_error ~pos:__POS__ (function #Pt2_tensor.error as e -> e)
  in
  let* () =
    match Safetensors.Dtype.of_string entry.dtype with
    | Ok d when d = safetensors_dtype_of_pt2 graph.dtype -> Err.return ()
    | Ok _ | Error _ ->
        mismatch (Graph_dtype { graph = graph.dtype; map = entry.dtype })
  in
  let* () =
    if graph.sizes = entry.shape then Err.return ()
    else mismatch (Graph_shape { graph = graph.sizes; map = entry.shape })
  in
  (* [is_contiguous] folds the sizes with an overflow check that raises for a
     tensor no buffer could hold; that is a layout the checkpoint cannot match
     either, so it reports as one. *)
  match Pt2_tensor.is_contiguous graph with
  | true -> Err.return ()
  | false -> mismatch Graph_layout
  | exception Invalid_argument _ -> mismatch Graph_layout

let check_graph ~(map : Checkpoint_map.t) ~weights ~constants :
    (unit, [> error ]) Err.t =
  let* () =
    match map.unmapped with
    | [] -> Err.return ()
    | names -> Err.fail (`Unmapped_constants names)
  in
  let check config =
    Err.List.iter
      (fun (name, (e : WeightEntry.t)) ->
        match String_map.find_opt name map.tensors with
        | None -> Err.fail (`Unmapped_tensor name)
        | Some entry -> check_graph_tensor name e.tensor_meta entry)
      (String_map.bindings config.ModelWeightsConfig.config)
  in
  let* () = check weights in
  check constants

(* The map against the checkpoint's own header: the key exists with the dtype
   and shape the map claims. *)
let check_checkpoint ~(map : Checkpoint_map.t) memory : (unit, [> error ]) Err.t
    =
  let open Mismatch in
  Err.List.iter
    (fun (name, (entry : Checkpoint_map.Entry.t)) ->
      let mismatch m = Err.fail (`Mismatch (name, m)) in
      match
        Safetensors.Index.find (Safetensors.Memory.index memory) entry.key
      with
      | None -> Err.fail (`Missing_in_checkpoint (name, entry.key))
      | Some tensor ->
          let checkpoint_dtype = Safetensors.Tensor.dtype tensor in
          let* () =
            match Safetensors.Dtype.of_string entry.dtype with
            | Ok d when d = checkpoint_dtype -> Err.return ()
            | Ok _ | Error _ ->
                mismatch
                  (Checkpoint_dtype
                     { checkpoint = checkpoint_dtype; map = entry.dtype })
          in
          if
            List.equal Int64.equal
              (Safetensors.Tensor.shape tensor)
              (List.map Int64.of_int entry.shape)
          then Err.return ()
          else
            mismatch
              (Checkpoint_shape
                 {
                   checkpoint = Safetensors.Tensor.shape tensor;
                   map = entry.shape;
                 }))
    (String_map.bindings map.tensors)

let of_parts ~(map : Checkpoint_map.t) ~program ~weights ~constants memory =
  let* () = check_graph ~map ~weights ~constants in
  let* () = check_checkpoint ~map memory in
  let load name =
    match String_map.find_opt name map.tensors with
    | None -> Error (Fmt.str "%S is not in safetensors.json" name)
    | Some entry -> (
        match Safetensors.Memory.tensor_view memory entry.key with
        | Ok view -> Ok view
        | Error e -> Error (Fmt.str "%a" Safetensors.Error.pp e))
  in
  Err.return (Pt2_archive.of_parts ~program ~weights ~constants ~load)
