(* High-level access to a `.pt2` model archive: the exported graph, the weight
   index, and on-demand loading of weight / sample-input tensors. *)

open Schema_runtime
open Pytorch_types
open Pytorch_weights_config
open Core.Syntax

type t = {
  zip : Pt2_zip.t;
  program : ExportedProgram.t;
  weights : ModelWeightsConfig.t;
  constants : ModelWeightsConfig.t;
}

type error =
  [ `Io of string * string
  | `Io_too_large of string * int64
  | `Zip_open of string * Pt2_zip.error
  | `Read_archive_member of string * Pt2_zip.error
  | `Model_json_decode of string
  | `Weights_config_decode of string
  | `Constants_config_decode of string
  | `Missing_weight of string
  | `Missing_captured_tensor of string
  | `Weight_tensor of string * Pt2_tensor.error
  | `Pt_pickle of string * Pt2_pickle.error ]

let pp_error ppf : error -> unit = function
  | `Io (path, message) -> Fmt.pf ppf "failed to read %S: %s" path message
  | `Io_too_large (path, limit) ->
      Fmt.pf ppf "%S is larger than %Ld bytes" path limit
  | `Zip_open (path, error) ->
      Fmt.pf ppf "failed to open zip %S: %a" path Pt2_zip.pp_error error
  | `Read_archive_member (path, error) ->
      Fmt.pf ppf "failed to read archive member %S: %a" path Pt2_zip.pp_error
        error
  | `Model_json_decode msg ->
      Fmt.pf ppf "failed to decode models/model.json: %s" msg
  | `Weights_config_decode msg ->
      Fmt.pf ppf "failed to decode data/weights/model_weights_config.json: %s"
        msg
  | `Constants_config_decode msg ->
      Fmt.pf ppf
        "failed to decode data/constants/model_constants_config.json: %s" msg
  | `Missing_weight name -> Fmt.pf ppf "no weight named %S" name
  | `Missing_captured_tensor name ->
      Fmt.pf ppf "no captured tensor named %S" name
  | `Weight_tensor (name, error) ->
      Fmt.pf ppf "invalid tensor metadata for weight %S: %a" name
        Pt2_tensor.pp_error error
  | `Pt_pickle (path, error) ->
      Fmt.pf ppf "failed to decode tensor pickle %S: %a" path
        Pt2_pickle.pp_error error

(* CHECKPOINT 1 — before the OCaml string exists.

   [In_channel.input_all] on an attacker-chosen path allocates whatever is
   there, so the size is taken from the channel and REJECTED FIRST; only then is
   that many bytes read. A bare [input_all] behind a later length check would
   have done the allocation the check exists to prevent.

   The two-step is deliberate about the race: the declared length is what gets
   bounded, and [really_input_string] then reads exactly that many bytes, so a
   file that grows between the two cannot make us read more than we approved.
   (One that SHRINKS yields [None], reported as a short read.) *)

(* The ceiling no [~max_bytes] may exceed, because the read narrows an [int64]
   length to the [int] that [really_input_string] takes — and that [int] is
   32-bit under js_of_ocaml. A caller-supplied bound is therefore not enough on
   its own: a generous one would push the narrowing outside the JS domain, which
   is the "narrow only after bounding" rule with the bound set too high to do any
   bounding. 512MB is inside both backends' string domains and two orders of
   magnitude above the largest catalog model. *)
let max_file_bytes = 0x2000_0000L
let default_max_file_bytes = max_file_bytes

let effective_max_bytes max_bytes =
  (* Named, and therefore testable. Inline, the clamp could only be observed by
     presenting a 512MB file, so the one case that matters — a caller asking for
     more than the domain allows — had no cheap fixture and would have been
     "tested" by a fixture that could not fail. *)
  if Int64.compare max_bytes max_file_bytes > 0 then max_file_bytes
  else max_bytes

let read_file ?(max_bytes = default_max_file_bytes) path =
  let ceiling = effective_max_bytes max_bytes in
  try
    In_channel.with_open_bin path (fun ic ->
        let length = In_channel.length ic in
        if Int64.compare length ceiling > 0 then
          Core.fail (`Io_too_large (path, ceiling))
        else
          (* Narrowed only now that it is bounded by [ceiling], which is at most
             [max_file_bytes] — in range on both backends by construction rather
             than by an argument about the caller. *)
          match In_channel.really_input_string ic (Int64.to_int length) with
          | Some s -> Core.return s
          | None -> Core.fail (`Io (path, "short read")))
  with Sys_error message -> Core.fail (`Io (path, message))

let read_member zip path =
  Pt2_zip.read_rel_required zip path
  |> Core.map_error (fun error -> `Read_archive_member (path, error))

(* Decoding is separated from reading so an archive that is already in memory
   never has to reach a filesystem: a JS build has no useful one, and a browser
   receives the bytes from a fetch or a file picker. [~name] is only a label for
   the error payloads -- nothing opens it. *)
let of_string ?limits ~name contents =
  let path = name in
  let* zip =
    Pt2_zip.of_string ?limits contents
    |> Core.map_error (fun error -> `Zip_open (path, error))
  in
  let* program_json = read_member zip "models/model.json" in
  let* program =
    match Jsont_bytesrw.decode_string ExportedProgram.jsont program_json with
    | Ok program -> Core.return program
    | Error e -> Core.fail (`Model_json_decode e)
  in
  let* weights_json =
    read_member zip "data/weights/model_weights_config.json"
  in
  let* weights =
    match Jsont_bytesrw.decode_string ModelWeightsConfig.jsont weights_json with
    | Ok weights -> Core.return weights
    | Error e -> Core.fail (`Weights_config_decode e)
  in
  let* constants_data =
    Pt2_zip.read_rel zip "data/constants/model_constants_config.json"
    |> Core.map_error (fun error ->
        `Read_archive_member
          ("data/constants/model_constants_config.json", error))
  in
  let* constants =
    match constants_data with
    | None -> Core.return { ModelWeightsConfig.config = String_map.empty }
    | Some json -> (
        match Jsont_bytesrw.decode_string ModelWeightsConfig.jsont json with
        | Ok constants -> Core.return constants
        | Error e -> Core.fail (`Constants_config_decode e))
  in
  Core.return { zip; program; weights; constants }

let open_pt2 ?limits ?max_bytes path =
  let* contents = read_file ?max_bytes path in
  of_string ?limits ~name:path contents

let program t = t.program
let weights_config t = t.weights
let constants_config t = t.constants

(* Names of all parameters/buffers, in config order. *)
let weight_names t =
  String_map.bindings t.weights.ModelWeightsConfig.config |> List.map fst

let constant_names t =
  String_map.bindings t.constants.ModelWeightsConfig.config |> List.map fst

let load_entry t ~dir name (e : WeightEntry.t) =
  let* data = read_member t.zip (dir ^ "/" ^ e.path_name) in
  Pt2_tensor.of_meta e.tensor_meta ~data:(Bytes.of_string data)
  |> Core.map_error (fun error -> `Weight_tensor (name, error))

(* Load a parameter/buffer by its config name (e.g. "conv1.weight"). *)
let load_weight t name =
  match String_map.find_opt name t.weights.ModelWeightsConfig.config with
  | None -> Core.fail (`Missing_weight name)
  | Some e -> load_entry t ~dir:"data/weights" name e

(* Resolve a native inference [Constant { target }] across both payload roots.
   PT2 guarantees the target naming; the archive location is deliberately not
   part of native IR's inference-only source classification. *)
let load_captured_tensor t name =
  match String_map.find_opt name t.weights.ModelWeightsConfig.config with
  | Some e -> load_entry t ~dir:"data/weights" name e
  | None -> (
      match String_map.find_opt name t.constants.ModelWeightsConfig.config with
      | Some e -> load_entry t ~dir:"data/constants" name e
      | None -> Core.fail (`Missing_captured_tensor name))

(* A standalone `.pt` tensor (a sample input image, or the archive's own
   data/sample_inputs/model.pt) already in memory. Same split as [of_string]. *)
let pt_of_string ?limits ~name contents =
  let path = name in
  let* zip =
    Pt2_zip.of_string ?limits contents
    |> Core.map_error (fun error -> `Zip_open (path, error))
  in
  let* data_pkl = read_member zip "data.pkl" in
  let* rb =
    Pt2_pickle.parse_tensor data_pkl
    |> Core.map_error (fun error -> `Pt_pickle (path, error))
  in
  let* data = read_member zip ("data/" ^ rb.Pt2_pickle.storage_key) in
  Core.return
    {
      Pt2_tensor.dtype = rb.Pt2_pickle.dtype;
      sizes = rb.Pt2_pickle.sizes;
      strides = rb.Pt2_pickle.strides;
      storage_offset = rb.Pt2_pickle.storage_offset;
      data = Bytes.of_string data;
    }

(* Load a standalone `.pt` tensor file from disk. *)
let load_pt ?limits ?max_bytes path =
  let* contents = read_file ?max_bytes path in
  pt_of_string ?limits ~name:path contents
