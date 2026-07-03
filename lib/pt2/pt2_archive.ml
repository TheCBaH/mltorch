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
}

type error =
  [ `Io of string * string
  | `Zip_open of string * Pt2_zip.error
  | `Read_archive_member of string * Pt2_zip.error
  | `Model_json_decode of string
  | `Weights_config_decode of string
  | `Missing_weight of string
  | `Weight_tensor of string * Pt2_tensor.error
  | `Pt_pickle of string * Pt2_pickle.error ]

let pp_error ppf : error -> unit = function
  | `Io (path, message) ->
      Format.fprintf ppf "failed to read %S: %s" path message
  | `Zip_open (path, error) ->
      Format.fprintf ppf "failed to open zip %S: %a" path Pt2_zip.pp_error error
  | `Read_archive_member (path, error) ->
      Format.fprintf ppf "failed to read archive member %S: %a" path
        Pt2_zip.pp_error error
  | `Model_json_decode msg ->
      Format.fprintf ppf "failed to decode models/model.json: %s" msg
  | `Weights_config_decode msg ->
      Format.fprintf ppf
        "failed to decode data/weights/model_weights_config.json: %s" msg
  | `Missing_weight name -> Format.fprintf ppf "no weight named %S" name
  | `Weight_tensor (name, error) ->
      Format.fprintf ppf "invalid tensor metadata for weight %S: %a" name
        Pt2_tensor.pp_error error
  | `Pt_pickle (path, error) ->
      Format.fprintf ppf "failed to decode tensor pickle %S: %a" path
        Pt2_pickle.pp_error error

let read_file path =
  try Core.return (In_channel.with_open_bin path In_channel.input_all)
  with Sys_error message -> Core.fail (`Io (path, message))

let read_member zip path =
  Pt2_zip.read_rel_required zip path
  |> Core.map_error (fun error -> `Read_archive_member (path, error))

let open_pt2 path =
  let* contents = read_file path in
  let* zip =
    Pt2_zip.of_string contents
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
  Core.return { zip; program; weights }

let program t = t.program
let weights_config t = t.weights

(* Names of all parameters/buffers, in config order. *)
let weight_names t =
  String_map.bindings t.weights.ModelWeightsConfig.config |> List.map fst

(* Load a parameter/buffer by its config name (e.g. "conv1.weight"). *)
let load_weight t name =
  match String_map.find_opt name t.weights.ModelWeightsConfig.config with
  | None -> Core.fail (`Missing_weight name)
  | Some (e : WeightEntry.t) ->
      let* data = read_member t.zip ("data/weights/" ^ e.path_name) in
      Pt2_tensor.of_meta e.tensor_meta ~data:(Bytes.of_string data)
      |> Core.map_error (fun error -> `Weight_tensor (name, error))

(* Load a standalone `.pt` tensor file (a sample input image, or the archive's
   own data/sample_inputs/model.pt extracted to a path). *)
let load_pt path =
  let* contents = read_file path in
  let* zip =
    Pt2_zip.of_string contents
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
