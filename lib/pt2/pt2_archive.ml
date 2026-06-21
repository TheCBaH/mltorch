(* High-level access to a `.pt2` model archive: the exported graph, the weight
   index, and on-demand loading of weight / sample-input tensors. *)

open Schema_runtime
open Pytorch_types
open Pytorch_weights_config

type t = {
  zip : Pt2_zip.t;
  program : ExportedProgram.t;
  weights : ModelWeightsConfig.t;
}

let decode_exn jsont s =
  match Jsont_bytesrw.decode_string jsont s with
  | Ok v -> v
  | Error e -> failwith ("Pt2_archive: JSON decode: " ^ e)

let read_file path = In_channel.with_open_bin path In_channel.input_all

let open_pt2 path =
  let zip = Pt2_zip.of_string (read_file path) in
  let program =
    decode_exn ExportedProgram.jsont
      (Pt2_zip.read_rel_exn zip "models/model.json")
  in
  let weights =
    decode_exn ModelWeightsConfig.jsont
      (Pt2_zip.read_rel_exn zip "data/weights/model_weights_config.json")
  in
  { zip; program; weights }

let program t = t.program
let weights_config t = t.weights

(* Names of all parameters/buffers, in config order. *)
let weight_names t =
  String_map.bindings t.weights.ModelWeightsConfig.config |> List.map fst

(* Load a parameter/buffer by its config name (e.g. "conv1.weight"). *)
let load_weight t name =
  match String_map.find_opt name t.weights.ModelWeightsConfig.config with
  | None -> failwith ("Pt2_archive: no weight named " ^ name)
  | Some (e : WeightEntry.t) ->
      let data =
        Bytes.of_string
          (Pt2_zip.read_rel_exn t.zip ("data/weights/" ^ e.path_name))
      in
      Pt2_tensor.of_meta e.tensor_meta ~data

(* Load a standalone `.pt` tensor file (a sample input image, or the archive's
   own data/sample_inputs/model.pt extracted to a path). *)
let load_pt path =
  let zip = Pt2_zip.of_string (read_file path) in
  let rb = Pt2_pickle.parse_tensor (Pt2_zip.read_rel_exn zip "data.pkl") in
  let data =
    Bytes.of_string
      (Pt2_zip.read_rel_exn zip ("data/" ^ rb.Pt2_pickle.storage_key))
  in
  {
    Pt2_tensor.dtype = rb.Pt2_pickle.dtype;
    sizes = rb.Pt2_pickle.sizes;
    strides = rb.Pt2_pickle.strides;
    storage_offset = rb.Pt2_pickle.storage_offset;
    data;
  }
