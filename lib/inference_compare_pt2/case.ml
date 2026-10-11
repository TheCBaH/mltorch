(* The pinned inputs of one case: the user tensor and every constant a node
   reads, as exact logical values with their payload digests. *)

module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module F = Pt2_fixture

type t = {
  values : (string * L.t) list;  (** By exporter name. *)
  bytes : (string * string) list;
}

let ( let* ) = Result.bind

let to_bytes (l : F.Logical.t) =
  String.init (Pt2_storage.length l.data) (fun i ->
      Char.chr (Pt2_storage.get_uint8 l.data i))

let logical (t : Pt2_tensor.t) =
  match F.Logical.of_pt2 t with
  | Ok l -> Ok l
  | Error e ->
      Error (Format.asprintf "%a" F.Logical.pp_error (Err.Error.kind e))

(* [inputs] are the user tensors by input name (already in the model's shape,
   batch dimension included); constants are read from the archive. *)
let load ctx (model : Aten_model.t) ~(inputs : (string * Pt2_tensor.t) list) :
    (t, string) result =
  let rec go acc = function
    | [] -> Ok (List.rev acc)
    | (s : Aten_model.source) :: rest ->
        let* tensor =
          match (s.captured, List.assoc_opt s.name inputs) with
          | Some target, _ -> (
              match Pt2_archive.load_captured_tensor model.archive target with
              | Ok t -> Ok t
              | Error e ->
                  Error
                    (Format.asprintf "%s: %a" target Pt2_archive.pp_error
                       (Err.Error.kind e)))
          | None, Some t -> Ok t
          | None, None -> Error ("no value for the input " ^ s.name)
        in
        let* l = logical tensor in
        let b = to_bytes l in
        let* v =
          L.make ctx s.spec b
          |> Result.map_error (fun e ->
              Format.asprintf "%s: %a" s.name Codecs.Codec_error.pp
                (Err.Error.kind e))
        in
        go ((s.name, v, b) :: acc) rest
  in
  let* items = go [] model.sources in
  Ok
    {
      values = List.map (fun (n, v, _) -> (n, v)) items;
      bytes = List.map (fun (n, _, b) -> (n, b)) items;
    }
