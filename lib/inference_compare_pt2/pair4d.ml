(* The ATen / Native4D pair for one exported model and one case: the ATen to
   Native import correspondence, the canonicalization pass map and the
   four-axis lowering map composed into one mapping artifact between the
   exporter graph and the Native4D graph. A model outside the four-axis domain
   does not load: there is no fall back to the Native route. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Core = Inference_compare_core
module Ex = Inference_compare_extract
module Compose = Inference_compare_compose.Compose

type t = {
  base : Pair.t;  (** The ATen / Native pair the Native4D one builds on. *)
  four : Native4d_model.t;
  mapping : Mapping_artifact.t;
  pins : (Id.Canonical_value.t * L.t) list;
      (** Constants that exist only in the Native4D graph's own sources. *)
}

let ( let* ) = Result.bind

let composed ~hashers ~algorithm first second =
  Compose.compose ~hashers ~algorithm first second
  |> Result.map_error (fun e ->
      Format.asprintf "composition: %a"
        (Err.Error.pp_kind Inference_compare_compose.Compose_error.pp)
        e)

let load ~hashers ~algorithm ~path ~(inputs : (string * Pt2_tensor.t) list) :
    (t, string) result =
  let* base = Pair.load ~hashers ~algorithm ~path ~inputs in
  let native = base.native in
  let* four =
    Native4d_model.make ~hashers ~algorithm ~ctx:base.case_ctx
      ~archive:base.aten.archive ~native ~source_bytes:(Pair.source_bytes base)
  in
  let* canonicalize = Lazy.force four.canonicalize in
  let* lower = Lazy.force four.lower in
  let* first =
    composed ~hashers ~algorithm base.import.mapping canonicalize.mapping
  in
  let* mapping = composed ~hashers ~algorithm first lower.mapping in
  let* pins =
    let rec go acc = function
      | [] -> Ok (List.rev acc)
      | ((key : Id.Canonical_value.t), read) :: rest -> (
          match
            List.find_opt
              (fun (c : Canonical_spec.t) -> Id.Canonical_value.equal c.key key)
              mapping.frontier.canonical_values
          with
          | None -> go acc rest
          | Some _
            when List.exists
                   (fun (k, _) -> Id.Canonical_value.equal k key)
                   (Pair.initial base) ->
              go acc rest
          | Some c ->
              let* bytes = read () in
              let* v =
                L.make base.case_ctx c.spec bytes
                |> Result.map_error (fun e ->
                    Format.asprintf "pinned %s: %a"
                      (Id.Canonical_value.to_string key)
                      Codecs.Codec_error.pp (Err.Error.kind e))
              in
              go ((key, v) :: acc) rest)
    in
    go [] (canonicalize.pins @ lower.pins)
  in
  Ok { base; four; mapping; pins }

let mapping t = t.mapping
let left_graph t = t.base.aten.desc
let right_graph t = t.four.n4.desc
let initial t = Pair.initial t.base @ t.pins

let node_attributes t side (r : Node_ref.t) =
  match (side : Side.t) with
  | Left -> Aten_model.attributes_of t.base.aten r
  | Right -> Graph_model.attributes_of t.four.n4 r

let aten_endpoint t = Pair.aten_endpoint t.base

let native4d_endpoint t : Core.Side_exec.endpoint =
  Endpoint
    {
      runner = Native4d_runner.create ();
      build =
        (fun ex ->
          Ex.Dialect.realize (module Native4d_dialect.Dialect) t.four.n4 ex);
    }

let aten_identity () = Aten_runner.identity ()
let native4d_identity () = Native4d_runner.identity ()
