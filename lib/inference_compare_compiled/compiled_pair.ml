(* The embedded CompCert route against the direct assembly route on one Native
   graph. The two endpoints run the same graph, so the correspondence is the
   identity map, written as the pass map of a pass that changes nothing and
   coarsened by the same provider. The graphs are described under different
   versions, so each side has its own key. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Core = Inference_compare_core
module Ex = Inference_compare_extract
module P = Inference_compare_pt2
module G = P.Graph_model
module Tid = Graph_ir.Tensor_id

module Dialect = P.Graph_dialect.Make (struct
  type op = Graph_ir.op

  let ops = G.native_ops
end)

type t = {
  compcert : Graph_ir.op G.t;
  assembly : Graph_ir.op G.t;
  mapping : Mapping_artifact.t;
  initial : (Id.Canonical_value.t * L.t) list;
}

let ( let* ) = Result.bind

let graph_text g =
  match Jsont_bytesrw.encode_string Graph_ir.graph_jsont g with
  | Ok s -> s
  | Error m -> m

let identity_export (g : Graph_ir.graph) : Map_export.t =
  {
    nodes =
      List.map
        (fun (n : Graph_ir.node) ->
          { Map_export.Node_cluster.src = [ n.id ]; dst = [ n.id ] })
        g.nodes;
    values =
      List.map
        (fun (id, _) ->
          {
            Map_export.Value_cluster.src = [ id ];
            dst = [ id ];
            claim = Identical;
          })
        (Tid.Map.bindings g.tensors);
  }

(* [bytes] is the exact value of each source of [graph]. *)
let make ~hashers ~algorithm ~(ctx : L.Context.t) ~(graph : Graph_ir.graph)
    ~(bytes : Tid.t -> (string, string) result) : (t, string) result =
  let text = graph_text graph in
  let* compcert =
    G.make ~ops:G.native_ops ~hashers ~algorithm ~version:"compcert"
      ~identity_text:text graph
  in
  let* assembly =
    G.make ~ops:G.native_ops ~hashers ~algorithm ~version:"assembly"
      ~identity_text:text graph
  in
  let digest b =
    match Codecs.Payload_hasher.digest_string hashers algorithm b with
    | Ok d -> Ok d
    | Error _ -> Error "digest backend unavailable"
  in
  let* p =
    P.Pass_map.make ~name:"r" ~digest
      ~src:{ P.Pass_map.model = compcert; bytes }
      ~dst:{ P.Pass_map.model = assembly; bytes }
      ~export:(identity_export graph)
  in
  let* initial =
    let rec go acc = function
      | [] -> Ok (List.rev acc)
      | ((key : Id.Canonical_value.t), read) :: rest -> (
          match
            List.find_opt
              (fun (c : Canonical_spec.t) -> Id.Canonical_value.equal c.key key)
              p.mapping.frontier.canonical_values
          with
          | None -> go acc rest
          | Some c ->
              let* b = read () in
              let* v =
                L.make ctx c.spec b
                |> Result.map_error (fun e ->
                    Format.asprintf "%s: %a"
                      (Id.Canonical_value.to_string key)
                      Codecs.Codec_error.pp (Err.Error.kind e))
              in
              go ((key, v) :: acc) rest)
    in
    go [] p.pins
  in
  Ok { compcert; assembly; mapping = p.mapping; initial }

let compcert_endpoint t : Core.Side_exec.endpoint =
  Endpoint
    {
      runner = Compiled_runners.compcert ();
      build =
        (fun ex -> Ex.Dialect.realize (module Dialect.Dialect) t.compcert ex);
    }

let assembly_endpoint t : Core.Side_exec.endpoint =
  Endpoint
    {
      runner = Compiled_runners.assembly ();
      build =
        (fun ex -> Ex.Dialect.realize (module Dialect.Dialect) t.assembly ex);
    }
