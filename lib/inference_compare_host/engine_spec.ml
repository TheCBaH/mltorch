(* Engine configuration as it appears in an experiment, and its instantiation
   into a coordinator endpoint. A new engine is one more case here; the
   coordinator never sees the name. *)

open Inference_compare_schema
module W = Wire
module R = Inference_compare_engines.Synthetic_runner
module Core = Inference_compare_core

(* A pinned exported model: the path (relative to the experiment, or absolute)
   and the SHA-256 of the file. *)
type pt2 = { model : string; sha256 : string }
type t = Aten of pt2 | Native of pt2 | Synthetic of R.config

let decode_pt2 o =
  let ( let* ) = W.( let* ) in
  let* model = W.Obj.required o "model" W.string in
  let* sha256 = W.Obj.required o "sha256" W.string in
  Ok { model; sha256 }

let encode_pt2 tag (p : pt2) =
  W.tagged tag
    [
      W.field "model" W.enc_string p.model;
      W.field "sha256" W.enc_string p.sha256;
    ]

let node_delta : (string * int64) W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* node = W.Obj.required o "node" W.string in
      let* delta = W.Obj.required o "delta" W.int64 in
      Ok (node, delta))

let ordinal_delta : (int * int64) W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* ordinal = W.Obj.required o "ordinal" W.nonneg_int in
      let* delta = W.Obj.required o "delta" W.int64 in
      Ok (ordinal, delta))

let decode : t W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "aten" ->
          let* p = decode_pt2 o in
          Ok (Aten p)
      | "native" ->
          let* p = decode_pt2 o in
          Ok (Native p)
      | "synthetic" ->
          let* name = W.Obj.required o "name" W.string in
          let* offsets = W.Obj.required o "offsets" (W.list node_delta) in
          let* corrupt_imports =
            W.Obj.required o "corrupt_imports" (W.list ordinal_delta)
          in
          let* refuse_ops = W.Obj.required o "refuse_ops" (W.list W.string) in
          let* fail_nodes = W.Obj.required o "fail_nodes" (W.list W.string) in
          let* fail_imports =
            W.Obj.required o "fail_imports" (W.list W.nonneg_int)
          in
          let* reuse_output_storage =
            W.Obj.required o "reuse_output_storage" W.bool
          in
          let* async_polls = W.Obj.required o "async_polls" W.nonneg_int in
          Ok
            (Synthetic
               {
                 R.name;
                 offsets;
                 corrupt_imports;
                 refuse_ops;
                 fail_nodes;
                 fail_imports;
                 reuse_output_storage;
                 async_polls;
               })
      | tag -> W.unknown_tag ~kind:"engine" tag)

let encode = function
  | Aten p -> encode_pt2 "aten" p
  | Native p -> encode_pt2 "native" p
  | Synthetic c ->
      W.tagged "synthetic"
        [
          W.field "name" W.enc_string c.name;
          W.field "offsets"
            (W.enc_list (fun (n, d) ->
                 W.obj
                   [
                     W.field "node" W.enc_string n;
                     W.field "delta" W.enc_int64 d;
                   ]))
            c.offsets;
          W.field "corrupt_imports"
            (W.enc_list (fun (n, d) ->
                 W.obj
                   [
                     W.field "ordinal" W.enc_int n;
                     W.field "delta" W.enc_int64 d;
                   ]))
            c.corrupt_imports;
          W.field "refuse_ops" (W.enc_list W.enc_string) c.refuse_ops;
          W.field "fail_nodes" (W.enc_list W.enc_string) c.fail_nodes;
          W.field "fail_imports" (W.enc_list W.enc_int) c.fail_imports;
          W.field "reuse_output_storage" W.enc_bool c.reuse_output_storage;
          W.field "async_polls" W.enc_int c.async_polls;
        ]

type instance = {
  endpoint : Core.Side_exec.endpoint;
  identity : Engine_identity.t;
  attributes : Node_ref.t -> string;
      (** The attribute text of an original node, folded into extraction
          digests; empty for an engine whose graphs carry no attributes. *)
}

(* The endpoint, the engine's identity and its node attributes. A model-backed
   engine must be the one the experiment's graph describes. *)
let instantiate (t : t) ~hashers ~algorithm ~base ~(source : Graph_desc.t) :
    (instance, string) result =
  let pt2 kind (p : pt2) =
    match
      Inference_compare_pt2.Route.load kind ~hashers ~algorithm
        ~path:(File_io.resolve ~base p.model)
        ~pin:p.sha256
    with
    | Error _ as e -> e
    | Ok r when Graph_key.equal r.graph source.key ->
        Ok
          {
            endpoint = r.endpoint;
            identity = r.identity;
            attributes = r.attributes;
          }
    | Ok _ -> Error (p.model ^ " is not the graph the experiment describes")
  in
  match t with
  | Aten p -> pt2 Inference_compare_pt2.Route.Aten p
  | Native p -> pt2 Inference_compare_pt2.Route.Native p
  | Synthetic cfg ->
      let runner, _ = R.create cfg in
      let module M = (val runner) in
      let identity = M.identity () in
      let module D =
        (val Inference_compare_engines.Synthetic_dialect.make ()
            : Inference_compare_extract.Dialect.S with type graph = Graph_desc.t
             and type body = Graph_desc.t)
      in
      Ok
        {
          endpoint =
            Core.Side_exec.Endpoint
              {
                runner;
                build =
                  (fun ex ->
                    Inference_compare_extract.Dialect.realize
                      (module D)
                      source ex);
              };
          identity;
          attributes = (fun _ -> "");
        }
