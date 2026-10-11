(* The Native4D side of the pair. The lowered Native graph goes through the
   canonical passes and then the checked four-axis lowering; each step is an
   ordinary graph with its own description, and each step's own correspondence
   (the pass map, the lowering map) is written as a mapping artifact between
   the two graphs it relates. A model outside the four-axis domain is a
   refusal here, never a fall back to Native. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module G = Graph_model
module Tid = Graph_ir.Tensor_id

type t = {
  n0 : Graph_ir.op G.t;  (** The lowered Native graph, as {!Native_model}. *)
  n1 : Graph_ir.op G.t;  (** After the canonical passes. *)
  canonical_bytes : Tid.t -> (string, string) result;
      (** The exact value of each source of the canonical graph. *)
  n4 : Native4d.Op.t G.t;  (** The four-axis graph. *)
  constants : (Tensor.packed Tid.Map.t, string) result Lazy.t;
      (** Every constant the four-axis graph reads, materialized on demand. *)
  canonicalize : (Pass_map.t, string) result Lazy.t;
      (** Lowered Native to canonical Native; read on demand because it hashes
          every source value, which an endpoint alone does not need. *)
  lower : (Pass_map.t, string) result Lazy.t;
      (** Canonical Native to Native4D. *)
}

let ( let* ) = Result.bind
let pp_err pp e = Format.asprintf "%a" pp (Err.Error.kind e)

let digest_of ~hashers ~algorithm bytes =
  match Codecs.Payload_hasher.digest_string hashers algorithm bytes with
  | Ok d -> Ok d
  | Error _ -> Error "digest backend unavailable"

(* A graph outside the four-axis domain is a refusal, never a Native run. *)
let refusal e =
  pp_err
    (fun ppf e ->
      Fmt.pf ppf "outside the Native4D domain: %a" Native4d.Error.pp e)
    e

let identity_text pp g = Format.asprintf "%a" pp g

(* Constants a graph reads, materialized: those the passes bound plus those
   they computed, which exist only symbolically until resolved. *)
let materialize archive ~constants ~store =
  let* store, _ =
    Const_ssa_materialize.materialize
      (Native_interp.capture_resolver archive)
      store
    |> Result.map_error (pp_err Const_ssa_materialize.pp_error)
  in
  Ok
    (Tid.Map.union
       (fun _ _ planned -> Some planned)
       constants
       (Constant_store.materialized store))

(* The exporter name of a source of the lowered graph. *)
let lowered_name (capture : Native_interp.Capture.t) id =
  List.find_map
    (fun (name, t) -> if Tid.equal t id then Some name else None)
    capture.sources

(* The name of a destination source, read off the map that produced it: the
   name of the source value it is a cluster with. *)
let through (export : Map_export.t) upstream id =
  List.find_map
    (fun (c : Map_export.Value_cluster.t) ->
      if List.exists (Tid.equal id) c.dst then List.find_map upstream c.src
      else None)
    export.values

(* A source is a materialized constant or a case value, found by the exporter
   name the importer gave its original tensor. *)
let bytes_of ~graph ~source_bytes ~name_of ctx (g : _ Graph_common.Graph.t)
    constants id =
  match Tid.Map.find_opt id constants with
  | Some packed ->
      let* spec = G.spec_of g id in
      let* v = Conv.logical_of_native ctx spec packed in
      let b = L.contents v in
      L.drop v;
      Ok b
  | None -> (
      match name_of id with
      | Some name -> source_bytes name
      | None ->
          Error
            (Printf.sprintf "no value for the source t%d of the %s graph"
               (Tid.to_int id) graph))

(* The canonical Native graph does not need the four-axis lowering, so a model
   Native supports but Native4D refuses still has its [n0] and [n1] views and
   their pass map. The lowering is tried only when [lowering] is forced. *)
module Canonical = struct
  type four = t

  type t = {
    n0 : Graph_ir.op G.t;
    n1 : Graph_ir.op G.t;
    canonical_bytes : Tid.t -> (string, string) result;
    canonicalize : (Pass_map.t, string) result Lazy.t;
    lowering : (four, string) result Lazy.t;
        (** The four-axis extension; a graph outside the domain is its [Error],
            never a fall back to Native. *)
  }
end

(* [source_bytes] is the value of an original source by its exporter name: the
   case holds the user inputs, which the archive does not. *)
let canonical ~hashers ~algorithm ~(ctx : L.Context.t)
    ~(archive : Pt2_archive.t) ~(native : Native_model.t)
    ~(source_bytes : string -> (string, string) result) :
    (Canonical.t, string) result =
  let lowered = native.lowered in
  let* constants0 =
    Native_interp.preload archive lowered
    |> Result.map_error (pp_err Native_interp.pp_error)
  in
  match
    Native_interp.transform_lowered ~constants:constants0 lowered
      ~passes:[ Pipeline.canonical ~fold:true ]
  with
  | Error e -> Error (pp_err Native_interp.pp_error e)
  | Ok (Native_interp.Transformed t) ->
      let graph_text g =
        match Jsont_bytesrw.encode_string Graph_ir.graph_jsont g with
        | Ok s -> s
        | Error m -> m
      in
      let* n0 =
        G.make ~ops:G.native_ops ~hashers ~algorithm ~version:"pt2-lowered"
          ~identity_text:(graph_text native.graph) native.graph
      in
      let* () =
        if Graph_key.equal n0.key native.key then Ok ()
        else
          Error "the lowered graph does not describe as the Native model does"
      in
      let* n1 =
        G.make ~ops:G.native_ops ~hashers ~algorithm ~version:"canonical"
          ~identity_text:(graph_text t.graph) t.graph
      in
      let constants1 =
        lazy
          (materialize archive ~constants:t.constants ~store:t.constant_store)
      in
      let digest = digest_of ~hashers ~algorithm in
      let name0 = lowered_name native.capture in
      let name1 = through t.map_export name0 in
      let n0_bytes =
        bytes_of ~graph:"lowered" ~source_bytes ~name_of:name0 ctx n0.graph
          constants0
      in
      let n1_bytes id =
        let* c = Lazy.force constants1 in
        bytes_of ~graph:"canonical" ~source_bytes ~name_of:name1 ctx n1.graph c
          id
      in
      let canonicalize =
        lazy
          (Pass_map.make ~name:"c" ~digest
             ~src:{ Pass_map.model = n0; bytes = n0_bytes }
             ~dst:{ Pass_map.model = n1; bytes = n1_bytes }
             ~export:t.map_export)
      in
      let lowering =
        lazy
          (match Snapshot.create t.graph with
          | Error _ -> Error "the canonical graph is not a valid snapshot"
          | Ok (Snapshot.Pack src) -> (
              match
                Native4d.Lower.convert ~constants:t.constants
                  ~constant_store:t.constant_store src
              with
              | Error e -> Error (refusal e)
              | Ok (Native4d.Lower.Pack r) ->
                  let g4 = Native4d.Lower.graph r in
                  let* n4 =
                    G.make ~ops:G.native4d_ops ~hashers ~algorithm
                      ~version:"native4d"
                      ~identity_text:
                        (identity_text
                           (fun f g -> Native4d.Graph.pp_with f g)
                           g4)
                      g4
                  in
                  let constants4 =
                    lazy
                      (materialize archive ~constants:r.constants
                         ~store:r.constant_store)
                  in
                  let module Export_lower =
                    Map_export.Make (Native_side) (Native4d.Framework.Side4)
                  in
                  let export = Export_lower.export r.map ~src ~dst:r.dst in
                  let name4 = through export name1 in
                  let n4_bytes id =
                    let* c = Lazy.force constants4 in
                    bytes_of ~graph:"Native4D" ~source_bytes ~name_of:name4 ctx
                      n4.graph c id
                  in
                  let lower =
                    lazy
                      (Pass_map.make ~name:"l" ~digest
                         ~src:{ Pass_map.model = n1; bytes = n1_bytes }
                         ~dst:{ Pass_map.model = n4; bytes = n4_bytes }
                         ~export)
                  in
                  Ok
                    {
                      n0;
                      n1;
                      canonical_bytes = n1_bytes;
                      n4;
                      constants = constants4;
                      canonicalize;
                      lower;
                    }))
      in
      Ok
        { Canonical.n0; n1; canonical_bytes = n1_bytes; canonicalize; lowering }

let make ~hashers ~algorithm ~ctx ~archive ~native ~source_bytes :
    (t, string) result =
  let* c = canonical ~hashers ~algorithm ~ctx ~archive ~native ~source_bytes in
  Lazy.force c.lowering
