(* Two variants of one model and case, ready for the coordinator: the mapping
   between the two graph views, the initial values keyed to it, and an
   endpoint and identity for each side. The mapping is the identity for two
   configurations of one view, and otherwise the composition of the
   correspondences between the views in between: the importer's (ATen to
   Native), the canonical passes' (Native to canonical Native) and the
   four-axis lowering's. The passes' correspondences are read in either
   direction; the importer's only from ATen. A graph outside the four-axis
   domain is a refusal, never a Native run. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Core = Inference_compare_core
module Ex = Inference_compare_extract
module Compose = Inference_compare_compose.Compose
module G = Graph_model
module View = Variant.View

type t = {
  mapping : Mapping_artifact.t;
  left_graph : Graph_desc.t;
  right_graph : Graph_desc.t;
  initial : (Id.Canonical_value.t * L.t) list;
  node_attributes : Side.t -> Node_ref.t -> string;
  left_endpoint : Core.Side_exec.endpoint;
  right_endpoint : Core.Side_exec.endpoint;
  left_identity : unit -> Engine_identity.t;
  right_identity : unit -> Engine_identity.t;
  left_notes : unit -> string option;
  right_notes : unit -> string option;
      (** What an endpoint resolved for the bodies it prepared (storage and
          schedule), once the run is over. *)
}

let ( let* ) = Result.bind

(* The graph a variant runs. *)
type model =
  | Aten of Aten_model.t
  | Native of Graph_ir.op G.t
  | Native4d of Native4d.Op.t G.t

let desc = function
  | Aten m -> m.desc
  | Native m -> m.desc
  | Native4d m -> m.desc

let attributes = function
  | Aten m -> Aten_model.attributes_of m
  | Native m -> G.attributes_of m
  | Native4d m -> G.attributes_of m

(* The configuration the identity names, when it is not the default. *)
let config_text (v : Variant.t) =
  if Storage_config.is_default v.storage then None
  else Some ("storage=" ^ String.concat "," (Storage_config.to_words v.storage))

(* A side of the pair: the endpoint of a variant on its graph, the identity it
   names, and what it will have resolved once the run is over. A provider's
   prerequisites are checked here, before anything is prepared. *)
let side (base : Pair.t) (v : Variant.t) model :
    ( Core.Side_exec.endpoint
      * (unit -> Engine_identity.t)
      * (unit -> string option),
      string )
    result =
  let none () = None in
  let native_endpoint runner (m : Graph_ir.op G.t) : Core.Side_exec.endpoint =
    Endpoint
      {
        runner;
        build =
          (fun ex ->
            Ex.Dialect.realize (module Native_direct.Dialect.Dialect) m ex);
      }
  in
  match (v.evaluator, model) with
  | _, Aten _ ->
      Ok (Pair.aten_endpoint base, (fun () -> Aten_runner.identity ()), none)
  | Provider name, Native m ->
      let* i = Provider.instantiate name in
      Ok (native_endpoint i.runner m, (fun () -> i.identity), none)
  | Provider name, Native4d _ ->
      Error (Printf.sprintf "%s runs Native graphs, not Native4D" name)
  | Symbolic, Native m ->
      Ok
        ( native_endpoint (Symbolic_runner.create_native ()) m,
          (fun () -> Symbolic_runner.native_identity ()),
          none )
  | Symbolic, Native4d m ->
      Ok
        ( Endpoint
            {
              runner = Symbolic_runner.create_native4d ();
              build =
                (fun ex ->
                  Ex.Dialect.realize (module Native4d_dialect.Dialect) m ex);
            },
          (fun () -> Symbolic_runner.native4d_identity ()),
          none )
  | Direct, Native m when Storage_config.is_default v.storage ->
      Ok
        ( native_endpoint (Native_direct.create ~dots:v.dots ()) m,
          (fun () -> Native_runner.identity ~dots:v.dots ()),
          none )
  | Direct, Native m ->
      let notes = Native_storage.Notes.create () in
      let identity () =
        Native_runner.identity ~dots:v.dots ?config:(config_text v) ()
      in
      Ok
        ( native_endpoint
            (Native_storage.create ~dots:v.dots ~config:v.storage ~notes
               ~identity)
            m,
          identity,
          fun () -> Some (Native_storage.Notes.summary notes) )
  | Direct, Native4d m ->
      Ok
        ( Endpoint
            {
              runner = Native4d_runner.create ();
              build =
                (fun ex ->
                  Ex.Dialect.realize (module Native4d_dialect.Dialect) m ex);
            },
          (fun () -> Native4d_runner.identity ()),
          none )

let composed (base : Pair.t) first second =
  Compose.compose ~hashers:base.hashers ~algorithm:base.algorithm first second
  |> Result.map_error (fun e ->
      Format.asprintf "composition: %a"
        (Err.Error.pp_kind Inference_compare_compose.Compose_error.pp)
        e)

(* The correspondence from the view of rank [j] to that of rank [j + 1], or the
   way back, with the values it pins. *)
let step (base : Pair.t) (c : Native4d_model.Canonical.t) four ~forward j =
  let pass (p : (Pass_map.t, string) result) =
    let* p = p in
    Ok (p.mapping, p.pins)
  in
  match (j, forward) with
  | 0, true -> Ok (base.import.mapping, [])
  | 0, false -> Error "the mapping from Native to ATen is not available"
  | 1, true -> pass (Lazy.force c.canonicalize)
  | 1, false -> pass (Lazy.force c.canonicalize_reverse)
  | _, true ->
      let* (f : Native4d_model.t) = four () in
      pass (Lazy.force f.lower)
  | _, false ->
      let* (f : Native4d_model.t) = four () in
      pass (Lazy.force f.lower_reverse)

(* The mapping from the view of rank [lr] to that of rank [rr]. *)
let chain base c four ~lr ~rr =
  let forward = lr < rr in
  let steps =
    if forward then List.init (rr - lr) (fun i -> lr + i)
    else List.init (lr - rr) (fun i -> lr - 1 - i)
  in
  let rec go acc = function
    | [] -> Ok acc
    | j :: rest -> (
        let* mapping, pins = step base c four ~forward j in
        match acc with
        | None -> go (Some (mapping, pins)) rest
        | Some (first, first_pins) ->
            let* mapping = composed base first mapping in
            go (Some (mapping, first_pins @ pins)) rest)
  in
  let* r = go None steps in
  match r with Some r -> Ok r | None -> Error "no correspondence to build"

(* One view under two keys, related by the identity. *)
let self_pair (type op) (base : Pair.t) (m : op G.t) ~bytes
    ~(wrap : op G.t -> model) =
  let copy side =
    G.make ~ops:m.ops ~hashers:base.hashers ~algorithm:base.algorithm
      ~version:side
      ~identity_text:(Format.asprintf "%a/%s" Graph_key.pp m.key side)
      m.graph
  in
  let digest b =
    match Codecs.Payload_hasher.digest_string base.hashers base.algorithm b with
    | Ok d -> Ok d
    | Error _ -> Error "digest backend unavailable"
  in
  let* l = copy "left" in
  let* r = copy "right" in
  let* p =
    Pass_map.make ~name:"s" ~digest
      ~src:{ Pass_map.model = l; bytes }
      ~dst:{ Pass_map.model = r; bytes }
      ~export:(Pass_map.identity_export m.graph)
  in
  Ok (wrap l, wrap r, p.mapping, p.pins)

let make (base : Pair.t) ~(left : Variant.t) ~(right : Variant.t) :
    (t, string) result =
  let lr = View.rank left.view and rr = View.rank right.view in
  if rr = 0 then
    Error
      (if lr = 0 then "ATen against ATen runs one engine twice"
       else "ATen on the right needs the mapping from Native to ATen")
  else
    let* c =
      Native4d_model.canonical ~hashers:base.hashers ~algorithm:base.algorithm
        ~ctx:base.case_ctx ~archive:base.aten.archive ~native:base.native
        ~source_bytes:(Pair.source_bytes base)
    in
    let four () = Lazy.force c.lowering in
    let model_of = function
      | 0 -> Ok (Aten base.aten)
      | 1 -> Ok (Native c.n0)
      | 2 -> Ok (Native c.n1)
      | _ ->
          let* (f : Native4d_model.t) = four () in
          Ok (Native4d f.n4)
    in
    let* lm, rm, mapping, pins =
      if lr <> rr then
        let* lm = model_of lr in
        let* rm = model_of rr in
        let* mapping, pins = chain base c four ~lr ~rr in
        Ok (lm, rm, mapping, pins)
      else
        match rr with
        | 1 -> self_pair base c.n0 ~bytes:c.n0_bytes ~wrap:(fun m -> Native m)
        | 2 ->
            self_pair base c.n1 ~bytes:c.canonical_bytes ~wrap:(fun m ->
                Native m)
        | _ ->
            let* (f : Native4d_model.t) = four () in
            self_pair base f.n4 ~bytes:f.n4_bytes ~wrap:(fun m -> Native4d m)
    in
    let given = if lr = 0 then Pair.initial base else [] in
    let* pinned =
      Pass_map.pinned_values ~ctx:base.case_ctx ~mapping
        ~known:(List.map fst given) pins
    in
    let* left_endpoint, left_identity, left_notes = side base left lm in
    let* right_endpoint, right_identity, right_notes = side base right rm in
    Ok
      {
        mapping;
        left_graph = desc lm;
        right_graph = desc rm;
        initial = given @ pinned;
        node_attributes =
          (fun side r ->
            match (side : Side.t) with
            | Left -> attributes lm r
            | Right -> attributes rm r);
        left_endpoint;
        right_endpoint;
        left_identity;
        right_identity;
        left_notes;
        right_notes;
      }
