(* A pass pipeline's own correspondence as a mapping artifact: the planner
   accepts it, deleted and created nodes are explained, and the units are the
   coarsest the dataflow forces. *)

open Inference_compare_schema
module Plan = Inference_compare_plan
module G = Inference_compare_pt2.Graph_model
module PM = Inference_compare_pt2.Pass_map
module Tid = Graph_ir.Tensor_id
module Export = Map_export.Make (Native_side) (Native_side)

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let ok = function Ok x -> x | Error m -> failwith m

let limits : Resource_limits.t =
  {
    max_live_bytes = 1_000_000_000L;
    max_conversion_scratch_bytes = 1_000_000_000L;
    max_snapshot_bytes = 1_000_000_000L;
    max_retained_disk_bytes = 0L;
    max_mismatch_samples = 8L;
    retention = No_retention;
    retention_requirement = Best_effort;
    verified_spill = false;
    max_nodes = 100_000L;
    max_edges = 100_000L;
    max_units = 100_000L;
    max_recipe_nodes = 100_000L;
    max_selector_nodes = 100L;
    max_selector_depth = 16L;
    max_path_length = 8L;
    max_retained_witnesses = 100L;
  }

let options : Plan.Correspondence_check.options =
  {
    codecs = Inference_compare_codecs.Element_codec.Registry.builtin;
    operations = Inference_compare_codecs.Conversion_registry.empty;
  }

let digest bytes =
  match
    Inference_compare_codecs.Payload_hasher.digest_string hashers algorithm
      bytes
  with
  | Ok d -> Ok d
  | Error _ -> Error "digest"

let graph_text g =
  match Jsont_bytesrw.encode_string Graph_ir.graph_jsont g with
  | Ok s -> s
  | Error m -> m

let model version g =
  ok
    (G.make ~ops:G.native_ops ~hashers ~algorithm ~version
       ~identity_text:(graph_text g) g)

let bytes _ = Ok "payload"

(* The canonical pipeline over [g]: both descriptions and the pass map. *)
let canonicalize g =
  let open Err.Syntax in
  match
    let* (Rewrite.Origin state) =
      (Rewrite.origin g :> (Rewrite.origin, Pass.error) Err.t)
    in
    let+ (Rewrite.Step (once, map)) =
      Pass.run_all state [ Pipeline.canonical ~fold:false ]
    in
    let export =
      Export.export map ~src:(Rewrite.snapshot state)
        ~dst:(Rewrite.snapshot once)
    in
    (Rewrite.graph once, export)
  with
  | Error _ -> failwith "pipeline"
  | Ok (after, export) ->
      let src = model "before" g and dst = model "after" after in
      (src, dst, export)

let plan (src : _ G.t) (dst : _ G.t) (m : Mapping_artifact.t) =
  Plan.Structural.check ~limits ~options ~left:src.desc ~right:dst.desc m

let make ?(name = "p") (src, dst, export) =
  PM.make ~name ~digest ~src:{ PM.model = src; bytes }
    ~dst:{ PM.model = dst; bytes } ~export

let summary (m : Mapping_artifact.t) =
  let nodes side = List.length side.Cluster.nodes in
  List.iter
    (fun (u : Matched_unit.t) ->
      Printf.printf "%s: %d node(s) -> %d node(s)\n"
        (Id.Unit.to_string u.unit_id)
        (nodes u.left) (nodes u.right))
    m.units;
  List.iter
    (function
      | Member_disposition.Created _ -> print_endline "created"
      | Eliminated _ -> print_endline "eliminated"
      | Internal_helper _ -> print_endline "helper"
      | Tensor_not_compared _ -> print_endline "not compared")
    m.dispositions

let%expect_test "the canonical pipeline's map is a legal partition" =
  let s = canonicalize (Native_test.Pipeline_test.mixed ()) in
  let src, dst, _ = s in
  let r = ok (make s) in
  summary r.mapping;
  (match plan src dst r.mapping with
  | Ok _ -> print_endline "planner: accepted"
  | Error e ->
      Format.printf "planner: %a@." (Err.Error.pp_kind Plan.Plan_error.pp) e);
  [%expect
    {|
    pu00000: 4 node(s) -> 1 node(s)
    pu00001: 1 node(s) -> 0 node(s)
    pu00002: 1 node(s) -> 1 node(s)
    eliminated
    eliminated
    eliminated
    eliminated
    planner: accepted |}]

(* x -> a -> b -> c, the same chain on both sides, with the pass claiming that
   a and c form one cluster while b stays alone: the units would depend on
   each other, so the provider must make them one. *)
let chain () =
  Native_test.Pipeline_test.build "chain"
    (let open Graph_builder in
     let* x = input ~shape:(Native_test.Pipeline_test.nhwc ~h:4 ~w:4 ~c:3) () in
     let* a = relu x in
     let* b = sqrt a in
     relu b)

let chain_export g : Map_export.t =
  let node i = (List.nth g.Graph_ir.Graph.nodes i).Graph_common.Node.id in
  let tensor i =
    match i with
    | 0 -> List.hd g.Graph_ir.Graph.inputs
    | i -> List.nth (List.nth g.Graph_ir.Graph.nodes (i - 1)).outputs 0
  in
  {
    nodes =
      [
        { src = [ node 0; node 2 ]; dst = [ node 0; node 2 ] };
        { src = [ node 1 ]; dst = [ node 1 ] };
      ];
    values =
      List.init 4 (fun i ->
          {
            Map_export.Value_cluster.src = [ tensor i ];
            dst = [ tensor i ];
            claim = Identical;
          });
  }

let%expect_test "units that would depend on each other become one" =
  let g = chain () in
  let src = model "left" g and dst = model "right" g in
  let r = ok (make (src, dst, chain_export g)) in
  summary r.mapping;
  (match plan src dst r.mapping with
  | Ok _ -> print_endline "planner: accepted"
  | Error e ->
      Format.printf "planner: %a@." (Err.Error.pp_kind Plan.Plan_error.pp) e);
  [%expect {|
    pu00000: 3 node(s) -> 3 node(s)
    planner: accepted |}]

module Export4 = Map_export.Make (Native_side) (Native4d.Framework.Side4)

let lower_to_4d g =
  match Snapshot.create g with
  | Error _ -> failwith "snapshot"
  | Ok (Snapshot.Pack src) -> (
      match Native4d.Lower.convert src with
      | Error e -> Error e
      | Ok (Native4d.Lower.Pack r) ->
          let g4 = Native4d.Lower.graph r in
          let n1 = model "canonical" g in
          let n4 =
            ok
              (G.make ~ops:G.native4d_ops ~hashers ~algorithm
                 ~version:"native4d"
                 ~identity_text:
                   (Format.asprintf "%a"
                      (fun f g -> Native4d.Graph.pp_with f g)
                      g4)
                 g4)
          in
          let e = Export4.export r.map ~src ~dst:r.dst in
          let p =
            ok
              (PM.make ~name:"l" ~digest ~src:{ PM.model = n1; bytes }
                 ~dst:{ PM.model = n4; bytes } ~export:e)
          in
          Ok
            (Plan.Structural.check ~limits ~options ~left:n1.desc ~right:n4.desc
               p.mapping
            |> Result.map (fun _ -> List.length p.mapping.units)))

let%expect_test "the four-axis lowering's map is a legal partition" =
  let _, dst, _ = canonicalize (Native_test.Pipeline_test.mixed ()) in
  (match lower_to_4d dst.graph with
  | Error e -> print_endline (Inference_compare_pt2.Native4d_model.refusal e)
  | Ok (Ok units) -> Printf.printf "planner: accepted, %d unit(s)\n" units
  | Ok (Error e) ->
      Format.printf "planner: %a@." (Err.Error.pp_kind Plan.Plan_error.pp) e);
  [%expect {| planner: accepted, 2 unit(s) |}]

let%expect_test "a graph outside the four-axis domain is a refusal" =
  let g =
    Native_test.Pipeline_test.build "batch"
      (let open Graph_builder in
       let* x = input ~shape:(Vec6.shape ~n:2 ~t:2 ~d:2 ~h:2 ~w:2 ~c:2) () in
       relu x)
  in
  (match lower_to_4d g with
  | Error e -> print_endline (Inference_compare_pt2.Native4d_model.refusal e)
  | Ok _ -> print_endline "admitted");
  [%expect
    {|
    outside the Native4D domain: tensor t0 has extent on T or D: [N=2 T=2 D=2 H=2
                                                                  W=2 C=2] |}]
