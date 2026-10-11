(* Run the embedded CompCert route against the direct assembly route on the
   canonical Native graph of one exported model and sample, through the shared
   frontier coordinator. *)
open Inference_compare_schema
module P = Inference_compare_pt2
module C = Inference_compare_compiled
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm

let with_batch (tensor : Pt2_tensor.t) =
  let stride = Pt2_tensor.numel tensor in
  {
    tensor with
    Pt2_tensor.sizes = 1 :: tensor.sizes;
    strides = stride :: tensor.strides;
  }

let donor : Side.t =
  match Sys.getenv_opt "DONOR" with Some "right" -> Right | _ -> Left

let tol =
  match Sys.getenv_opt "TOL" with Some s -> float_of_string s | None -> 1e-4

let get = function Ok x -> x | Error m -> failwith m

let () =
  let model = Sys.argv.(1) and inputs_pt = Sys.argv.(2) in
  let sample =
    match Pt2_archive.load_pt_tensor_map inputs_pt with
    | Ok ((_, t) :: _) -> with_batch t
    | _ -> failwith "inputs"
  in
  let aten = get (P.Aten_model.of_file ~hashers ~algorithm model) in
  let input_name =
    match
      List.find_opt
        (fun (s : P.Aten_model.source) -> s.captured = None)
        aten.sources
    with
    | Some s -> s.name
    | None -> failwith "no user input"
  in
  let base =
    get
      (P.Pair.load ~hashers ~algorithm ~path:model
         ~inputs:[ (input_name, sample) ])
  in
  let four =
    get
      (P.Native4d_model.make ~hashers ~algorithm ~ctx:base.case_ctx
         ~archive:base.aten.archive ~native:base.native
         ~source_bytes:(P.Pair.source_bytes base))
  in
  let pair =
    get
      (C.Compiled_pair.make ~hashers ~algorithm ~ctx:base.case_ctx
         ~graph:four.n1.graph ~bytes:four.canonical_bytes)
  in
  let r, _ =
    Pair_harness.Harness.run ~donor ~tol ~mapping:pair.mapping
      ~left_graph:pair.compcert.desc ~right_graph:pair.assembly.desc
      ~initial:pair.initial
      ~node_attributes:(fun side r ->
        match (side : Side.t) with
        | Left -> P.Graph_model.attributes_of pair.compcert r
        | Right -> P.Graph_model.attributes_of pair.assembly r)
      ~left_endpoint:(C.Compiled_pair.compcert_endpoint pair)
      ~right_endpoint:(C.Compiled_pair.assembly_endpoint pair)
  in
  (* Each route's segmented run (every unit fed the donor's own committed
     values) against the same route's whole-graph run, bitwise. *)
  if Sys.getenv_opt "CALIBRATE" <> None then begin
    let g = four.n1.graph in
    let whole_graph =
      { g with Graph_ir.Graph.input_kinds = Graph_ir.Tensor_id.Map.empty }
    in
    let inputs =
      List.map
        (fun id ->
          let spec = get (P.Graph_model.spec_of g id) in
          let v =
            get
              (Result.map_error
                 (fun _ -> "value")
                 (L.make base.case_ctx spec (get (four.canonical_bytes id))))
          in
          (id, get (P.Conv.native_of_logical v)))
        whole_graph.inputs
    in
    let timed f =
      let t0 = Unix.gettimeofday () in
      let r = f () in
      (r, Unix.gettimeofday () -. t0)
    in
    let outs, compile_s, execute_s =
      match donor with
      | Left ->
          let p, c =
            timed (fun () -> get (C.Compcert_engine.compile whole_graph))
          in
          let o, e =
            timed (fun () ->
                get (C.Compcert_engine.execute p whole_graph ~inputs))
          in
          C.Compcert_engine.release p;
          (o, c, e)
      | Right ->
          let p, c =
            timed (fun () ->
                get (Inference_compare_rivet.Image_engine.compile whole_graph))
          in
          let o, e =
            timed (fun () ->
                get
                  (Inference_compare_rivet.Image_engine.execute p whole_graph
                     ~inputs))
          in
          (o, c, e)
    in
    if Sys.getenv_opt "TIMING" <> None then
      Printf.eprintf "whole graph: compile %.3fs, execute %.3fs\n" compile_s
        execute_s;
    let same =
      List.for_all2
        (fun (packed : Tensor.packed) (_, v) ->
          let spec = L.spec v in
          let (Tensor.Tensor t) = packed in
          let shape =
            List.map
              (fun a -> Int64.of_int (Vec6.get t.shape a :> int))
              Axis.all
          in
          let spec = { spec with Tensor_spec.logical_shape = shape } in
          String.equal
            (L.contents
               (get (P.Conv.logical_of_native base.case_ctx spec packed)))
            (L.contents v))
        outs r.finals
    in
    Printf.printf "calibration: %d final output(s) %s the whole-graph run\n"
      (List.length outs)
      (if same then "bitwise equal" else "DIFFER from")
  end
