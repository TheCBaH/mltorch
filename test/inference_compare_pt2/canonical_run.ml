(* The Native views of one exported model without the four-axis lowering: the
   canonical graph is built first, and the lowering runs only when forced.
   Usage: canonical_run MODEL.pt2 INPUTS.pt *)
module P = Inference_compare_pt2

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let get = function Ok x -> x | Error m -> failwith m

let with_batch (tensor : Pt2_tensor.t) =
  let stride = Pt2_tensor.numel tensor in
  {
    tensor with
    Pt2_tensor.sizes = 1 :: tensor.sizes;
    strides = stride :: tensor.strides;
  }

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
      (P.Pair.of_aten ~hashers ~algorithm ~inputs:[ (input_name, sample) ] aten)
  in
  let c =
    get
      (P.Native4d_model.canonical ~hashers ~algorithm ~ctx:base.case_ctx
         ~archive:base.aten.archive ~native:base.native
         ~source_bytes:(P.Pair.source_bytes base))
  in
  Printf.printf "canonical: built, lowering forced: %b\n"
    (Lazy.is_val c.lowering);
  let canonicalize = get (Lazy.force c.canonicalize) in
  Printf.printf "canonicalize map: %d unit(s)\n"
    (List.length canonicalize.mapping.units);
  (match Lazy.force c.lowering with
  | Ok four ->
      Printf.printf "lowering: ok, %d node(s)\n"
        (List.length four.n4.graph.Graph_common.Graph.nodes)
  | Error m -> Printf.printf "lowering: refused: %s\n" m);
  Printf.printf "lowering forced: %b\n" (Lazy.is_val c.lowering)
