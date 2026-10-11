(* Run two engines over one released Transformers fixture and case through the
   shared-frontier coordinator, listing each cluster as it is compared.
   Usage: fixture_pair_run COHORT CACHE_DIR ARTIFACT_ID CASE_ID LEFT RIGHT
   LEFT is [aten] (the import correspondence is ATen to Native); RIGHT is
   [native] (binary64 dots) or [native-sequential] (binary32 sequential dots).
   Exit status: 0 agreement, 1 disagreement, 3 refused or failed, 4 not
   established.
   Environment: TOL (default 1e-4), DONOR=left|right, DUMP=dir (write the
   donor's final outputs). The fixture must already be in the cache and
   HF_HUB_CACHE must hold its Hub checkpoint: see [make transformers.compare]. *)
open Inference_compare_schema
module P = Inference_compare_pt2
module Core = Inference_compare_core
module U = Pt2_fixture_unix

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let ok = function Ok x -> x | Error m -> failwith m

let unwrap = function
  | Ok x -> x
  | Error e -> failwith (Fmt.str "%a" U.Fixture.pp_error e)

let usage () =
  prerr_endline
    "usage: fixture_pair_run COHORT CACHE_DIR ARTIFACT_ID CASE_ID LEFT RIGHT\n\
     engines: LEFT is aten; RIGHT is native or native-sequential";
  exit 2

(* The operations of a cluster, in graph order: "view, permute, clone". *)
let operations (g : Graph_desc.t) (c : Cluster.t) =
  let op (n : Node_desc.t) = Id.Operator.to_string n.operator.operation in
  let nodes =
    List.filter
      (fun (n : Node_desc.t) ->
        List.exists (fun r -> Node_ref.equal r n.node) c.nodes)
      g.nodes
  in
  match List.map op nodes with
  | [] -> "(no nodes)"
  | ops -> String.concat ", " ops

let () =
  let cohort_path, cache_dir, artifact, case, left, right =
    match Array.to_list Sys.argv with
    | [ _; a; b; c; d; e; f ] -> (a, b, c, d, e, f)
    | _ -> usage ()
  in
  let dots =
    match (left, right) with
    | "aten", "native" -> Direct.Binary64
    | "aten", "native-sequential" -> Direct.Binary32_sequential
    | _ ->
        Printf.eprintf "unsupported engine pair %s / %s\n" left right;
        usage ()
  in
  let donor : Side.t =
    match Sys.getenv_opt "DONOR" with Some "right" -> Right | _ -> Left
  in
  let tol =
    match Sys.getenv_opt "TOL" with Some s -> float_of_string s | None -> 1e-4
  in
  let cohort =
    unwrap
      (Err.payload
         (let open Err.Syntax in
          let* text = U.Fetch.read cohort_path in
          Pt2_fixture.Cohort.of_string text))
  in
  let entry =
    match Pt2_fixture.Cohort.find cohort artifact with
    | Some e -> e
    | None -> failwith ("unknown artifact " ^ artifact)
  in
  let cache = unwrap (Err.payload (U.Cache.create cache_dir)) in
  let fixture =
    unwrap (Err.payload (U.Fixture.open_ (U.Bundle.config cache) cohort entry))
  in
  let identity_text =
    unwrap
      (Err.payload (U.Bundle.read_member fixture.bundle "models/model.json"))
  in
  let inputs =
    match
      Pt2_archive.load_pt_tensor_map
        (Filename.concat fixture.bundle.dir
           (Printf.sprintf "cases/%s/inputs.pt" case))
    with
    | Ok l -> l
    | Error _ -> failwith ("no inputs for " ^ case)
  in
  let aten =
    ok
      (P.Aten_model.of_archive ~hashers ~algorithm ~identity_text
         fixture.archive)
  in
  let pair = ok (P.Pair.of_aten ~hashers ~algorithm ~inputs aten) in
  let mapping = P.Pair.mapping pair in
  let left_graph = P.Pair.left_graph pair
  and right_graph = P.Pair.right_graph pair in
  let total = List.length mapping.units in
  let describe =
    List.map
      (fun (u : Matched_unit.t) ->
        ( Id.Unit.to_string u.unit_id,
          (operations left_graph u.left, operations right_graph u.right) ))
      mapping.units
  in
  Printf.printf "%s, %s\n" artifact case;
  Printf.printf "left %s | right %s | donor %s | tolerance %g | %d clusters\n%!"
    left right
    (match donor with Left -> "left" | Right -> "right")
    tol total;
  let started = ref 0 in
  let on_event (e : Event.t) =
    match (e.payload, e.unit) with
    | Inputs_verified _, Some u ->
        incr started;
        let name = Id.Unit.to_string u in
        let l, r =
          Option.value (List.assoc_opt name describe) ~default:("?", "?")
        in
        Printf.printf "[%d/%d] %s\n         %s: %s\n         %s: %s\n%!"
          !started total name left l right r
    | Outputs_compared results, _ ->
        let n v =
          List.length
            (List.filter
               (fun (c : Comparison_result.t) -> c.verdict = v)
               results)
        in
        let worst =
          List.fold_left
            (fun acc (c : Comparison_result.t) ->
              match
                Option.bind c.element (fun el -> el.absolute.maximum_finite)
              with
              | Some v -> Float.max acc v
              | None -> acc)
            0.0 results
        in
        Printf.printf
          "         compared %d output(s): %d pass, %d mismatch, max abs %.3g\n\
           %!"
          (List.length results) (n Pass) (n Mismatch) worst
    | Failed f, _ ->
        Printf.printf "         FAILED %s\n%!"
          (match Failure.encode_code f.code with
          | Jsont.String (s, _) -> s
          | _ -> "?")
    | _ -> ()
  in
  let r, _ =
    Pair_harness.Harness.run_with ~on_event ~donor ~tol ~mapping ~left_graph
      ~right_graph ~initial:(P.Pair.initial pair)
      ~node_attributes:(P.Pair.node_attributes pair)
      ~left_endpoint:(P.Pair.aten_endpoint pair)
      ~right_endpoint:(P.Pair.native_endpoint ~dots pair)
  in
  (match Sys.getenv_opt "DUMP" with
  | None -> ()
  | Some dir ->
      List.iteri
        (fun i (_, v) ->
          Out_channel.with_open_bin
            (Filename.concat dir (Printf.sprintf "final-%d.bin" i))
            (fun oc ->
              output_string oc
                (Inference_compare_codecs.Logical_value.contents v)))
        r.Core.Coordinator.finals);
  (* The inference_compare command's codes: 0 agreement, 1 disagreement,
     3 a refused or failed run, 4 agreement not established. *)
  exit
    (match (r.completion.execution, r.completion.numerical) with
    | Complete, Agreement -> 0
    | Complete, Disagreement -> 1
    | Complete, Not_established -> 4
    | (Failed | Refused | Interrupted), _ -> 3)
