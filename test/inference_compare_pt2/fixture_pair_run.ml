(* Run the ATen / Native pair over one released Transformers fixture and case
   through the shared-frontier coordinator, printing every unit that differs.
   Usage: fixture_pair_run COHORT CACHE_DIR ARTIFACT_ID CASE_ID
   Environment: TOL (default 1e-4), DONOR=left|right, DOTS=exact|sequential
   (the Native dot accumulation), DUMP=dir (write the donor's final outputs). The fixture must already be in the cache, and
   HF_HUB_CACHE must hold its Hub checkpoint. *)
open Inference_compare_schema
module P = Inference_compare_pt2
module Core = Inference_compare_core
module Ex = Inference_compare_extract
module U = Pt2_fixture_unix

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let ok = function Ok x -> x | Error m -> failwith m

let unwrap = function
  | Ok x -> x
  | Error e -> failwith (Fmt.str "%a" U.Fixture.pp_error e)

let () =
  let cohort_path = Sys.argv.(1)
  and cache_dir = Sys.argv.(2)
  and artifact = Sys.argv.(3)
  and case = Sys.argv.(4) in
  let donor : Side.t =
    match Sys.getenv_opt "DONOR" with Some "right" -> Right | _ -> Left
  in
  let tol =
    match Sys.getenv_opt "TOL" with Some s -> float_of_string s | None -> 1e-4
  in
  let dots =
    match Sys.getenv_opt "DOTS" with
    | Some "sequential" -> Direct.Binary32_sequential
    | _ -> Direct.Binary64
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
    | None -> failwith "unknown artifact"
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
    ok
      (match
         Pt2_archive.load_pt_tensor_map
           (Filename.concat fixture.bundle.dir
              (Printf.sprintf "cases/%s/inputs.pt" case))
       with
      | Ok l -> Ok l
      | Error _ -> Error "inputs.pt")
  in
  let aten =
    ok
      (P.Aten_model.of_archive ~hashers ~algorithm ~identity_text
         fixture.archive)
  in
  let pair = ok (P.Pair.of_aten ~hashers ~algorithm ~inputs aten) in
  let left_endpoint = P.Pair.aten_endpoint pair
  and right_endpoint = P.Pair.native_endpoint pair in
  Printf.printf "artifact %s %s donor=%s tol=%g dots=%s\n%!" artifact case
    (match donor with Left -> "left" | Right -> "right")
    tol
    (match dots with Direct.Binary64 -> "exact" | _ -> "sequential");
  let r, _ =
    Direct.with_dot_accumulation dots (fun () ->
        Pair_harness.Harness.run ~donor ~tol ~mapping:(P.Pair.mapping pair)
          ~left_graph:(P.Pair.left_graph pair)
          ~right_graph:(P.Pair.right_graph pair) ~initial:(P.Pair.initial pair)
          ~node_attributes:(P.Pair.node_attributes pair)
          ~left_endpoint ~right_endpoint)
  in
  (* The committed final outputs of the donor, as raw logical bytes, for
     offline analysis (DUMP=dir). *)
  match Sys.getenv_opt "DUMP" with
  | None -> ()
  | Some dir ->
      List.iteri
        (fun i (_, v) ->
          Out_channel.with_open_bin
            (Filename.concat dir (Printf.sprintf "final-%d.bin" i))
            (fun oc ->
              output_string oc
                (Inference_compare_codecs.Logical_value.contents v)))
        r.Core.Coordinator.finals
