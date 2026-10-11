(* Run two engines over one model and case through the shared-frontier
   coordinator, listing each cluster as it is compared.
   Usage:
     fixture_pair_run fixture COHORT CACHE_DIR ARTIFACT_ID CASE_ID LEFT RIGHT
     fixture_pair_run safetensors MODEL_DIR INPUTS_PT SAMPLE LEFT RIGHT
     fixture_pair_run model NAME LEFT RIGHT
   The first is a released Transformers fixture; the second a timm model whose
   graph and weight map are a submodule model directory and whose weights are
   the pinned Hub checkpoint (no .pt2), with the release's inputs.pt for the
   sample.
   The third resolves NAME itself (see [Model_source.resolve]) to one of the two
   above, from the directories MODELS_DIR (the timm models), COHORT, CACHE_DIR
   (the fixture cache) and RELEASE_DIR (the timm releases' inputs), and takes
   SAMPLE or CASE (default 0 and case-00).
   LEFT and RIGHT are variants: a graph view [aten], [native],
   [native.canonical] or [native4d], with [:sequential32] after a Native view
   for binary32 sequential dots ([native-sequential] is the same as
   [native:sequential32]). ATen can only be on the left. A Native view also
   takes a generated-code engine, [loop-c], [ssa-c], [loop-wasm] or [ssa-wasm]
   (the host C compiler, or node): [native.canonical:loop-c]. The optional
   routes are words too, where the executable was built with them: the Rivet
   AArch64 image route and table host ([rivet], [rivet-table], each with a
   [-scanned] allocator) and the embedded CompCert ([compcert]); a word whose
   engine was not built refuses, naming what is missing. The JavaScript
   engines ([loop-js], [ssa-js], [jsoo]) run under node in a helper process
   (js/jsoo/compare_js_engine, or the file MLTORCH_COMPARE_JS names).
   Exit status: 0 agreement, 1 disagreement, 3 refused or failed, 4 not
   established.
   Environment: TOL (default 1e-4), DONOR=left|right, DUMP=dir (write the
   donor's final outputs). The fixture must already be in the cache and
   HF_HUB_CACHE must hold its Hub checkpoint: see [make model.compare.<model>], which also resolves the model name (bin/model_source). *)
open Inference_compare_schema
module P = Inference_compare_pt2
module Core = Inference_compare_core
module U = Pt2_fixture_unix
module Timm = Timm_fixture
module J = Timm.Json
module TU = Timm_fixture_unix

let timm_or_raise r = Err.or_raise ~pp_error:Timm.pp_error r
let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let ok = function Ok x -> x | Error m -> failwith m

let unwrap = function
  | Ok x -> x
  | Error e -> failwith (Fmt.str "%a" U.Fixture.pp_error e)

let usage () =
  Printf.eprintf
    "usage: fixture_pair_run fixture COHORT CACHE_DIR ARTIFACT_ID CASE_ID LEFT \
     RIGHT\n\
    \       fixture_pair_run safetensors MODEL_DIR INPUTS_PT SAMPLE LEFT RIGHT\n\
    \       fixture_pair_run model NAME LEFT RIGHT\n\
     variants: aten, native, native.canonical, native4d (native views also \
     VIEW:sequential32, or an engine: %s)\n"
    (String.concat ", " (P.Provider.names ()));
  exit 2

(* The sample of the release's [inputs.pt] the exporter picks, with the batch
   axis the model takes (the samples are single images). *)
let sample_input ~inputs ~index (aten : P.Aten_model.t) =
  let with_batch (t : Pt2_tensor.t) =
    let stride = Pt2_tensor.numel t in
    { t with Pt2_tensor.sizes = 1 :: t.sizes; strides = stride :: t.strides }
  in
  match Pt2_archive.load_pt_tensor_map inputs with
  | Error _ -> failwith (inputs ^ ": unreadable")
  | Ok samples -> (
      let name =
        match
          List.find_opt
            (fun (s : P.Aten_model.source) -> s.captured = None)
            aten.sources
        with
        | Some s -> s.name
        | None -> failwith "the graph has no user input"
      in
      match List.nth_opt samples index with
      | Some (_, t) -> [ (name, with_batch t) ]
      | None -> failwith (Printf.sprintf "%s has no sample %d" inputs index))

(* Where a model comes from: its label, archive, the text that identifies its
   graph, and the case's inputs. *)
let load = function
  | `Fixture (cohort_path, cache_dir, artifact, case) ->
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
        unwrap
          (Err.payload (U.Fixture.open_ (U.Bundle.config cache) cohort entry))
      in
      let identity_text =
        unwrap
          (Err.payload
             (U.Bundle.read_member fixture.bundle "models/model.json"))
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
      ( Printf.sprintf "%s, %s" artifact case,
        fixture.archive,
        identity_text,
        inputs,
        None )
  | `Timm_v2 (config, index, name, model_dir, sample) ->
      let shared = Sys.getenv_opt "TIMM_MODE" = Some "images" in
      let fixture = timm_or_raise (TU.Bundle.open_ ~shared config index name) in
      let evaluation = fixture.evaluation in
      let case =
        timm_or_raise
          (Err.of_option
             (`Timm_fixture J.Failure.Case_index_outside_publication)
             (List.nth_opt evaluation.cases sample))
      in
      let archive =
        timm_or_raise (TU.Bundle.archive config fixture ~model_dir)
      in
      let inputs =
        timm_or_raise (Timm.bind_inputs evaluation.contract case fixture.inputs)
      in
      ( Printf.sprintf "%s, %s (v2 verified bindings)" name case.id,
        archive,
        fixture.graph_bytes,
        inputs,
        Some (evaluation, case) )
  | `Safetensors (model_dir, inputs_pt, index) ->
      let archive =
        match Err.payload (Pt2_safetensors_unix.open_dir model_dir) with
        | Ok a -> a
        | Error e -> failwith (Fmt.str "%a" Pt2_safetensors_unix.pp_error e)
      in
      let identity_text =
        match
          Err.payload
            (Pt2_archive.read_file
               (Filename.concat model_dir "models/model.json"))
        with
        | Ok t -> t
        | Error _ -> failwith (model_dir ^ ": no models/model.json")
      in
      (* The user input's name comes from the graph itself. *)
      let aten =
        ok (P.Aten_model.of_archive ~hashers ~algorithm ~identity_text archive)
      in
      ( Printf.sprintf "%s, sample %d (weights from the Hub checkpoint)"
          (Filename.basename model_dir)
          index,
        archive,
        identity_text,
        sample_input ~inputs:inputs_pt ~index aten,
        None )

(* NAME as one of the two sources, from the directories in the environment. *)
let resolve_model name =
  let env var default = Option.value (Sys.getenv_opt var) ~default in
  let dirs =
    {
      Model_source.models =
        env "MODELS_DIR" "modules/devcontainer.pytorch-image-models/models";
      cohort = env "COHORT" "data/transformers/cohort.json";
      cache = env "CACHE_DIR" "data/transformers-cache";
    }
  in
  match Model_source.resolve dirs name with
  | Error m ->
      prerr_endline m;
      exit 2
  | Ok (Timm { name; dir; _ }) -> (
      let discovered =
        match Sys.getenv_opt "TIMM_MANIFEST" with
        | Some path when path <> "" -> TU.Discovery.local path
        | _ ->
            TU.Discovery.remote
              ~url:
                (env "TIMM_MANIFEST_URL"
                   (Printf.sprintf
                      "https://github.com/%s/releases/download/%s/manifest-v2.json"
                      (env "TIMM_REPO"
                         "TheCBaH/devcontainer.pytorch-image-models")
                      (env "TIMM_RELEASE" "v0.0.6")))
      in
      match timm_or_raise discovered with
      | TU.Discovery.V2 index ->
          let source_dir =
            match Sys.getenv_opt "TIMM_SOURCE_DIR" with
            | Some "" -> None
            | value -> value
          in
          let transport =
            if Sys.getenv_opt "TIMM_OFFLINE" = Some "1" then None
            else Some U.Transport.curl
          in
          let config =
            timm_or_raise
              (TU.Config.create ?source_dir ?transport
                 (env "TIMM_CACHE" "data/timm-cache"))
          in
          `Timm_v2 (config, index, name, dir, int_of_string (env "SAMPLE" "0"))
      | Absent ->
          let release = env "RELEASE_DIR" "data/pt2-functional" in
          `Safetensors
            ( dir,
              Filename.concat (Filename.concat release name) "inputs.pt",
              int_of_string (env "SAMPLE" "0") ))
  | Ok (Fixture { id; _ }) ->
      `Fixture (dirs.cohort, dirs.cache, id, env "CASE" "case-00")

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

(* One compared output on one line: how many elements differ and by how much.
   The sign convention is the policy's (right minus left). *)
let describe_result (c : Comparison_result.t) =
  let verdict =
    match c.verdict with
    | Pass -> "pass"
    | Mismatch -> "MISMATCH"
    | Metadata_error -> "METADATA ERROR"
    | Metric_unavailable -> "METRIC UNAVAILABLE"
    | Unsupported -> "UNSUPPORTED"
  in
  match c.element with
  | None ->
      Printf.sprintf "%s: %s" (Id.Comparison.to_string c.comparison_id) verdict
  | Some e ->
      let g = function Some v -> Printf.sprintf "%.3g" v | None -> "-" in
      let scalar : Difference_statistics.scalar -> string = function
        | Finite_value v -> Printf.sprintf "%.3g" v
        | Unavailable _ -> "-"
      in
      let s = e.statistics in
      let total = c.elements_checked in
      if Int64.equal e.bitwise_mismatches 0L then
        Printf.sprintf "%s: %s, identical (%Ld elements)"
          (Id.Comparison.to_string c.comparison_id)
          verdict total
      else
        Printf.sprintf
          "%s: %s, %Ld of %Ld elements differ, %s outside tolerance; max abs \
           %s, max rel %s; bias %s, MAE %s, RMS %s; signs +%Ld -%Ld"
          (Id.Comparison.to_string c.comparison_id)
          verdict e.bitwise_mismatches total
          (match e.tolerance_violations with
          | Some v -> Int64.to_string v
          | None -> "-")
          (g e.absolute.maximum_finite)
          (g e.relative.maximum_finite)
          (scalar s.bias)
          (scalar s.mean_absolute_error)
          (scalar s.root_mean_square_error)
          s.positive_errors s.negative_errors

let () =
  Inference_compare_providers.Providers.register_all ();
  let source, left, right =
    match Array.to_list Sys.argv with
    | [ _; "fixture"; cohort; cache; artifact; case; l; r ] ->
        (`Fixture (cohort, cache, artifact, case), l, r)
    | [ _; "safetensors"; dir; inputs; sample; l; r ] ->
        (`Safetensors (dir, inputs, int_of_string sample), l, r)
    | [ _; "model"; name; l; r ] -> (resolve_model name, l, r)
    | _ -> usage ()
  in
  let variant s =
    match P.Variant.of_string s with
    | Ok v -> v
    | Error m ->
        prerr_endline m;
        usage ()
  in
  let left_variant = variant left and right_variant = variant right in
  let donor : Side.t =
    match Sys.getenv_opt "DONOR" with Some "right" -> Right | _ -> Left
  in
  let tol =
    match Sys.getenv_opt "TOL" with Some s -> float_of_string s | None -> 1e-4
  in
  let label, archive, identity_text, inputs, reference = load source in
  let aten =
    ok (P.Aten_model.of_archive ~hashers ~algorithm ~identity_text archive)
  in
  let base = ok (P.Pair.of_aten ~hashers ~algorithm ~inputs aten) in
  let pair =
    match P.Variant_pair.make base ~left:left_variant ~right:right_variant with
    | Ok pair -> pair
    | Error m ->
        (* Not a pair this run can build: refused, as a failed run is. *)
        Printf.printf "refused: %s\n" m;
        exit 3
  in
  let mapping = pair.mapping in
  let left_graph = pair.left_graph and right_graph = pair.right_graph in
  let total = List.length mapping.units in
  let describe =
    List.map
      (fun (u : Matched_unit.t) ->
        ( Id.Unit.to_string u.unit_id,
          (operations left_graph u.left, operations right_graph u.right) ))
      mapping.units
  in
  Printf.printf "%s\n" label;
  Printf.printf "left %s | right %s | donor %s | tolerance %g | %d clusters\n%!"
    (P.Variant.to_string left_variant)
    (P.Variant.to_string right_variant)
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
          !started total name
          (P.Variant.to_string left_variant)
          l
          (P.Variant.to_string right_variant)
          r
    | Outputs_compared results, _ ->
        List.iter
          (fun c -> Printf.printf "         %s\n%!" (describe_result c))
          results
    | Failed f, _ ->
        Printf.printf "         FAILED %s\n%!"
          (match Failure.encode_code f.code with
          | Jsont.String (s, _) -> s
          | _ -> "?")
    | _ -> ()
  in
  let r, _ =
    Pair_harness.Harness.run_with ~on_event ~donor ~tol ~mapping ~left_graph
      ~right_graph ~initial:pair.initial ~node_attributes:pair.node_attributes
      ~left_endpoint:pair.left_endpoint ~right_endpoint:pair.right_endpoint
  in
  List.iter
    (fun (name, notes) ->
      Option.iter (Printf.printf "%s storage: %s\n" name) (notes ()))
    [
      (P.Variant.to_string left_variant, pair.left_notes);
      (P.Variant.to_string right_variant, pair.right_notes);
    ];
  let reference_passed =
    match (reference, r.completion.execution) with
    | None, _ -> true
    | Some _, (Failed | Refused | Interrupted) -> false
    | Some (evaluation, case), Complete -> (
        let open Err.Syntax in
        let check =
          Err.List.iter2
            ~unequal_lengths:(fun _ _ ->
              `Timm_fixture J.Failure.Final_output_arity_differs)
            (fun binding (_, value) ->
              let module V = Inference_compare_codecs.Logical_value in
              let spec = V.spec value in
              let* () =
                Timm.Json.same "final dtype"
                  (Id.Codec.to_string spec.dtype.codec)
                  "f32"
              in
              let* logical =
                Pt2_fixture.Logical.of_bytes ~dtype:Pt2_checkpoint_map.Dtype.F32
                  ~shape:spec.logical_shape (V.contents value)
                |> TU.wrap J.External.Source.Logical
                     Pt2_fixture.Logical.pp_error
              in
              let reference =
                List.assoc binding.Timm.Binding.sample case.Timm.Case.outputs
              in
              let tensor =
                List.assoc reference.Timm.Reference.key
                  evaluation.TU.Evaluation.outputs
              in
              Timm.compare_output evaluation.contract binding tensor logical)
            evaluation.contract.outputs r.Core.Coordinator.finals
        in
        match Err.export ~pos:__POS__ check with
        | Ok () ->
            print_endline "published full outputs: pass";
            true
        | Error e ->
            prerr_endline (Fmt.str "%a" Timm.pp_error e);
            false)
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
  (* The inference_compare command's codes. *)
  let code = Inference_compare_host.Pair_run.exit_code r in
  exit (if code = 0 && not reference_passed then 1 else code)
