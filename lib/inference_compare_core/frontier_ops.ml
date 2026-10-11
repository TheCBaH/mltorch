open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Plan = Inference_compare_plan
module Recipe_eval = Codecs.Recipe_eval
module Metrics = Inference_compare_metrics

let ( let* ) = Result.bind

(* ---- configuration shared by every operation ----------------------------- *)

type env = {
  ctx : L.Context.t;
  codecs : Codecs.Element_codec.Registry.t;
  operations : Codecs.Conversion_registry.t;
  metrics : Metrics.Tensor_metrics.env;
  canonical_spec : Id.Canonical_value.t -> Tensor_spec.t option;
}

let recipe_env env ~boundary : Recipe_eval.env =
  {
    codecs = env.codecs;
    operations = env.operations;
    canonical = env.canonical_spec;
    boundary;
  }

let codec_failure ?side ?unit ?tensor phase
    (e : Codecs.Codec_error.t Err.Error.t) =
  let text = Format.asprintf "%a" Codecs.Codec_error.pp (Err.Error.kind e) in
  let code : Failure.code =
    match Err.Error.kind e with
    | Resource_exceeded _ -> Resource_exhausted
    | Lossy_recipe | Undeclared_lossy_cast | Cast_overflow _ ->
        Lossy_common_input
    | _ -> Boundary_mismatch
  in
  Run_state.failure ?side ?unit ?tensor ~text code phase

(* Evaluate a checked-on-demand recipe: [canonical] reads the store,
   [boundary] reads the supplied raw tensors. *)
let evaluate env ~boundary_spec ~canonical ~boundary ?side ?unit ?tensor phase
    recipe : (L.t * Recipe_eval.Checked.t, Failure.t) result =
  let renv = recipe_env env ~boundary:boundary_spec in
  match Recipe_eval.check renv recipe with
  | Error e -> Error (codec_failure ?side ?unit ?tensor phase e)
  | Ok checked -> (
      match
        Recipe_eval.evaluate env.ctx renv checked { canonical; boundary }
      with
      | Ok (v, _) -> Ok (v, checked)
      | Error e -> Error (codec_failure ?side ?unit ?tensor phase e))

(* ---- id maps ------------------------------------------------------------- *)

let to_extracted (ex : Inference_compare_extract.Extraction.t) t =
  List.find_map
    (fun (o, e) -> if Tensor_ref.equal o t then Some e else None)
    ex.descriptor.original_to_extracted_tensors

let to_original (ex : Inference_compare_extract.Extraction.t) t =
  List.find_map
    (fun (o, e) -> if Tensor_ref.equal e t then Some o else None)
    ex.descriptor.original_to_extracted_tensors

(* ---- sealed policies ----------------------------------------------------- *)

let binding_for (bindings : Policy_bindings.t) comparison =
  List.find_opt
    (fun (b : Policy_binding.t) ->
      Id.Comparison.equal b.site.comparison comparison)
    bindings.bindings

(* ---- comparisons --------------------------------------------------------- *)

let compare_pair env bindings ~unit ~comparison ~left ~right :
    (Comparison_result.t, Failure.t) result =
  match binding_for bindings comparison with
  | None -> Error (Run_state.failure ~unit ~comparison Missing_policy Commit)
  | Some b -> (
      match
        Metrics.Comparison.compare env.ctx env.metrics b.policy
          ~comparison_id:comparison ~policy_digest:b.policy_digest ~left ~right
      with
      | Ok r -> Ok r
      | Error e ->
          Error
            (Run_state.failure ~unit ~comparison
               ~text:
                 (Format.asprintf "%a" Metrics.Comparison.pp_error
                    (Err.Error.kind e))
               Unsupported_policy Commit))
