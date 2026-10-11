open Inference_compare_schema
module Codecs = Inference_compare_codecs
module E = Codecs.Element_codec
module L = Codecs.Logical_value
module O = Metric_outcome

type env = {
  registry : Metric_registry.t;
  supports : Metric_support.t -> float array option;
}

let empty_env = { registry = Metric_registry.empty; supports = (fun _ -> None) }

type error =
  | Config of Metric_config_error.t
  | Spec_mismatch
  | Unknown_codec of Id.Codec.t

let pp_error ppf = function
  | Config c -> Metric_config_error.pp ppf c
  | Spec_mismatch -> Fmt.string ppf "left and right specs differ"
  | Unknown_codec c -> Fmt.pf ppf "unknown codec %a" Id.Codec.pp c

let ( let* ) = Err.bind
let cfg e = Err.fail ~pp_error:Metric_config_error.pp e

(* Built-in metrics support real binary32/binary64 values. *)
let float_spec (spec : Tensor_spec.t) =
  let name = Id.Codec.to_string spec.dtype.codec in
  name = "f32" || name = "f64"

let direction_of (k : Tensor_metric.kind) env : Capability.threshold_direction =
  match k with
  | Cosine_similarity _ -> At_least
  | Registered { comparator; _ } -> (
      match
        List.find_opt
          (fun (c : Capability.Metric.t) ->
            Id.Comparator.equal c.comparator comparator)
          (Metric_registry.capabilities env.registry)
      with
      | Some c -> c.threshold_direction
      | None -> Declared)
  | _ -> At_most

let validate env (spec : Tensor_spec.t) (m : Tensor_metric.t) =
  let* groups = Tensor_groups.make ~shape:spec.logical_shape m.partition in
  let* () =
    match m.kind with
    | Registered _ -> Ok ()
    | _ ->
        if float_spec spec then Ok ()
        else cfg (Metric_config_error.Unsupported_codec spec.dtype.codec)
  in
  let* () =
    match m.role with
    | Diagnostic_only -> Ok ()
    | Required threshold -> (
        let bound, dir =
          match threshold with
          | At_most v -> (v, `Most)
          | At_least v -> (v, `Least)
        in
        if not (Float.is_finite bound) then
          cfg Metric_config_error.Nonfinite_bound
        else
          match (direction_of m.kind env, dir) with
          | At_most, `Most ->
              if bound < 0.0 then
                cfg Metric_config_error.Negative_distance_bound
              else Ok ()
          | At_least, `Least -> (
              match m.kind with
              | Cosine_similarity _ ->
                  if bound < -1.0 || bound > 1.0 then
                    cfg Metric_config_error.Cosine_bound_out_of_range
                  else Ok ()
              | _ -> Ok ())
          | Declared, _ -> Ok ()
          | _ -> cfg Metric_config_error.Wrong_threshold_direction)
  in
  match m.kind with
  | Wasserstein_1d (Histogram_masses { support; mass }) -> (
      let size = Int64.of_int (Tensor_groups.group_size groups) in
      let* () =
        if Int64.equal support.positions size then Ok ()
        else
          cfg
            (Metric_config_error.Support_positions_mismatch
               { declared = support.positions; group_size = size })
      in
      match mass with
      | Require_unit_mass { absolute_tolerance }
        when not
               (Float.is_finite absolute_tolerance && absolute_tolerance >= 0.0)
        ->
          cfg Metric_config_error.Invalid_mass_tolerance
      | _ -> Ok ())
  | Registered { comparator; _ } ->
      if Metric_registry.known env.registry comparator then Ok ()
      else cfg (Metric_config_error.Unregistered_comparator comparator)
  | _ -> Ok ()

let group_outcome env (m : Tensor_metric.t) a b : O.t =
  let open Metric_kernels in
  match m.kind with
  | Mean_absolute_error -> mean_absolute_error a b
  | Root_mean_square_error -> root_mean_square_error a b
  | Absolute_bias d -> absolute_bias d a b
  | Error_standard_deviation p -> error_standard_deviation p a b
  | Normalized_l2 { scale; zero_norm } -> normalized_l2 scale zero_norm a b
  | Cosine_similarity { zero_norm } -> cosine_similarity zero_norm a b
  | Wasserstein_1d Empirical_values -> wasserstein_empirical a b
  | Wasserstein_1d (Histogram_masses { support; mass }) -> (
      match env.supports support with
      | None -> Unsupported
      | Some positions -> wasserstein_histogram mass positions a b)
  | Registered { comparator; implementation_digest; configuration_digest } -> (
      match
        Metric_registry.find env.registry ~comparator ~implementation_digest
          ~configuration_digest
      with
      | None -> Unsupported
      | Some impl ->
          if Array.length a > impl.max_group_elements then Resource_failure
          else if not (both_finite a b) then Nonfinite
          else impl.compute a b)

let violates (role : Tensor_metric.role) v =
  match role with
  | Diagnostic_only -> false
  | Required (At_most t) -> v > t
  | Required (At_least t) -> v < t

let status_of (o : O.t) : Metric_result.status =
  match o with
  | Score _ -> Complete
  | Undefined -> Undefined
  | Unsupported -> Unsupported
  | Nonfinite -> Nonfinite
  | Resource_failure -> Resource_failure
  | Numerical_failure -> Numerical_failure

let compute ?(max_samples = 8) ctx env (m : Tensor_metric.t) ~left ~right =
  let spec = L.spec left in
  if not (Tensor_spec.equal spec (L.spec right)) then
    Err.fail ~pp_error Spec_mismatch
  else
    match E.Registry.find ctx.L.Context.codecs spec.dtype.codec with
    | None -> Err.fail ~pp_error (Unknown_codec spec.dtype.codec)
    | Some codec -> (
        let wrap r = Err.map_error (fun c -> Config c) r in
        match wrap (validate env spec m) with
        | Error e -> Error e
        | Ok () -> (
            match
              wrap (Tensor_groups.make ~shape:spec.logical_shape m.partition)
            with
            | Error e -> Error e
            | Ok groups ->
                let lc = L.contents left and rc = L.contents right in
                let read contents idx =
                  Array.map (fun i -> E.float_value codec contents i) idx
                in
                let n_groups = Tensor_groups.groups groups in
                let worst = ref (O.Score 0.0) in
                let checked = ref 0 and violations = ref 0 in
                let first_violation = ref None in
                let lo = ref None and hi = ref None in
                let samples = ref [] in
                for g = 0 to n_groups - 1 do
                  let idx = Tensor_groups.indices groups g in
                  let outcome =
                    group_outcome env m (read lc idx) (read rc idx)
                  in
                  if O.severity outcome > O.severity !worst then
                    worst := outcome;
                  match outcome with
                  | Score v ->
                      incr checked;
                      lo :=
                        Some
                          (match !lo with None -> v | Some x -> Float.min x v);
                      hi :=
                        Some
                          (match !hi with None -> v | Some x -> Float.max x v);
                      if List.length !samples < max_samples then
                        samples :=
                          (Tensor_groups.coords groups g, v) :: !samples;
                      if violates m.role v then (
                        incr violations;
                        if !first_violation = None then
                          first_violation :=
                            Some (Tensor_groups.coords groups g))
                  | _ -> ()
                done;
                let metric_value = function
                  | None -> Error_metric.No_finite_value
                  | Some v -> Error_metric.Finite_value v
                in
                Ok
                  {
                    Metric_result.metric = m.id;
                    status = status_of !worst;
                    elements_used =
                      Int64.of_int (n_groups * Tensor_groups.group_size groups);
                    groups_expected = Int64.of_int n_groups;
                    groups_checked = Int64.of_int !checked;
                    minimum = metric_value !lo;
                    maximum = metric_value !hi;
                    violations =
                      (match m.role with
                      | Diagnostic_only -> None
                      | Required _ -> Some (Int64.of_int !violations));
                    first_violation_group = !first_violation;
                    samples = List.rev !samples;
                  }))

let satisfied (m : Tensor_metric.t) (r : Metric_result.t) =
  match m.role with
  | Diagnostic_only -> None
  | Required _ ->
      Some
        (r.status = Complete
        && Int64.equal r.groups_checked r.groups_expected
        && r.violations = Some 0L)
