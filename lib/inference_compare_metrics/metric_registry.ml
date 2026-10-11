open Inference_compare_schema

type impl = {
  comparator : Id.Comparator.t;
  implementation_digest : Digest.Implementation.t;
  configuration_digest : Digest.Comparison_policy.t;
  threshold_direction : Capability.threshold_direction;
  max_group_elements : int;
  compute : float array -> float array -> Metric_outcome.t;
}

type t = impl list

let empty = []

let register t (i : impl) =
  if
    List.exists
      (fun (x : impl) -> Id.Comparator.equal x.comparator i.comparator)
      t
  then Err.fail (`Duplicate_comparator i.comparator)
  else Ok (t @ [ i ])

(* A step is admitted by comparator and both digests together: a different
   implementation or configuration is a different metric. *)
let find t ~comparator ~implementation_digest ~configuration_digest =
  List.find_opt
    (fun (i : impl) ->
      Id.Comparator.equal i.comparator comparator
      && Digest.Implementation.equal i.implementation_digest
           implementation_digest
      && Digest.Comparison_policy.equal i.configuration_digest
           configuration_digest)
    t

let known t comparator =
  List.exists (fun (i : impl) -> Id.Comparator.equal i.comparator comparator) t

let capabilities t =
  List.map
    (fun (i : impl) ->
      {
        Capability.Metric.comparator = i.comparator;
        implementation_digest = i.implementation_digest;
        threshold_direction = i.threshold_direction;
        finite_operands_only = true;
      })
    t
