module W = Wire
module Experiment_id = W.Text (Id.Experiment)

type t = {
  experiment_id : Id.Experiment.t;
  default_donor : Side.t;
  overrides : Donor_override.t list;
  compare_policy_digest : Digest.Comparison_policy.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* experiment_id =
        W.Obj.required o "experiment_id" Experiment_id.decode
      in
      let* default_donor = W.Obj.required o "default_donor" Side.decode in
      let* overrides =
        W.Obj.required o "overrides" (W.list Donor_override.decode)
      in
      let* compare_policy_digest =
        W.Obj.required o "compare_policy_digest" Wire_digest.comparison_policy
      in
      Ok { experiment_id; default_donor; overrides; compare_policy_digest })

let encode e =
  W.obj
    [
      W.field "experiment_id" Experiment_id.encode e.experiment_id;
      W.field "default_donor" Side.encode e.default_donor;
      W.field "overrides" (W.enc_list Donor_override.encode) e.overrides;
      W.field "compare_policy_digest" Wire_digest.encode_comparison_policy
        e.compare_policy_digest;
    ]
