module W = Wire
module Metric_id = W.Text (Id.Tensor_metric)

type status =
  | Complete
  | Nonfinite
  | Numerical_failure
  | Resource_failure
  | Undefined
  | Unsupported

type t = {
  metric : Id.Tensor_metric.t;
  status : status;
  elements_used : int64;
  groups_expected : int64;
  groups_checked : int64;
  minimum : Error_metric.t;
  maximum : Error_metric.t;
  violations : int64 option;
  first_violation_group : int64 list option;
  samples : (int64 list * float) list;
}

let decode_status, encode_status =
  W.enum ~kind:"metric_status"
    [
      (Complete, "complete");
      (Nonfinite, "nonfinite");
      (Numerical_failure, "numerical_failure");
      (Resource_failure, "resource_failure");
      (Undefined, "undefined");
      (Unsupported, "unsupported");
    ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let count n = W.Obj.required o n W.nonneg_int64 in
      let* metric = W.Obj.required o "metric" Metric_id.decode in
      let* status = W.Obj.required o "status" decode_status in
      let* elements_used = count "elements_used" in
      let* groups_expected = count "groups_expected" in
      let* groups_checked = count "groups_checked" in
      let* minimum = W.Obj.required o "minimum" Error_metric.decode in
      let* maximum = W.Obj.required o "maximum" Error_metric.decode in
      let* violations = W.Obj.optional o "violations" W.nonneg_int64 in
      let* first_violation_group =
        W.Obj.optional o "first_violation_group" (W.list W.nonneg_int64)
      in
      let* samples =
        W.Obj.required o "samples"
          (W.list (W.pair (W.list W.nonneg_int64) W.float))
      in
      Ok
        {
          metric;
          status;
          elements_used;
          groups_expected;
          groups_checked;
          minimum;
          maximum;
          violations;
          first_violation_group;
          samples;
        })

let encode r =
  W.obj
    [
      W.field "metric" Metric_id.encode r.metric;
      W.field "status" encode_status r.status;
      W.field "elements_used" W.enc_int64 r.elements_used;
      W.field "groups_expected" W.enc_int64 r.groups_expected;
      W.field "groups_checked" W.enc_int64 r.groups_checked;
      W.field "minimum" Error_metric.encode r.minimum;
      W.field "maximum" Error_metric.encode r.maximum;
      W.field "violations" (W.enc_option W.enc_int64) r.violations;
      W.field "first_violation_group"
        (W.enc_option (W.enc_list W.enc_int64))
        r.first_violation_group;
      W.field "samples"
        (W.enc_list (W.enc_pair (W.enc_list W.enc_int64) W.enc_float))
        r.samples;
    ]
