module W = Wire
module Comparison = W.Text (Id.Comparison)

type verdict =
  | Metadata_error
  | Metric_unavailable
  | Mismatch
  | Pass
  | Unsupported

type t = {
  comparison_id : Id.Comparison.t;
  policy_digest : Digest.Comparison_policy.t;
  elements_expected : int64;
  elements_checked : int64;
  element : Element_result.t option;
  tensor_metrics : Metric_result.t list;
  verdict : verdict;
}

let decode_verdict, encode_verdict =
  W.enum ~kind:"verdict"
    [
      (Metadata_error, "metadata_error");
      (Metric_unavailable, "metric_unavailable");
      (Mismatch, "mismatch");
      (Pass, "pass");
      (Unsupported, "unsupported");
    ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* comparison_id = W.Obj.required o "comparison_id" Comparison.decode in
      let* policy_digest =
        W.Obj.required o "policy_digest" Wire_digest.comparison_policy
      in
      let* elements_expected =
        W.Obj.required o "elements_expected" W.nonneg_int64
      in
      let* elements_checked =
        W.Obj.required o "elements_checked" W.nonneg_int64
      in
      let* element = W.Obj.optional o "element" Element_result.decode in
      let* tensor_metrics =
        W.Obj.required o "tensor_metrics" (W.list Metric_result.decode)
      in
      let* verdict = W.Obj.required o "verdict" decode_verdict in
      Ok
        {
          comparison_id;
          policy_digest;
          elements_expected;
          elements_checked;
          element;
          tensor_metrics;
          verdict;
        })

let encode r =
  W.obj
    [
      W.field "comparison_id" Comparison.encode r.comparison_id;
      W.field "policy_digest" Wire_digest.encode_comparison_policy
        r.policy_digest;
      W.field "elements_expected" W.enc_int64 r.elements_expected;
      W.field "elements_checked" W.enc_int64 r.elements_checked;
      W.field "element" (W.enc_option Element_result.encode) r.element;
      W.field "tensor_metrics"
        (W.enc_list Metric_result.encode)
        r.tensor_metrics;
      W.field "verdict" encode_verdict r.verdict;
    ]
