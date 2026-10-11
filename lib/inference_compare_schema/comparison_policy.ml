module W = Wire

type t = {
  element : Element_policy.t option;
  statistics : Difference_statistics.policy;
  relative_scale : Float_policy.scale;
  tensor_metrics : Tensor_metric.t list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* element = W.Obj.optional o "element" Element_policy.decode in
      let* statistics =
        W.Obj.required o "statistics" Difference_statistics.decode_policy
      in
      let* relative_scale =
        W.Obj.required o "relative_scale" Float_policy.decode_scale
      in
      let* tensor_metrics =
        W.Obj.required o "tensor_metrics" (W.list Tensor_metric.decode)
      in
      Ok { element; statistics; relative_scale; tensor_metrics })

let encode p =
  W.obj
    [
      W.field "element" (W.enc_option Element_policy.encode) p.element;
      W.field "statistics" Difference_statistics.encode_policy p.statistics;
      W.field "relative_scale" Float_policy.encode_scale p.relative_scale;
      W.field "tensor_metrics"
        (W.enc_list Tensor_metric.encode)
        p.tensor_metrics;
    ]

(* Metric order within a policy is not semantic: results are keyed by metric
   id. *)
let sort_metrics =
  List.stable_sort (fun (a : Tensor_metric.t) (b : Tensor_metric.t) ->
      Id.Tensor_metric.compare a.id b.id)

let canonicalize p = { p with tensor_metrics = sort_metrics p.tensor_metrics }

let preimage p =
  Digest_preimage.of_json Comparison_policy (encode (canonicalize p))
