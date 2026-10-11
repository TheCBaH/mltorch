module W = Wire
module Comparison = W.Text (Id.Comparison)
module Unit_id = W.Text (Id.Unit)
module Canonical = W.Text (Id.Canonical_value)
module Pair = W.Text (Id.Member_pair)
module Occurrence = W.Text (Id.Output_occurrence)

type kind =
  | Canonical_candidate of Id.Canonical_value.t
  | Member_pair of Id.Member_pair.t

type t = {
  comparison : Id.Comparison.t;
  unit : Id.Unit.t;
  kind : kind;
  spec : Tensor_spec.t;
  left_anchors : Tensor_ref.t list;
  right_anchors : Tensor_ref.t list;
  final_occurrences : Id.Output_occurrence.t list;
}

let decode_kind : kind W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "canonical_candidate" ->
          let* c = W.Obj.required o "key" Canonical.decode in
          Ok (Canonical_candidate c)
      | "member_pair" ->
          let* p = W.Obj.required o "pair" Pair.decode in
          Ok (Member_pair p)
      | tag -> W.unknown_tag ~kind:"site_kind" tag)

let encode_kind = function
  | Canonical_candidate c ->
      W.tagged "canonical_candidate" [ W.field "key" Canonical.encode c ]
  | Member_pair p -> W.tagged "member_pair" [ W.field "pair" Pair.encode p ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let tensors = W.list Tensor_ref.decode in
      let* comparison = W.Obj.required o "comparison" Comparison.decode in
      let* unit = W.Obj.required o "unit" Unit_id.decode in
      let* kind = W.Obj.required o "kind" decode_kind in
      let* spec = W.Obj.required o "spec" Tensor_spec.decode in
      let* left_anchors = W.Obj.required o "left_anchors" tensors in
      let* right_anchors = W.Obj.required o "right_anchors" tensors in
      let* final_occurrences =
        W.Obj.required o "final_occurrences" (W.list Occurrence.decode)
      in
      Ok
        {
          comparison;
          unit;
          kind;
          spec;
          left_anchors;
          right_anchors;
          final_occurrences;
        })

let encode s =
  let tensors = W.enc_list Tensor_ref.encode in
  W.obj
    [
      W.field "comparison" Comparison.encode s.comparison;
      W.field "unit" Unit_id.encode s.unit;
      W.field "kind" encode_kind s.kind;
      W.field "spec" Tensor_spec.encode s.spec;
      W.field "left_anchors" tensors s.left_anchors;
      W.field "right_anchors" tensors s.right_anchors;
      W.field "final_occurrences"
        (W.enc_list Occurrence.encode)
        s.final_occurrences;
    ]
