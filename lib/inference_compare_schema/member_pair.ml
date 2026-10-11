module W = Wire
module Pair_id = W.Text (Id.Member_pair)
module Comparison_id = W.Text (Id.Comparison)

type t = {
  pair_id : Id.Member_pair.t;
  comparison_id : Id.Comparison.t;
  left : Tensor_ref.t;
  right : Tensor_ref.t;
  claim : Claim.t;
  left_to_comparison : Conversion_step.t list;
  right_to_comparison : Conversion_step.t list;
  comparison_spec : Tensor_spec.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let steps = W.list Conversion_step.decode in
      let* pair_id = W.Obj.required o "pair_id" Pair_id.decode in
      let* comparison_id =
        W.Obj.required o "comparison_id" Comparison_id.decode
      in
      let* left = W.Obj.required o "left" Tensor_ref.decode in
      let* right = W.Obj.required o "right" Tensor_ref.decode in
      let* claim = W.Obj.required o "claim" Claim.decode in
      let* left_to_comparison = W.Obj.required o "left_to_comparison" steps in
      let* right_to_comparison = W.Obj.required o "right_to_comparison" steps in
      let* comparison_spec =
        W.Obj.required o "comparison_spec" Tensor_spec.decode
      in
      Ok
        {
          pair_id;
          comparison_id;
          left;
          right;
          claim;
          left_to_comparison;
          right_to_comparison;
          comparison_spec;
        })

let encode p =
  let steps = W.enc_list Conversion_step.encode in
  W.obj
    [
      W.field "pair_id" Pair_id.encode p.pair_id;
      W.field "comparison_id" Comparison_id.encode p.comparison_id;
      W.field "left" Tensor_ref.encode p.left;
      W.field "right" Tensor_ref.encode p.right;
      W.field "claim" Claim.encode p.claim;
      W.field "left_to_comparison" steps p.left_to_comparison;
      W.field "right_to_comparison" steps p.right_to_comparison;
      W.field "comparison_spec" Tensor_spec.encode p.comparison_spec;
    ]
