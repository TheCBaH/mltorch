module W = Wire
module Comparator = W.Text (Id.Comparator)

type t =
  | Exact_boolean
  | Exact_integer
  | Floating of Float_policy.t
  | Registered of {
      comparator : Id.Comparator.t;
      policy_digest : Digest.Comparison_policy.t;
    }

let decode : t W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "exact_boolean" -> Ok Exact_boolean
      | "exact_integer" -> Ok Exact_integer
      | "floating" ->
          let* p = W.Obj.required o "policy" Float_policy.decode in
          Ok (Floating p)
      | "registered" ->
          let* comparator = W.Obj.required o "comparator" Comparator.decode in
          let* policy_digest =
            W.Obj.required o "policy_digest" Wire_digest.comparison_policy
          in
          Ok (Registered { comparator; policy_digest })
      | tag -> W.unknown_tag ~kind:"element_policy" tag)

let encode = function
  | Exact_boolean -> W.tagged "exact_boolean" []
  | Exact_integer -> W.tagged "exact_integer" []
  | Floating p -> W.tagged "floating" [ W.field "policy" Float_policy.encode p ]
  | Registered { comparator; policy_digest } ->
      W.tagged "registered"
        [
          W.field "comparator" Comparator.encode comparator;
          W.field "policy_digest" Wire_digest.encode_comparison_policy
            policy_digest;
        ]
