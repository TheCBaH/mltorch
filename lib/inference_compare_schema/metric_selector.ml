module W = Wire
module Comparison = W.Text (Id.Comparison)
module Canonical = W.Text (Id.Canonical_value)
module Pair = W.Text (Id.Member_pair)
module Unit_id = W.Text (Id.Unit)
module Occurrence = W.Text (Id.Output_occurrence)

type anchor_quantifier = Any_anchor | Every_anchor

type t =
  | All_of of t list
  | Any_of of t list
  | Any_site
  | Canonical_value of Id.Canonical_value.t
  | Comparison of Id.Comparison.t
  | Final_output of Id.Output_occurrence.t
  | Member_pair of Id.Member_pair.t
  | Not of t
  | On_side of {
      side : Side.t;
      anchors : anchor_quantifier;
      tensor : Tensor_selector.t;
    }
  | Unit of Id.Unit.t

let decode_quantifier, encode_quantifier =
  W.enum ~kind:"anchor_quantifier"
    [ (Any_anchor, "any_anchor"); (Every_anchor, "every_anchor") ]

let nonempty items =
  if items = [] then W.fail (W.Constraint Empty_list) else Ok items

let rec decode : t W.decoder =
 fun j ->
  W.variant
    (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "all_of" ->
          let* items = W.Obj.required o "items" (W.list decode) in
          let* items = nonempty items in
          Ok (All_of items)
      | "any_of" ->
          let* items = W.Obj.required o "items" (W.list decode) in
          let* items = nonempty items in
          Ok (Any_of items)
      | "any_site" -> Ok Any_site
      | "canonical_value" ->
          let* c = W.Obj.required o "key" Canonical.decode in
          Ok (Canonical_value c)
      | "comparison" ->
          let* c = W.Obj.required o "comparison" Comparison.decode in
          Ok (Comparison c)
      | "final_output" ->
          let* c = W.Obj.required o "occurrence" Occurrence.decode in
          Ok (Final_output c)
      | "member_pair" ->
          let* p = W.Obj.required o "pair" Pair.decode in
          Ok (Member_pair p)
      | "not" ->
          let* t = W.Obj.required o "operand" decode in
          Ok (Not t)
      | "on_side" ->
          let* side = W.Obj.required o "side" Side.decode in
          let* anchors = W.Obj.required o "anchors" decode_quantifier in
          let* tensor = W.Obj.required o "tensor" Tensor_selector.decode in
          Ok (On_side { side; anchors; tensor })
      | "unit" ->
          let* u = W.Obj.required o "unit" Unit_id.decode in
          Ok (Unit u)
      | tag -> W.unknown_tag ~kind:"metric_selector" tag)
    j

let rec encode = function
  | All_of items ->
      W.tagged "all_of" [ W.field "items" (W.enc_list encode) items ]
  | Any_of items ->
      W.tagged "any_of" [ W.field "items" (W.enc_list encode) items ]
  | Any_site -> W.tagged "any_site" []
  | Canonical_value c ->
      W.tagged "canonical_value" [ W.field "key" Canonical.encode c ]
  | Comparison c ->
      W.tagged "comparison" [ W.field "comparison" Comparison.encode c ]
  | Final_output o ->
      W.tagged "final_output" [ W.field "occurrence" Occurrence.encode o ]
  | Member_pair p -> W.tagged "member_pair" [ W.field "pair" Pair.encode p ]
  | Not t -> W.tagged "not" [ W.field "operand" encode t ]
  | On_side { side; anchors; tensor } ->
      W.tagged "on_side"
        [
          W.field "side" Side.encode side;
          W.field "anchors" encode_quantifier anchors;
          W.field "tensor" Tensor_selector.encode tensor;
        ]
  | Unit u -> W.tagged "unit" [ W.field "unit" Unit_id.encode u ]
