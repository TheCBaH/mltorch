module W = Wire
module Occurrence = W.Text (Id.Output_occurrence)
module Canonical = W.Text (Id.Canonical_value)

(* One position of the endpoint graphs' ordered final outputs: the left and
   right tensors at that position and the canonical value they represent. A
   repeated tensor appears once per occurrence. *)
type t = {
  occurrence : Id.Output_occurrence.t;
  canonical : Id.Canonical_value.t;
  left : Tensor_ref.t;
  right : Tensor_ref.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* occurrence = W.Obj.required o "occurrence" Occurrence.decode in
      let* canonical = W.Obj.required o "canonical" Canonical.decode in
      let* left = W.Obj.required o "left" Tensor_ref.decode in
      let* right = W.Obj.required o "right" Tensor_ref.decode in
      Ok { occurrence; canonical; left; right })

let encode f =
  W.obj
    [
      W.field "occurrence" Occurrence.encode f.occurrence;
      W.field "canonical" Canonical.encode f.canonical;
      W.field "left" Tensor_ref.encode f.left;
      W.field "right" Tensor_ref.encode f.right;
    ]
