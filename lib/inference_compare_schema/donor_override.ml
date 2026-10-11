module W = Wire
module Unit_id = W.Text (Id.Unit)
module Canonical = W.Text (Id.Canonical_value)

type t = { unit : Id.Unit.t; output : Id.Canonical_value.t; donor : Side.t }

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* unit = W.Obj.required o "unit" Unit_id.decode in
      let* output = W.Obj.required o "output" Canonical.decode in
      let* donor = W.Obj.required o "donor" Side.decode in
      Ok { unit; output; donor })

let encode d =
  W.obj
    [
      W.field "unit" Unit_id.encode d.unit;
      W.field "output" Canonical.encode d.output;
      W.field "donor" Side.encode d.donor;
    ]
