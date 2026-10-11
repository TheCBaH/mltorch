module W = Wire
module Unit_id = W.Text (Id.Unit)
module Canonical = W.Text (Id.Canonical_value)

type t = {
  unit : Id.Unit.t;
  side : Side.t;
  canonical : Id.Canonical_value.t;
  from_imported_inputs : Recipe.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* unit = W.Obj.required o "unit" Unit_id.decode in
      let* side = W.Obj.required o "side" Side.decode in
      let* canonical = W.Obj.required o "canonical" Canonical.decode in
      let* from_imported_inputs =
        W.Obj.required o "from_imported_inputs" Recipe.decode
      in
      Ok { unit; side; canonical; from_imported_inputs })

let encode c =
  W.obj
    [
      W.field "unit" Unit_id.encode c.unit;
      W.field "side" Side.encode c.side;
      W.field "canonical" Canonical.encode c.canonical;
      W.field "from_imported_inputs" Recipe.encode c.from_imported_inputs;
    ]
