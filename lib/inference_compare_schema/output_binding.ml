module W = Wire
module Unit_id = W.Text (Id.Unit)
module Canonical = W.Text (Id.Canonical_value)
module Comparison = W.Text (Id.Comparison)

type t = {
  unit : Id.Unit.t;
  key : Id.Canonical_value.t;
  from_left_outputs : Recipe.t;
  from_right_outputs : Recipe.t;
  comparison_id : Id.Comparison.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* unit = W.Obj.required o "unit" Unit_id.decode in
      let* key = W.Obj.required o "key" Canonical.decode in
      let* from_left_outputs =
        W.Obj.required o "from_left_outputs" Recipe.decode
      in
      let* from_right_outputs =
        W.Obj.required o "from_right_outputs" Recipe.decode
      in
      let* comparison_id = W.Obj.required o "comparison_id" Comparison.decode in
      Ok { unit; key; from_left_outputs; from_right_outputs; comparison_id })

let encode b =
  W.obj
    [
      W.field "unit" Unit_id.encode b.unit;
      W.field "key" Canonical.encode b.key;
      W.field "from_left_outputs" Recipe.encode b.from_left_outputs;
      W.field "from_right_outputs" Recipe.encode b.from_right_outputs;
      W.field "comparison_id" Comparison.encode b.comparison_id;
    ]
