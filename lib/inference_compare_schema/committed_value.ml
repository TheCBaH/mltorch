module W = Wire
module Canonical = W.Text (Id.Canonical_value)
module Instance = W.Text (Id.Value_instance)
module Recipe_id = W.Text (Id.Recipe)

type t = {
  key : Id.Canonical_value.t;
  donor : Side.t;
  selected_instance : Id.Value_instance.t;
  converted_instance : Id.Value_instance.t;
  recipe : Id.Recipe.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* key = W.Obj.required o "key" Canonical.decode in
      let* donor = W.Obj.required o "donor" Side.decode in
      let* selected_instance =
        W.Obj.required o "selected_instance" Instance.decode
      in
      let* converted_instance =
        W.Obj.required o "converted_instance" Instance.decode
      in
      let* recipe = W.Obj.required o "recipe" Recipe_id.decode in
      Ok { key; donor; selected_instance; converted_instance; recipe })

let encode v =
  W.obj
    [
      W.field "key" Canonical.encode v.key;
      W.field "donor" Side.encode v.donor;
      W.field "selected_instance" Instance.encode v.selected_instance;
      W.field "converted_instance" Instance.encode v.converted_instance;
      W.field "recipe" Recipe_id.encode v.recipe;
    ]
