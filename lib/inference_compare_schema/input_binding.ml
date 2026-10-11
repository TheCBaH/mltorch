module W = Wire
module Unit_id = W.Text (Id.Unit)

type t = {
  unit : Id.Unit.t;
  side : Side.t;
  tensor : Tensor_ref.t;
  from_common : Recipe.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* unit = W.Obj.required o "unit" Unit_id.decode in
      let* side = W.Obj.required o "side" Side.decode in
      let* tensor = W.Obj.required o "tensor" Tensor_ref.decode in
      let* from_common = W.Obj.required o "from_common" Recipe.decode in
      Ok { unit; side; tensor; from_common })

let encode b =
  W.obj
    [
      W.field "unit" Unit_id.encode b.unit;
      W.field "side" Side.encode b.side;
      W.field "tensor" Tensor_ref.encode b.tensor;
      W.field "from_common" Recipe.encode b.from_common;
    ]
