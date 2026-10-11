module W = Wire
module Unit_id = W.Text (Id.Unit)
module Pair_id = W.Text (Id.Member_pair)

type t = {
  unit_id : Id.Unit.t;
  left : Cluster.t;
  right : Cluster.t;
  output_pairs : Id.Member_pair.t list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* unit_id = W.Obj.required o "unit_id" Unit_id.decode in
      let* left = W.Obj.required o "left" Cluster.decode in
      let* right = W.Obj.required o "right" Cluster.decode in
      let* output_pairs =
        W.Obj.required o "output_pairs" (W.list Pair_id.decode)
      in
      Ok { unit_id; left; right; output_pairs })

let encode u =
  W.obj
    [
      W.field "unit_id" Unit_id.encode u.unit_id;
      W.field "left" Cluster.encode u.left;
      W.field "right" Cluster.encode u.right;
      W.field "output_pairs" (W.enc_list Pair_id.encode) u.output_pairs;
    ]
