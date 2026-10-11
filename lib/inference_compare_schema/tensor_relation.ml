module W = Wire
module Relation_id = W.Text (Id.Tensor_relation)

type t = {
  relation_id : Id.Tensor_relation.t;
  left_members : Tensor_ref.t list;
  right_members : Tensor_ref.t list;
  member_pairs : Member_pair.t list;
  routes : Value_route.t list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let tensors = W.list Tensor_ref.decode in
      let* relation_id = W.Obj.required o "relation_id" Relation_id.decode in
      let* left_members = W.Obj.required o "left_members" tensors in
      let* right_members = W.Obj.required o "right_members" tensors in
      let* member_pairs =
        W.Obj.required o "member_pairs"
          (W.unique
             ~key:(fun (p : Member_pair.t) ->
               Id.Member_pair.to_string p.pair_id)
             (W.list Member_pair.decode))
      in
      let* routes =
        W.Obj.required o "routes"
          (W.unique
             ~key:(fun (v : Value_route.t) ->
               Id.Value_route.to_string v.route_id)
             (W.list Value_route.decode))
      in
      Ok { relation_id; left_members; right_members; member_pairs; routes })

let encode r =
  let tensors = W.enc_list Tensor_ref.encode in
  W.obj
    [
      W.field "relation_id" Relation_id.encode r.relation_id;
      W.field "left_members" tensors r.left_members;
      W.field "right_members" tensors r.right_members;
      W.field "member_pairs" (W.enc_list Member_pair.encode) r.member_pairs;
      W.field "routes" (W.enc_list Value_route.encode) r.routes;
    ]
