module W = Wire
module Route_id = W.Text (Id.Value_route)

type t = {
  route_id : Id.Value_route.t;
  sources : Tensor_ref.t list;
  destination : Tensor_ref.t;
  steps : Conversion_step.t list;
  claim : Claim.t;
  lossy : bool;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* route_id = W.Obj.required o "route_id" Route_id.decode in
      let* sources = W.Obj.required o "sources" (W.list Tensor_ref.decode) in
      let* destination = W.Obj.required o "destination" Tensor_ref.decode in
      let* steps = W.Obj.required o "steps" (W.list Conversion_step.decode) in
      let* claim = W.Obj.required o "claim" Claim.decode in
      let* lossy = W.Obj.required o "lossy" W.bool in
      Ok { route_id; sources; destination; steps; claim; lossy })

let encode r =
  W.obj
    [
      W.field "route_id" Route_id.encode r.route_id;
      W.field "sources" (W.enc_list Tensor_ref.encode) r.sources;
      W.field "destination" Tensor_ref.encode r.destination;
      W.field "steps" (W.enc_list Conversion_step.encode) r.steps;
      W.field "claim" Claim.encode r.claim;
      W.field "lossy" W.enc_bool r.lossy;
    ]
