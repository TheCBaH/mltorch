module W = Wire
module Local = W.Text (Id.Tensor_local)

type t = { graph : Graph_key.t; scope : Scope.t; local_id : Id.Tensor_local.t }

let compare a b =
  match Graph_key.compare a.graph b.graph with
  | 0 -> (
      match Scope.compare a.scope b.scope with
      | 0 -> Id.Tensor_local.compare a.local_id b.local_id
      | c -> c)
  | c -> c

let equal a b = compare a b = 0

let pp ppf r =
  Fmt.pf ppf "%a:%a/%a" Graph_key.pp r.graph Scope.pp r.scope Id.Tensor_local.pp
    r.local_id

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* graph = W.Obj.required o "graph" Graph_key.decode in
      let* scope = W.Obj.required o "scope" Scope.decode in
      let* local_id = W.Obj.required o "local_id" Local.decode in
      Ok { graph; scope; local_id })

let encode r =
  W.obj
    [
      W.field "graph" Graph_key.encode r.graph;
      W.field "scope" Scope.encode r.scope;
      W.field "local_id" Local.encode r.local_id;
    ]

module Set = Set.Make (struct
  type nonrec t = t

  let compare = compare
end)

module Map = Map.Make (struct
  type nonrec t = t

  let compare = compare
end)
