module W = Wire

type t = {
  dialect : Id.Dialect.t;
  version : Id.Graph_version.t;
  graph_digest : Digest.Graph.t;
}

let compare a b =
  match Id.Dialect.compare a.dialect b.dialect with
  | 0 -> (
      match Id.Graph_version.compare a.version b.version with
      | 0 -> Digest.Graph.compare a.graph_digest b.graph_digest
      | c -> c)
  | c -> c

let equal a b = compare a b = 0

let pp ppf k =
  Fmt.pf ppf "%a/%a@%a" Id.Dialect.pp k.dialect Id.Graph_version.pp k.version
    Digest.Graph.pp k.graph_digest

module Dialect = W.Text (Id.Dialect)
module Version = W.Text (Id.Graph_version)

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* dialect = W.Obj.required o "dialect" Dialect.decode in
      let* version = W.Obj.required o "version" Version.decode in
      let* graph_digest = W.Obj.required o "graph_digest" Wire_digest.graph in
      Ok { dialect; version; graph_digest })

let encode k =
  W.obj
    [
      W.field "dialect" Dialect.encode k.dialect;
      W.field "version" Version.encode k.version;
      W.field "graph_digest" Wire_digest.encode_graph k.graph_digest;
    ]
