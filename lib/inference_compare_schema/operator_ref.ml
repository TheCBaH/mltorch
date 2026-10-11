module W = Wire
module Dialect = W.Text (Id.Dialect)
module Operation = W.Text (Id.Operator)

type t = { dialect : Id.Dialect.t; operation : Id.Operator.t }

let compare a b =
  match Id.Dialect.compare a.dialect b.dialect with
  | 0 -> Id.Operator.compare a.operation b.operation
  | c -> c

let equal a b = compare a b = 0

let pp ppf r =
  Fmt.pf ppf "%a.%a" Id.Dialect.pp r.dialect Id.Operator.pp r.operation

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* dialect = W.Obj.required o "dialect" Dialect.decode in
      let* operation = W.Obj.required o "operation" Operation.decode in
      Ok { dialect; operation })

let encode r =
  W.obj
    [
      W.field "dialect" Dialect.encode r.dialect;
      W.field "operation" Operation.encode r.operation;
    ]
