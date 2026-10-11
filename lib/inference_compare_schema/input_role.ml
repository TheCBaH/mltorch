module W = Wire
module User_input = W.Text (Name.User_input)
module Constant = W.Text (Name.Constant)
module Explicit_state = W.Text (Name.Explicit_state)

type t =
  | Constant of Name.Constant.t
  | Explicit_state of Name.Explicit_state.t
  | User_input of Name.User_input.t

let compare a b =
  match (a, b) with
  | Constant x, Constant y -> Name.Constant.compare x y
  | Explicit_state x, Explicit_state y -> Name.Explicit_state.compare x y
  | User_input x, User_input y -> Name.User_input.compare x y
  | Constant _, _ -> -1
  | _, Constant _ -> 1
  | Explicit_state _, _ -> -1
  | _, Explicit_state _ -> 1

let equal a b = compare a b = 0

let decode : t W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "constant" ->
          let* n = W.Obj.required o "name" Constant.decode in
          Ok (Constant n)
      | "explicit_state" ->
          let* n = W.Obj.required o "name" Explicit_state.decode in
          Ok (Explicit_state n)
      | "user_input" ->
          let* n = W.Obj.required o "name" User_input.decode in
          Ok (User_input n)
      | tag -> W.unknown_tag ~kind:"input_role" tag)

let encode = function
  | Constant n -> W.tagged "constant" [ W.field "name" Constant.encode n ]
  | Explicit_state n ->
      W.tagged "explicit_state" [ W.field "name" Explicit_state.encode n ]
  | User_input n -> W.tagged "user_input" [ W.field "name" User_input.encode n ]
