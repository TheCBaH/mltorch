module Ident = Js_ident
module Global = Js_global

type unop = Neg | Not | Plus

(* Alphabetical; precedence lives in [Js_print]'s table. The bitwise operators
   exist only for the 16-bit pattern helpers, and only [Js_build.Bits] makes
   them. *)
type binop =
  | Add
  | And
  | Bit_and
  | Bit_or
  | Div
  | Eq_strict
  | Ge
  | Gt
  | Le
  | Lt
  | Mul
  | Ne_strict
  | Or
  | Shl
  | Shr
  | Sub

type assign_op = Eq | Minus_eq | Plus_eq

type expr =
  | Array of expr list
  | Bigint of int64
  | Binary of binop * expr * expr
  | Bool of bool
  | Call of expr * expr list
  | Cond of expr * expr * expr
  | Global of Global.t
  | Index of expr * expr
  | Member of expr * Ident.t
  | New of expr * expr list
  | Null
  | Number of float
  | Object of (Ident.t * expr) list
  | String of string
  | Unary of unop * expr
  | Var of Ident.t

type lvalue = Lindex of expr * expr | Lvar of Ident.t

module rec Stmt : sig
  type t =
    | Assign of lvalue * assign_op * expr
    | Const of Ident.t * expr
    | Expr of expr
    | For of { var : Ident.t; init : expr; test : expr; body : t list }
    | Function of Func.t
    | If of expr * t list * t list
    | Let of Ident.t * expr
    | Return of expr option
end =
  Stmt

and Func : sig
  type t = { name : Ident.t; params : Ident.t list; body : Stmt.t list }
end =
  Func

type stmt = Stmt.t

module Program = struct
  type t = { prelude : Stmt.t list; entry : Func.t }
end
