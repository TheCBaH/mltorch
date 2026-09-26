(** Lexical scope checking for {!Js_ast}: every identifier a program uses is
    bound, and no block declares a name twice. {!Js_global.t} is closed, so a
    global cannot be misspelled into a free variable. *)

module Fault : sig
  type t = Duplicate of Js_ident.t | Unbound of Js_ident.t

  val pp : t Fmt.t
end

val closed : Js_ast.Program.t -> (unit, Fault.t list) result
(** Scopes: program-level functions ([function] declarations are hoisted) and
    [const]/[let] (bound from the next statement), parameters (in the body's
    block), block-level [let]/[const], and the [For] variable in its test and
    body only. Reports every fault, in walk order. *)

val free : Js_ast.Stmt.t list -> Js_ident.Set.t
(** The identifiers the statements use without binding, by the same walk. *)
