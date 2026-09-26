(** The only code that produces JavaScript text. Deterministic: printing the
    same value twice is byte-identical, layout is two-space indentation and one
    statement per line, and no [Format] box (whose output would depend on a
    margin setting) is involved. *)

type parens =
  [ `All  (** every compound child parenthesised: the differential's oracle *)
  | `Minimal  (** a child only when the precedence table requires it *) ]

val expr : ?parens:parens -> Js_ast.expr -> string
val stmts : ?parens:parens -> Js_ast.Stmt.t list -> string

val script : ?parens:parens -> Js_ast.Program.t -> string
(** ["use strict";], the prelude, then [function entry(...) {...}]: what the
    node gate writes to a file. *)

val factory_body : ?parens:parens -> Js_ast.Program.t -> string
(** [script] followed by [return entry;]: what [new Function] receives. *)
