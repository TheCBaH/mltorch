(** Ordinals the SSA emitters hand out and the generated code names. Each is a
    fresh type, so a buffer's parameter position cannot be passed where a
    failure site is wanted. *)

module Param : Core.Tagged_int.S
(** A buffer's position among the generated function's buffer parameters. *)

module Site : Core.Tagged_int.S
(** A failure site's position in the table a decoder reads a record against. *)
