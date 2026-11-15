(** Ordinals the SSA emitters hand out and the generated code names. Each is a
    fresh type, so a buffer's parameter position cannot be passed where a
    failure site is wanted. *)

module Param : Core.Tagged_int.S
(** A buffer's position among the generated function's buffer parameters. *)

module Site : Core.Tagged_int.S
(** A failure site's position in the table a decoder reads a record against. *)

module Name : Core.Tagged_int.S
(** The ordinal in a generated variable's name ([v3], [t7]): a value's and a
    temporary's own sequence, neither a failure site nor a parameter. *)
