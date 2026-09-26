(** Lowering a [Fusion_plan.t] to a [Loop_program.t]. Everything it does not
    handle is a typed [`Unsupported], never a partial program. *)

type error = [ `Unsupported of Loop_unsupported.t ]

val pp_error : Format.formatter -> [< error ] -> unit
val lower : Fusion_plan.t -> (Loop_program.t, error) Err.t
