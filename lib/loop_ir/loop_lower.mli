(** Lowering a [Fusion_plan.t] to a [Loop_program.t]. Everything it does not
    handle is a typed [`Unsupported], never a partial program. *)

type error = [ `Unsupported of Loop_unsupported.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val lower : Fusion_plan.t -> (Loop_program.t, error) Err.t
(** [lower_unoptimized] followed by {!Loop_opt.run}. Every consumer
    ([Loop_node_program], [Loop_region_program], [Loop_check]) reaches a program
    through this, so the interpreter and both JS backends always see the
    optimized form. *)

val lower_structured : Fusion_plan.t -> (Loop_program.t, error) Err.t
(** {!lower_unoptimized} with each float sum kept as one
    {!Loop_stmt.Reduce_sum}: the form the performance planner reads before any
    pass has reshaped the reduction. {!Loop_sum.program} of it is exactly
    {!lower_unoptimized}'s program. *)

val lower_unoptimized : Fusion_plan.t -> (Loop_program.t, error) Err.t
(** The raw lowering, with no {!Loop_opt} pass applied. Kept for the
    optimized-vs-raw differential and for inspecting a regression. *)
