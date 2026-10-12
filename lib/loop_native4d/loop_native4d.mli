(** The Native4D dialect for the Loop bundle builders.

    Every entry point is the Native4D twin of the call
    {!Loop_ir.Loop_dialect.native} makes, so a bundle built here is scheduled,
    lowered and bound exactly as a Native one is. *)

val dialect :
  (Native4d.Op.t, Native4d.Eval_direct4.error) Loop_ir.Loop_dialect.t

val build :
  ?limits:Kernel.Limits.t ->
  ?config:Storage_script.Config.t ->
  ?plan:Storage_plan.t ->
  Native4d.Graph.graph ->
  ( Loop_ir.Loop_bundle.t,
    Native4d.Eval_direct4.error Loop_ir.Loop_bundle.generic_error )
  Err.t
(** {!Loop_ir.Loop_bundle.build_with} with {!dialect}. *)

val pp_error :
  Format.formatter ->
  [< Native4d.Eval_direct4.error Loop_ir.Loop_bundle.generic_error ] ->
  unit
