(* Materialized constant-fold observations. A trace records only successful
   [Eval_direct]-backed folds; symbolic Const-SSA folds are deliberately absent
   because the trace is the baseline used to measure language coverage. *)

module Trace : sig
  type event

  val canonical : event list -> event list
  val op : event -> Graph_ir.op
  val pp : Format.formatter -> event -> unit
end

val pass : Pass.t
val pass_with_trace : (Trace.event -> unit) -> Pass.t
