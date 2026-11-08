(** Final engine selection for SSA contractions. The operation's descriptor
    remains its generic semantics; this pass records its implementation before
    lowering or estimation. It preserves an already-selected legal engine. *)

module Unsupported : sig
  type t = {
    capabilities : Contraction.Select.Capabilities.t;
    desc : Contraction.t;
    engine : Contraction.Select.Engine.t;
  }
end

module Conflict : sig
  type t = {
    desc : Contraction.t;
    first : Contraction.Select.Engine.t;
    second : Contraction.Select.Engine.t;
  }
end

type error =
  [ `Contraction_engine_conflict of Conflict.t
  | `Unsupported_contraction_engine of Unsupported.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val program :
  capabilities:Contraction.Select.Capabilities.t ->
  Ssa_program.t ->
  (Ssa_program.t, [> error ]) Err.t
(** Only vector/matrix helpers are selected here. A contraction needing loops
    must be expanded before this stage; no scalar helper is supplied. Two
    different engines for one descriptor are refused: current helper bindings
    share one symbol per descriptor. *)

val choices :
  Ssa_program.t -> (Contraction.t * Contraction.Select.Engine.t) list
(** Recorded choices, in operation order, duplicates included. *)
