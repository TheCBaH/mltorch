(** Linear-scan allocation over a selected program. *)

open Machine_ir

module type POOL = Mir_scan.POOL

module Mutation = Mir_scan.Mutation

module Make (T : Mir_sel.TARGET) (_ : POOL) : sig
  val allocate :
    ?mutation:Mutation.t ->
    Mir_sel.Make(T).Verified.t ->
    (T.op, T.test) Mir_phys.Program.t
end
