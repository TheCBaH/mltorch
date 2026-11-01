(** An invocation's cycles on a Coral core as a weighted sum of features: the
    census's class counts, and for each hand kernel it calls the features that
    kernel's cost grows with. The weights ({!Rates.t}) are cycles per unit of
    feature, fitted to RTL measurements ({!Coral_fit}). *)

(** What the core has, which decides the kernel a contraction runs on. *)
module Core : sig
  type t = {
    matrix : bool;
    vector : bool;
    packed : Contraction.t -> bool;
        (** the matrix engine's contraction is the packed function (the route's
            own rule, which lives with the kernels) *)
  }

  val vme : packed:(Contraction.t -> bool) -> t
end

val features : Core.t -> Coral_census.t -> (string * float) list
(** Feature names: [class.<census class>], [invocation] (one per run),
    [zvt.<format>[.packed].call | .tiles.<C's region> | .rows_k.<A's> |
     .cols_k.<B's>] for the matrix engine's contraction (tiles of 16 x 16
    outputs; the A rows and the B columns each tile loads, times k),
    [rvv.call | .macs] for the vector unit's, [loops.macs] for one left in
    loops, and
    [attend.call | .qblocks.<Q's region> | .out.<O's> | .kblocks.<K's> | .scores
     | .dwork.<K's> | .stage_q.<Q's> | .stage_kv.<K's>] for the fused attention
    (blocks of 16 queries per head, their outputs times d, their key blocks, the
    padded score cells, those times the head dimension, and the Q and K, V
    elements staged into the DTCM). *)

module Rates : sig
  type t = (string * float) list
  (** cycles per unit; absent is zero *)

  val predict : t -> (string * float) list -> float
end
