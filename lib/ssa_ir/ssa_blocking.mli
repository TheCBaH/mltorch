(** The factors a target and the blocking passes agree on. Each is a count of
    copies of a loop body, so a row-block factor cannot be passed where an
    unroll factor or a partial-sum count is wanted. *)

module Rows : Core.Tagged_int.S
(** Rows of a scalar loop jammed around one vector loop; 1 is no blocking. *)

module Unroll : Core.Tagged_int.S
(** Copies of a vector loop body interleaved; 1 is no unrolling. *)

module Group : Core.Tagged_int.S
(** Outputs of a reduction jammed into one loop body; 2 or more. *)

module Parts : Core.Tagged_int.S
(** Independent partial sums a reduction is split into; 1 is sequential. *)
