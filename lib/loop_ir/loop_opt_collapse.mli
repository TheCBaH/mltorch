(** Loop collapsing: a perfectly nested pair of constant loops whose variables
    reach the body only through dense, contiguous access offsets becomes one
    loop over the product extent, those accesses flat. Pairs collapse bottom-up,
    so an elementwise nest becomes a single loop. *)

val run : Loop_program.t -> Loop_program.t

val with_ratio :
  (c1:int -> c2:int -> n_inner:int -> bool) -> Loop_program.t -> Loop_program.t
(** [run] with the contiguity test ([c1 = n_inner * c2] for the outer and inner
    variable's coefficients in an access's offset) replaced, for the mutation
    test. *)
