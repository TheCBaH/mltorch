(** The scalar-lane reference for a vector program: every vector value becomes
    one scalar value per lane, every vector operation the scalar operations it
    stands for, and a loop that carries a vector carries its lanes. Built from
    the vector program alone, so running the result is an answer the vector
    interpreter did not produce.

    Memory operations become one scalar access per lane at the lane's own
    coordinates, with the same proofs the vector operation carried: the expanded
    program verifies only if every lane is provably in bounds, which checks the
    vector operation's proof a second way. A scalar program expands to itself.
*)

val program : Ssa_program.t -> Ssa_program.t
(** The program must verify; [Invalid_argument] otherwise. *)
