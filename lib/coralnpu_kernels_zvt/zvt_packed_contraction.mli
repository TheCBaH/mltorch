(** A [Paired] bfloat16 contraction descriptor as one function on the matrix
    engine at the rate of its tile blocks: the suite's double-buffered blocks of
    two tile rows by two tile columns, fed from panels the core packs (B once a
    batch, A a block of rows at a time) in the layout the blocks read: per tile,
    k-major, the two tiles of a row or column side by side.

    The ILP32F convention of {!Zvt_contraction}: [a0]-[a2] the operands'
    addresses, [0] in [a0], under the descriptor's symbol. The panels are the
    function's own [.bss] ({!panel_bytes}), zero before the first call: the
    padding the blocks read past a panel's end is never written. Blocks of 32
    rows, then the rest in blocks of at most 16; k is taken to a multiple of
    four with zero steps, which add nothing to a pair's sum.

    An operand whose k rows are contiguous runs of its tile's indices (A
    transposed, B as it is) is read where it is, by row pitch, when k needs no
    zero steps: no packing, but the blocks read a few k rows past its end, which
    the caller must leave readable. With [stage_b] a B read in place is first
    copied, rows and pitch as they are, into a panel of the function's own by
    the core: for B in SRAM and reused by more than one block of A, whose reads
    from SRAM are four bytes a cycle against the DTCM's. The other is packed (a
    row-major A by a strided load for two k steps, the slow case). C has columns
    or rows one element apart: with rows, the contraction is run as its
    {!Contraction.transpose}, the operands exchanged, which makes B's role the
    transposed one and a k-major result the next layer's A. It refuses another
    numerics or format. *)

(** Fault injection for the evidence suite: each is one deliberate defect a run
    must detect. No consumer passes one. *)
module Mutation : sig
  type t =
    | A_pitch  (** A's rows one element further apart than they are *)
    | Short_copy  (** the staged B's copy 64 elements short *)
    | Short_k  (** the last k step of A and B not packed *)
end

val overread : Contraction.t -> int
(** The bytes past the end of an operand read in place that the blocks read, for
    the caller to leave readable: none when both operands are packed. *)

val fast : Contraction.t -> bool
(** Every operand the function packs is packed by words, two k steps a strided
    load, which is where it beats the plain function; an operand with an odd
    stride packs an element a load. *)

(** Where a B read in place is read from. *)
module Stage : sig
  type t =
    | Always  (** copied into a panel of the function's own *)
    | Outside of { base : int; bytes : int }
        (** copied unless it lies in the DTCM [bytes] from [base], which the
            function checks at run time *)
end

val reuses_b : Contraction.t -> bool
(** B is read in place and by more than one block of A: where staging it into
    the DTCM pays when it lies in SRAM. *)

val panel_bytes : ?stage_b:Stage.t -> Contraction.t -> (int, string) result
(** The [.bss] the function declares: B's packed panels and the largest block of
    A's, none for an operand read in place. *)

val module_ :
  ?mutation:Mutation.t ->
  ?stage_b:Stage.t ->
  Contraction.t ->
  (Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_, string) result
