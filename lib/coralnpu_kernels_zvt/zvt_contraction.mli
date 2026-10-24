(** An [Ordered] contraction descriptor as one function of Zvt binary32 tile
    multiplies ([vtfmm.tvv], tk = 1, which round the product and then the sum),
    under the same ILP32F convention as {!Coralnpu_kernels.Rvv_contraction}:
    [a0]-[a2] the operands' addresses, [0] in [a0], under the same symbol. Each
    16x16 block of C (fewer rows or columns at the edges, through the tile's
    [tm] and [tn]) accumulates in tile [mt0] from zero, k ascending: per step
    A's column slice and B's row slice load straight from the operands
    ([vle32.v] or [vlse32.v], [vl] the slice's length) and one tile multiply
    adds their outer product. The rows go back by [vtse32]. It needs C's columns
    contiguous and refuses the fused requirement, which the engine does not
    compute. *)

(** Fault injection for the evidence suite: each is one deliberate kernel defect
    a run must detect. No consumer passes one. *)
module Mutation : sig
  type t =
    | Full_tn  (** a partial last column's tile written 16 columns wide *)
    | Swapped_operands  (** B's slice where A's goes, and A's where B's *)
end

val module_ :
  ?mutation:Mutation.t ->
  Contraction.t ->
  (Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_, string) result
