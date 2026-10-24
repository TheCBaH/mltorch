(** A contraction descriptor as one RVV (Zve32f) function under the ILP32F
    calling convention: [a0], [a1], [a2] hold the addresses of A's, B's and C's
    first elements, the function returns [0] in [a0], and it writes C alone.
    Every extent and stride is the descriptor's, so the code is specialized to
    it.

    Columns of C are computed 32 at a time (an e32 m8 register group; fewer for
    the last tile), rows two or three at a time, each output's sum in k order
    from positive zero: [vfmacc.vf] per step under [Fused_sequential] (three
    rows, accumulators v8, v16, v24), [vfmul.vf] then [vfadd.vv] under [Ordered]
    (two rows, the product in v24). Each k step loads the rows' A values with
    [flw] before B's row ([vle32.v] when B's columns are contiguous, else
    [vlse32.v]). It saves and restores [s0]-[s3] and leaves the vector
    configuration undefined.

    With [stage] (default [false]: the subsystem, operands in SRAM) the function
    first copies A and B into its own DTCM panels (a [.bss] of their spans) by
    one DMA chain it builds from its arguments, starts and waits for, then
    computes from the panels; an operand not 16-byte aligned or a chain ending
    in error leaves it reading the operands in place. *)

(** Fault injection for the evidence suite: each is one deliberate kernel defect
    a run must detect. No consumer passes one. *)
module Mutation : sig
  type t =
    | A_row_stride  (** A's rows one element further apart than they are *)
    | No_wait  (** the staging chain started and not waited for *)
    | Tail_vl  (** the last column tile one lane short *)
end

(** Which operands to copy into the function's own DTCM panels first. *)
module Stage : sig
  type t = { a : bool; b : bool }
end

val in_sram : Contraction.t -> Stage.t
(** The staging the subsystem measurements support for operands in SRAM: A
    always, B only when its columns are strided. *)

val symbol : Contraction.t -> string
(** {!Contraction.symbol}. *)

val module_ :
  ?mutation:Mutation.t ->
  ?stage:Stage.t ->
  Contraction.t ->
  (Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_, string) result
(** The function under {!symbol}, or why it cannot be made (a seed the kernel
    does not take, an offset past 32 bits). *)
