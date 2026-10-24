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

    With [plan] (default both in place; for the subsystem, operands in SRAM, a
    plan checked for the same descriptor) the function first copies the copied
    operands into its own DTCM panels (a [.bss] of their spans) by one DMA chain
    it builds from its arguments, starts and waits for, then computes from the
    panels; an operand not 16-byte aligned or a chain ending in error leaves it
    reading the operands in place. A streamed A goes through two DTCM halves a
    block of rows each: the chain carrying block j + 1 starts once block j has
    arrived, before block j's product, into the half block j - 1 used, so the
    copy runs beside the product; the loop order is then blocks, column tiles,
    row groups. A ring source not 16-byte aligned or a chain ending in error
    executes [unimp]: the function's arguments are buffer bases, which every
    Coral layout aligns. It saves [s5]-[s11] too when it streams. *)

(** Fault injection for the evidence suite: each is one deliberate kernel defect
    a run must detect. No consumer passes one. *)
module Mutation : sig
  type t =
    | A_row_stride  (** A's rows one element further apart than they are *)
    | Aliased_ring  (** each next block fetched into the half being read *)
    | C_current_half
        (** each block's chain carries C out of the half being computed, not the
            one the block before filled *)
    | No_ring_wait  (** a block computed without waiting for its chain *)
    | No_share  (** a shared B's core part not copied *)
    | No_wait  (** the staging chain started and not waited for *)
    | Tail_vl  (** the last column tile one lane short *)
end

val symbol : Contraction.t -> string
(** {!Contraction.symbol}. *)

val module_ :
  ?mutation:Mutation.t ->
  ?plan:Coral_transfer.Transfer_plan.Witness.t ->
  Contraction.t ->
  (Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_, string) result
(** The function under {!symbol}, or why it cannot be made (a seed the kernel
    does not take, an offset past 32 bits). *)
