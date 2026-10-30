(** Streaming scaled dot-product attention on the matrix engine as one function
    per {!Attention_block.t}: the keys in blocks of [block_keys], the scores
    block and the context block on the tile engine, the online softmax on the
    vector unit, S and P never held whole. It computes {!Attention_block.eval}
    bit for bit.

    The convention is {!Zvt_contraction}'s: [a0]-[a4] the addresses of Q, K, V,
    the output and the mask (ignored when the descriptor has none), [0] in [a0]
    on return, the descriptor's symbol. Q is taken [16] queries at a time; its
    panels, the tile, the accumulator and the staging buffer are the function's
    own [.bss] ({!panel_bytes}). Q, K, V and the output must have feature stride
    one, [d] be even and at most 96, and [block_keys] be at most 16 or 32. The
    blocks read [8 d + 128] bytes past the end of the staging copy of V, which
    the function leaves readable; no operand is read in place. *)

(** Phases a cycle measurement may leave out (the result is then wrong).
    Alphabetical. *)
module Phase : sig
  type t = Context | Exp | Normalize | Scores
end

(** Fault injection for the evidence suite: each is one deliberate defect a run
    must detect. No consumer passes one. Alphabetical. *)
module Mutation : sig
  type t =
    | Max_stale  (** the exponentials taken against the previous maximum *)
    | Rescale_omitted
        (** the accumulator not scaled by the new maximum's factor *)
    | Seed_ignored
        (** the context block starting from zero, not the accumulator *)
    | V_in_place_in_dtcm
        (** V read in place when it lies in the DTCM (needs [dtcm]) *)
    | V_not_advanced  (** V's pointer not moved on between key blocks *)
    | V_tail_unzeroed  (** V's rows past the valid keys not zeroed *)
end

val panel_bytes : Attention_block.t -> (int, string) result
(** The [.bss] the function declares, or why the descriptor is outside what it
    computes. *)

val block_queries : Attention_block.t -> (int, string) result
(** The queries a block holds (16), or why the descriptor is refused. *)

val module_ :
  ?mutation:Mutation.t ->
  ?budget:int ->
  ?dtcm:int * int ->
  ?skip:Phase.t list ->
  Attention_block.t ->
  (Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_, string) result
(** [budget] bounds {!panel_bytes}; [dtcm] is the DTCM's base and size, for the
    mutation that reads V in place there. *)
