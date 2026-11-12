(** Why a Coral kernel generator declines a descriptor: the reasons the RVV
    function and the Zvt matrix-engine functions share. A generator exits its
    walk with one through {!refuse}, and the caller sees it as the [Error] of
    the generator's result. *)

type t =
  | Assembler of string  (** the assembler refused a form: its text *)
  | Bf16_ordered
      (** a bfloat16 ordered sum: the engine adds two k steps first *)
  | Block_keys_unsupported  (** a [block_keys] neither at most 16 nor 32 *)
  | C_columns_not_contiguous  (** C's columns are not one element apart *)
  | C_strides_not_one  (** neither of C's strides is one element *)
  | Constant  (** a constant the encoder cannot load *)
  | D_odd  (** an odd head dimension *)
  | D_past_128  (** a head dimension past 128 *)
  | Empty_k  (** a contraction of no k steps *)
  | Encoder of string  (** the encoder or lowering failed: its text *)
  | Extent_past_2_16  (** an extent past 2^16 *)
  | Feature_stride_not_one  (** a feature stride other than one element *)
  | Fused_requirement
      (** the fused requirement: the engine rounds each product *)
  | Int_rows_too_far_apart  (** integer A rows too far apart for one pointer *)
  | Invariant of string  (** a check inside the generator failed: its text *)
  | No_rounding_registers  (** no registers left for the rounding *)
  | Not_paired_bf16  (** not a paired bfloat16 contraction *)
  | Offset_past_31_bits  (** a byte offset past 31 bits *)
  | Operand_past_dma_transfer  (** an operand past one DMA transfer *)
  | Paired_requirement
      (** the paired requirement: only the matrix engine rounds pairs *)
  | Panels_over_budget of { needed : int; available : int }
      (** the panels need more bytes than the budget leaves *)
  | Past_31_bits of string  (** the named quantity past 31 bits *)
  | Precision_not_paired_bf16
      (** a precision other than bfloat16 with paired sums *)
  | Register_or_immediate  (** a register or immediate the encoder rejects *)
  | Ring_block_past_dma_transfer  (** a ring block past one DMA transfer *)
  | Seeded_from_output  (** a contraction seeded from its output *)
  | Seeded_k_padding  (** a seeded k that the depth padding changes *)
  | Stride_past_30_bits  (** a stride past 30 bits *)
  | Too_many_output_tiles  (** more than 256 output tiles *)
  | Transfer_plan_mismatch
      (** a transfer plan checked for another contraction *)
  | Vector_expression of string
      (** the vector-expression emitter's own reason *)

val pp : Format.formatter -> t -> unit
val to_string : t -> string

type escape = t Err.Escape.t
(** The exit of one generator call. *)

val refuse : escape -> t -> 'a
(** Exits the generator with the reason. *)

val generate : (escape -> 'a) -> ('a, t) result
(** Runs a generator body under its own exit: the reason it refuses with is the
    [Error]. The wrapper is exported here, so the detection origin is recorded.
*)

val generate_exn : (escape -> 'a) -> 'a
(** {!generate} for a caller that holds a descriptor the generator accepts: a
    refusal raises [Err.Exn.E]. *)

val generate_checked : (escape -> 'a) -> ('a, t) result
(** {!generate}, also taking an [Invalid_argument] raised by the generator's own
    checks as an [Invariant] refusal: the entry points that report why a
    descriptor cannot be made, never raise. *)
