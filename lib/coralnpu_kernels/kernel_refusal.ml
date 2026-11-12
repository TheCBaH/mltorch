type t =
  | Assembler of string
  | Bf16_ordered
  | Block_keys_unsupported
  | C_columns_not_contiguous
  | C_strides_not_one
  | Constant
  | D_odd
  | D_past_128
  | Empty_k
  | Encoder of string
  | Extent_past_2_16
  | Feature_stride_not_one
  | Fused_requirement
  | Int_rows_too_far_apart
  | Invariant of string
  | No_rounding_registers
  | Not_paired_bf16
  | Offset_past_31_bits
  | Operand_past_dma_transfer
  | Paired_requirement
  | Panels_over_budget of { needed : int; available : int }
  | Past_31_bits of string
  | Precision_not_paired_bf16
  | Register_or_immediate
  | Ring_block_past_dma_transfer
  | Seeded_from_output
  | Seeded_k_padding
  | Stride_past_30_bits
  | Too_many_output_tiles
  | Transfer_plan_mismatch
  | Vector_expression of string

let pp fmt = function
  | Assembler what -> Fmt.string fmt what
  | Bf16_ordered ->
      Fmt.string fmt "bfloat16 ordered: the engine sums two k steps first"
  | Block_keys_unsupported ->
      Fmt.string fmt "block_keys that is neither at most 16 nor 32"
  | C_columns_not_contiguous -> Fmt.string fmt "C's columns not contiguous"
  | C_strides_not_one -> Fmt.string fmt "neither of C's strides is one element"
  | Constant -> Fmt.string fmt "a constant"
  | D_odd -> Fmt.string fmt "an odd d"
  | D_past_128 -> Fmt.string fmt "d past 128"
  | Empty_k -> Fmt.string fmt "an empty k"
  | Encoder what -> Fmt.string fmt what
  | Extent_past_2_16 -> Fmt.string fmt "an extent past 2^16"
  | Feature_stride_not_one ->
      Fmt.string fmt "a feature stride other than one element"
  | Fused_requirement ->
      Fmt.string fmt "the fused requirement: the engine rounds each product"
  | Int_rows_too_far_apart ->
      Fmt.string fmt "integer A rows too far apart for one pointer"
  | Invariant why -> Fmt.string fmt why
  | No_rounding_registers -> Fmt.string fmt "no registers left for the rounding"
  | Not_paired_bf16 -> Fmt.string fmt "not a paired bfloat16 contraction"
  | Offset_past_31_bits -> Fmt.string fmt "an offset past 31 bits"
  | Operand_past_dma_transfer ->
      Fmt.string fmt "an operand past one DMA transfer"
  | Paired_requirement ->
      Fmt.string fmt
        "the paired requirement: only the matrix engine rounds pairs"
  | Panels_over_budget { needed; available } ->
      Fmt.pf fmt "the panels need %d bytes, %d available" needed available
  | Past_31_bits what -> Fmt.pf fmt "%s past 31 bits" what
  | Precision_not_paired_bf16 ->
      Fmt.string fmt "a precision other than bfloat16 with paired sums"
  | Register_or_immediate -> Fmt.string fmt "a register or immediate"
  | Ring_block_past_dma_transfer ->
      Fmt.string fmt "a ring block past one DMA transfer"
  | Seeded_from_output -> Fmt.string fmt "a contraction seeded from its output"
  | Seeded_k_padding ->
      Fmt.string fmt "a seeded k that the depth padding changes"
  | Stride_past_30_bits -> Fmt.string fmt "a stride past 30 bits"
  | Too_many_output_tiles -> Fmt.string fmt "more than 256 output tiles"
  | Transfer_plan_mismatch ->
      Fmt.string fmt "a transfer plan checked for another contraction"
  | Vector_expression why -> Fmt.string fmt why

let to_string r = Fmt.str "%a" pp r

type escape = t Err.Escape.t

let refuse esc r = Err.Escape.throw esc ~pp_error:pp r

let generate body =
  Err.export ~pos:__POS__ ~pp_error:pp (Err.Escape.with_escape body)

let generate_exn body = Err.or_raise ~pp_error:pp (Err.Escape.with_escape body)

let generate_checked body =
  generate (fun esc ->
      try body esc with Invalid_argument why -> refuse esc (Invariant why))
