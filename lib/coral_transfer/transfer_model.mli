(** What a generated contraction function takes on the subsystem, cycles, under
    a transfer plan. Each k step of a row group costs the longer of its vector
    work (per row, per four lanes) and its loads (A's values, B's row: by beats
    when contiguous, by elements when gathered, each rate by where the operand
    is), which overlap, plus its own scalar instructions; each group its own
    instructions and C's stores by beats; copies by the DMA alone before the
    product; a streamed A's later blocks beside the product, costing what the
    block before does not hide and, when the product reads B from SRAM, a
    contention per byte. The rates are fitted to the subsystem's measurements,
    so the model ranks the plans {!Transfer_plan.candidates} offers; {!choose}
    takes the cheapest. *)

module Rates : sig
  type t = {
    vector_fused : float;  (** a row's [vfmacc.vf] at m8, per four lanes *)
    vector_ordered : float;  (** a row's [vfmul.vf] and [vfadd.vv] *)
    vector_row_fused : float;
        (** a row's vector instruction's own, whatever its lanes *)
    vector_row_ordered : float;
    step : float;  (** a k step's own scalar instructions *)
    a_dtcm : float;  (** a row's A value loaded from DTCM *)
    a_sram : float;  (** the same from SRAM *)
    a_bf16 : float;  (** a bfloat16 A value's widening *)
    b_beat_dtcm : float;  (** a beat of a contiguous B row from DTCM *)
    b_row_sram : float;  (** a contiguous B row's access to SRAM *)
    b_beat_sram : float;  (** a beat of it *)
    b_element_dtcm : float;  (** a strided B element gathered from DTCM *)
    b_element_sram : float;  (** the same from SRAM *)
    group : float;  (** a row group's own: accumulators, pointers *)
    c_beat : float;  (** a beat of a row of C stored to SRAM *)
    dma : float;  (** the DMA alone, bytes a cycle *)
    dma_beside : float;  (** the DMA beside the product, bytes a cycle *)
    dma_start : float;  (** a chain's start and wait, cycles *)
    contention : float;
        (** a byte a streamed block's chain moves beside a product that reads B
            from SRAM *)
    core_copy : float;  (** the core's RVV copy from SRAM to DTCM, B/cycle *)
    dma_beside_copy : float;  (** the DMA beside that copy, B/cycle *)
    c_beat_dtcm : float;  (** a beat of a row of C stored to the DTCM ring *)
  }

  val fitted : t
end

val predict : ?rates:Rates.t -> Contraction.t -> Transfer_plan.t -> float

val choose : ?rates:Rates.t -> budget:int -> Contraction.t -> Transfer_plan.t
(** The candidate with the least {!predict}ed cycles; the first of equals. *)
