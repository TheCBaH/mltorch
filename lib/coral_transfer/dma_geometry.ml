(* The DMA-fed matrix multiplies' geometry as data ({!Coralnpu_kernels_zvt.Dma_gemm}
   emits them, {!Coralnpu_kernels_zvt.Dma_model} times them): which engine
   computes a block, who moves C and the resident operand, which operand stays
   in DTCM, and the problem's extents. Pure, so a planner can hold one. *)

(* What computes a block. Alphabetical. *)
module Engine = struct
  type t =
    | Rvv_bf16  (** BF16 x BF16 -> FP32, [vfwmaccbf16.vf]: three rows, W = 32 *)
    | Rvv_f32  (** FP32 [vfmacc.vf]: three rows, W = 32 *)
    | Rvv_i8  (** INT8 x INT8 -> INT16, [vwmacc.vx]: two rows, W = 64 *)
    | Zvt_bf16
        (** BF16 x BF16 -> FP32 [vtfmm.alt]: 32 rows, W a multiple of 32 *)
    | Zvt_f32  (** FP32 [vtfmm]: 32 rows, W a multiple of 32 *)
    | Zvt_i8  (** signed INT8 -> INT32 [vtmms]: 32 rows, W a multiple of 32 *)

  let name = function
    | Rvv_bf16 -> "rvv bf16"
    | Rvv_f32 -> "rvv f32"
    | Rvv_i8 -> "rvv i8"
    | Zvt_bf16 -> "zvt bf16"
    | Zvt_f32 -> "zvt f32"
    | Zvt_i8 -> "zvt i8"

  let rows = function
    | Rvv_bf16 | Rvv_f32 -> 3
    | Rvv_i8 -> 2
    | Zvt_bf16 | Zvt_f32 | Zvt_i8 -> 32

  let matrix = function
    | Rvv_bf16 | Rvv_f32 | Rvv_i8 -> false
    | Zvt_bf16 | Zvt_f32 | Zvt_i8 -> true

  let rvv = function
    | Rvv_bf16 | Rvv_f32 | Rvv_i8 -> true
    | Zvt_bf16 | Zvt_f32 | Zvt_i8 -> false

  (* The columns an RVV block computes, one vector's lanes; a Zvt block's are
     any multiple of 32. *)
  let lanes = function
    | Rvv_bf16 | Rvv_f32 -> Some 32
    | Rvv_i8 -> Some 64
    | Zvt_bf16 | Zvt_f32 | Zvt_i8 -> None
end

(* Who copies C's rows out to SRAM. Alphabetical. *)
module Out = struct
  type t =
    | Core  (** the block's own stores, straight to SRAM: no C ring *)
    | Dma  (** staged in the C ring, copied by the next step's chain *)
    | Split of int
        (** Zvt: the block's first [n] rows by its own stores, the rest by the
            DMA, through the C ring *)

  let name = function
    | Core -> "c-by-core"
    | Dma -> "c-by-dma"
    | Split n -> Printf.sprintf "c-split-%d" n

  (* The rows of a block the core stores, of [rows]. *)
  let by_core ~rows = function Core -> rows | Dma -> 0 | Split n -> n
end

(* Who copies the resident operand into DTCM in the prologue. Alphabetical. *)
module Prologue = struct
  type t =
    | Dma  (** the prologue chain, with the core waiting *)
    | Shared
        (** the chain takes a quarter, the core's RVV copy the rest beside it:
            the core copies 2.4 bytes a cycle, the DMA 0.8 while the core writes
            DTCM *)

  let name ~operand = function
    | Dma -> operand ^ "-by-dma"
    | Shared -> operand ^ "-shared"
end

(* Which operand stays in DTCM. Alphabetical. *)
module Geometry = struct
  type t =
    | Resident_a of { panel : int }
        (** A, with B streamed by panels of [panel] columns: for W too wide for
            DTCM *)
    | Resident_b  (** B, with A streamed by blocks of rows *)
    | Row_major_a
        (** as resident B, with A row-major in SRAM as an activation is: each
            block's rows arrive as they are and the core packs them in DTCM *)
    | Split_k of { chunk : int }
        (** B by chunks of [chunk] rows, for K too deep for DTCM: the next chunk
            arrives beside the current one's steps, A streams by blocks of a
            chunk's columns, and C's partial sums go out and come back *)

  let name = function
    | Resident_a { panel } -> Printf.sprintf "resident-a/%d" panel
    | Resident_b -> "resident-b"
    | Row_major_a -> "row-major-a"
    | Split_k { chunk } -> Printf.sprintf "split-k/%d" chunk
end

type t = {
  blocks : int;
  depth : int;
  engine : Engine.t;
  geometry : Geometry.t;
  out : Out.t;
  prologue : Prologue.t;
  width : int;
}
