(* What the descriptor fixes of the fused attention function before any code
   is made: the block shapes, the buffers and their sizes. *)

module R = Machine_rivet_riscv32
module Insn = R.Rv32_kernel_insn
module A = Attention_block
module P = Peak_zvt
module K = Coralnpu_kernels.Peak_kernels
module Kernel_refusal = Coralnpu_kernels.Kernel_refusal
module Bytes = Coral_transfer.Transfer_units.Bytes

let bq = Core.Dim.extent 16

(* The shape of a block product, as the packed contraction plans it. *)
let plan ~(rows : Core.Dim.extent Core.Dim.t)
    ~(depth : Core.Dim.extent Core.Dim.t) ~(n : Core.Dim.extent Core.Dim.t) =
  P.Tile_gemm.plan Insn.Tile_op.Vtfmm_alt
    (P.Tile.shape Insn.Tile_op.Vtfmm_alt)
    {
      K.Gemm.rows :> int;
      width = (n :> int);
      depth :> int;
      blocks = 1;
      src = K.Elem.Bf16;
      dst = K.Elem.F32;
      sa = 1;
      sb = 1;
    }

(* Geometry derived once from the descriptor. *)
module Geometry = struct
  type t = {
    d : Core.Dim.extent Core.Dim.t;
    depth : Core.Dim.extent Core.Dim.t;  (** [d] taken to a multiple of four *)
    bk : Core.Dim.extent Core.Dim.t;
    tile_bytes : Bytes.t;
    oacc_bytes : Bytes.t;
    q_panel : Bytes.t;
    k_panel : Bytes.t;
    stage_bytes : Bytes.t;
    arrays_bytes : Bytes.t;
    q_blocks : Core.Dim.extent Core.Dim.t;
    q_tail : Core.Dim.extent Core.Dim.t;
    k_blocks : Core.Dim.extent Core.Dim.t;
    k_tail : Core.Dim.extent Core.Dim.t;
  }

  let make esc (t : A.t) =
    let narrow name x =
      if Int64.compare x 0L < 0 || Int64.compare x 0x7FFF_FFFFL > 0 then
        Kernel_refusal.refuse esc (Past_31_bits name)
      else Int64.to_int x
    in
    if t.A.precision <> A.Precision.Bf16_paired then
      Kernel_refusal.refuse esc Precision_not_paired_bf16;
    let d = narrow "d" t.A.d and bk = narrow "block_keys" t.A.block_keys in
    let tq = narrow "tq" t.A.tq and tk = narrow "tk" t.A.tk in
    if d > 128 then Kernel_refusal.refuse esc D_past_128;
    if d mod 2 <> 0 then Kernel_refusal.refuse esc D_odd;
    if bk > 32 || (bk > 16 && bk <> 32) then
      Kernel_refusal.refuse esc Block_keys_unsupported;
    let col (s : A.Strides.t) = Int64.equal s.A.Strides.col 1L in
    if not (col t.A.q && col t.A.k && col t.A.v && col t.A.o) then
      Kernel_refusal.refuse esc Feature_stride_not_one;
    let depth = (d + 3) / 4 * 4 in
    let bq_n = (bq :> int) in
    let ceil_div a b = (a + b - 1) / b in
    let q_blocks = ceil_div tq bq_n and k_blocks = ceil_div tk bk in
    let b_plan =
      plan ~rows:(Core.Dim.extent bk) ~depth:(Core.Dim.extent depth) ~n:bq
    in
    let q_panel =
      List.fold_left
        (fun acc cols -> acc + (depth * List.length cols * 32))
        128
        (P.Tile_gemm.passes b_plan)
    in
    let k_panel = ((bk + 15) / 16 * 32 * depth) + 128 in
    (* V is read in place by the context blocks: [4 * 2 * d + 128] past its
       end *)
    let stage_bytes =
      max ((bk * d * 2) + (8 * d) + 128) (max (bq_n * d * 2) (bk * d * 2))
    in
    {
      d = Core.Dim.extent d;
      depth = Core.Dim.extent depth;
      bk = Core.Dim.extent bk;
      (* the bfloat16 P^T rows lie in the tile's first half; the context
         block reads [4 * 2 * bq + 128] bytes past them *)
      tile_bytes =
        Bytes.of_int (max (bk * bq_n * 4) ((bk * bq_n * 2) + (8 * bq_n) + 128));
      oacc_bytes = Bytes.of_int (bq_n * d * 4);
      q_panel = Bytes.of_int q_panel;
      k_panel = Bytes.of_int k_panel;
      stage_bytes = Bytes.of_int stage_bytes;
      arrays_bytes = Bytes.of_int (6 * bq_n * 4);
      q_blocks = Core.Dim.extent q_blocks;
      q_tail = Core.Dim.extent (tq - ((q_blocks - 1) * bq_n));
      k_blocks = Core.Dim.extent k_blocks;
      k_tail = Core.Dim.extent (tk - ((k_blocks - 1) * bk));
    }

  let align64 b = Bytes.round_up ~to_:(Bytes.of_int 64) b

  let total g =
    List.fold_left
      (fun acc b -> Bytes.add acc (align64 b))
      Bytes.zero
      [
        g.tile_bytes;
        g.oacc_bytes;
        g.q_panel;
        g.k_panel;
        g.stage_bytes;
        g.arrays_bytes;
      ]
end
