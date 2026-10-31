(* What the descriptor fixes of the fused attention function before any code
   is made: the block shapes, the buffers and their sizes. *)

module R = Machine_rivet_riscv32
module Insn = R.Rv32_kernel_insn
module A = Attention_block
module P = Peak_zvt
module K = Coralnpu_kernels.Peak_kernels

exception Refused of string

let refuse why = raise (Refused why)
let bq = 16

(* The shape of a block product, as the packed contraction plans it. *)
let plan ~rows ~depth ~n =
  P.Tile_gemm.plan Insn.Tile_op.Vtfmm_alt
    (P.Tile.shape Insn.Tile_op.Vtfmm_alt)
    {
      K.Gemm.rows;
      width = n;
      depth;
      blocks = 1;
      src = K.Elem.Bf16;
      dst = K.Elem.F32;
      sa = 1;
      sb = 1;
    }

(* Geometry derived once from the descriptor. *)
module Geometry = struct
  type t = {
    d : int;
    depth : int;  (** [d] taken to a multiple of four *)
    bk : int;
    tile_bytes : int;
    oacc_bytes : int;
    q_panel : int;
    k_panel : int;
    stage_bytes : int;
    arrays_bytes : int;
    q_blocks : int;
    q_tail : int;
    k_blocks : int;
    k_tail : int;
  }

  let make (t : A.t) =
    let narrow name x =
      if Int64.compare x 0L < 0 || Int64.compare x 0x7FFF_FFFFL > 0 then
        refuse (name ^ " past 31 bits")
      else Int64.to_int x
    in
    if t.A.precision <> A.Precision.Bf16_paired then
      refuse "a precision other than bfloat16 with paired sums";
    let d = narrow "d" t.A.d and bk = narrow "block_keys" t.A.block_keys in
    let tq = narrow "tq" t.A.tq and tk = narrow "tk" t.A.tk in
    if d > 128 then refuse "d past 128";
    if d mod 2 <> 0 then refuse "an odd d";
    if bk > 32 || (bk > 16 && bk <> 32) then
      refuse "block_keys that is neither at most 16 nor 32";
    let col (s : A.Strides.t) = Int64.equal s.A.Strides.col 1L in
    if not (col t.A.q && col t.A.k && col t.A.v && col t.A.o) then
      refuse "a feature stride other than one element";
    let depth = (d + 3) / 4 * 4 in
    let ceil_div a b = (a + b - 1) / b in
    let q_blocks = ceil_div tq bq and k_blocks = ceil_div tk bk in
    let b_plan = plan ~rows:bk ~depth ~n:bq in
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
      max ((bk * d * 2) + (8 * d) + 128) (max (bq * d * 2) (bk * d * 2))
    in
    {
      d;
      depth;
      bk;
      (* the bfloat16 P^T rows lie in the tile's first half; the context
         block reads [4 * 2 * bq + 128] bytes past them *)
      tile_bytes = max (bk * bq * 4) ((bk * bq * 2) + (8 * bq) + 128);
      oacc_bytes = bq * d * 4;
      q_panel;
      k_panel;
      stage_bytes;
      arrays_bytes = 6 * bq * 4;
      q_blocks;
      q_tail = tq - ((q_blocks - 1) * bq);
      k_blocks;
      k_tail = tk - ((k_blocks - 1) * bk);
    }

  let align64 n = (n + 63) / 64 * 64

  let total g =
    align64 g.tile_bytes + align64 g.oacc_bytes + align64 g.q_panel
    + align64 g.k_panel + align64 g.stage_bytes + align64 g.arrays_bytes
end
