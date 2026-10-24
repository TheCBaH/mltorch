(* The RVV blocks the DMA-fed matrix multiplies run ({!Dma_gemm}), the peak
   kernels' own ({!Peak_kernels.Gemm}) with their operands where the pipeline
   puts them, and the packing of a row-major block of A for them. *)

open Peak_kernels

(* Where a block finds its operands: A at [a1], packed a k step's rows
   together; B's rows at [b], [b_row] bytes apart, [lanes] wide; C's rows at
   [a6], [c_row] apart, which with [accumulate] hold the partial sums the
   accumulators start from. *)
module Shape = struct
  type t = {
    accumulate : bool;
    b : Gpr.t;
    b_row : int;
    c_row : int;
    depth : int;
    lanes : int;
  }
end

(* The accumulators [vs], zero or C's rows from [a6], which is left as it
   was. *)
let accumulators (s : Shape.t) ~eew vs =
  if s.accumulate then
    List.concat_map
      (fun v -> [ vle eew v Gpr.a6; addi Gpr.a6 Gpr.a6 s.c_row ])
      vs
    @ [ addi Gpr.a6 Gpr.a6 (-List.length vs * s.c_row) ]
  else List.map vmv_v_i vs

(* INT8 x INT8 -> INT16: two rows of 64 lanes. Writes [a0]-[a6], [t0],
   [t3]-[t5]. *)
let i8 (s : Shape.t) =
  let open Gpr in
  let v0 = Vreg.v0 and v4 = Vreg.v4 and v16 = Vreg.v16 and v24 = Vreg.v24 in
  let mac acc scalar src = Item.insn (Vwmacc_vx { acc; scalar; src }) in
  let rows2 = 2 * s.b_row in
  [
    li t0 s.lanes;
    mv a2 s.b;
    addi a3 a2 s.b_row;
    li a0 (s.depth / 4);
    vset_item ~avl:t0 8 4;
    vle E8 v0 a2;
    lw a4 a1 0;
    vset_item ~avl:t0 16 8;
  ]
  @ accumulators s ~eew:E16 [ v16; v24 ]
  @ [ vset_item ~avl:t0 8 4 ]
  @ body_items
      [
        lw a5 a1 4;
        vle E8 v4 a3;
        srli t3 a4 8;
        mac v16 a4 v0;
        mac v24 t3 v0;
        addi a2 a2 rows2;
        vle E8 v0 a2;
        srli t4 a4 16;
        srli t5 a4 24;
        mac v16 t4 v4;
        mac v24 t5 v4;
        addi a3 a3 rows2;
        vle E8 v4 a3;
        srli t3 a5 8;
        mac v16 a5 v0;
        mac v24 t3 v0;
        addi a2 a2 rows2;
        addi a1 a1 8;
        lw a4 a1 0;
        vle E8 v0 a2;
        srli t4 a5 16;
        srli t5 a5 24;
        mac v16 t4 v4;
        mac v24 t5 v4;
        addi a3 a3 rows2;
        addi a0 a0 (-1);
        bnez a0 body_label;
      ]
  @ [
      vset_item ~avl:t0 16 8; vse E16 v16 a6; addi a6 a6 s.c_row; vse E16 v24 a6;
    ]

(* BF16 x BF16 -> FP32: three rows of 32 lanes, A by [flh] a k step ahead, B
   rows alternating between v0 and v4. Writes [a0]-[a3], [a6], [t0],
   [fa0]-[fa5]. *)
let bf16 (s : Shape.t) =
  let open Gpr in
  let open Reg.Fpr in
  let v0 = Vreg.v0 and v4 = Vreg.v4 and v8 = Vreg.v8 in
  let v16 = Vreg.v16 and v24 = Vreg.v24 in
  let mac acc scalar src = Item.insn (Vfwmaccbf16_vf { acc; scalar; src }) in
  let rows2 = 2 * s.b_row in
  [
    li t0 s.lanes;
    mv a2 s.b;
    addi a3 a2 s.b_row;
    li a0 (s.depth / 2);
    vset_item ~avl:t0 16 4;
    vle E16 v0 a2;
    flh fa0 a1 0;
    flh fa1 a1 2;
    flh fa2 a1 4;
    vset_item ~avl:t0 32 8;
  ]
  @ accumulators s ~eew:E32 [ v8; v16; v24 ]
  @ [ vset_item ~avl:t0 16 4 ]
  @ body_items
      [
        flh fa3 a1 6;
        flh fa4 a1 8;
        flh fa5 a1 10;
        vle E16 v4 a3;
        mac v8 fa0 v0;
        mac v16 fa1 v0;
        mac v24 fa2 v0;
        addi a1 a1 12;
        flh fa0 a1 0;
        flh fa1 a1 2;
        flh fa2 a1 4;
        addi a2 a2 rows2;
        vle E16 v0 a2;
        mac v8 fa3 v4;
        mac v16 fa4 v4;
        mac v24 fa5 v4;
        addi a3 a3 rows2;
        addi a0 a0 (-1);
        bnez a0 body_label;
      ]
  @ [ vset_item ~avl:t0 32 8 ]
  @ List.concat_map
      (fun v -> [ vse E32 v a6; addi a6 a6 s.c_row ])
      [ v8; v16; v24 ]

(* FP32: three rows of 32 lanes, A by [flw] a k step ahead, B rows into v0.
   Writes [a0]-[a2], [a6], [t0], [fa4]-[fa6], [ft1]-[ft3]. *)
let f32 (s : Shape.t) =
  let open Gpr in
  let open Reg.Fpr in
  let v0 = Vreg.v0 and v8 = Vreg.v8 and v16 = Vreg.v16 and v24 = Vreg.v24 in
  let mac acc scalar src = Item.insn (Vfmacc_vf { acc; scalar; src }) in
  let loads (r0, r1, r2) off =
    [ flw r0 a1 off; flw r1 a1 (off + 4); flw r2 a1 (off + 8) ]
  in
  let macs (r0, r1, r2) = [ mac v8 r0 v0; mac v16 r1 v0; mac v24 r2 v0 ] in
  let cur = (fa4, fa5, fa6) and next = (ft1, ft2, ft3) in
  [ li t0 s.lanes; vset_item ~avl:t0 32 8 ]
  @ accumulators s ~eew:E32 [ v8; v16; v24 ]
  @ [ mv a2 s.b; li a0 (s.depth / 2) ]
  @ loads cur 0
  @ body_items
      (loads next 12
      @ [ vle E32 v0 a2 ]
      @ macs cur
      @ [ addi a2 a2 s.b_row; addi a1 a1 24 ]
      @ loads cur 0
      @ [ vle E32 v0 a2 ]
      @ macs next
      @ [ addi a2 a2 s.b_row; addi a0 a0 (-1); bnez a0 body_label ])
  @ List.concat_map
      (fun v -> [ vse E32 v a6; addi a6 a6 s.c_row ])
      [ v8; v16; v24 ]

(* A block's [rows] rows of [depth] [bytes]-byte elements, row-major from
   [src], packed into [s11] as a block reads them: each row's elements [rows]
   apart, by [vle] and a strided [vsse] at m8, VLMAX at a time. [src] is left
   at the packed block. Writes [a2], [t0]-[t3], [t5], [t6], v0. *)
let repack ~bytes ~rows ~depth ~src =
  let open Err.Syntax in
  let open Gpr in
  let eew = sew_of_bits (8 * bytes) in
  (* VLEN 128 at m8 *)
  let vl = 128 / bytes in
  if depth mod vl <> 0 then
    invalid_arg "Dma_block: row-major A takes K a multiple of a register group";
  let row = Label.named "repack_row" and trip = Label.named "repack_trip" in
  let* step = Insn.load_imm t2 (vl * rows * bytes) in
  Ok
    ([ vset_item (8 * bytes) 8; li t5 (rows * bytes) ]
    @ List.map Item.insn step
    @ [
        mv a2 s11;
        li t1 rows;
        Item.label row;
        mv t3 a2;
        li t6 (depth / vl);
        Item.label trip;
        vle eew Vreg.v0 src;
        addi src src (vl * bytes);
        Item.insn (Vsse { eew; src = Vreg.v0; base = t3; stride = t5 });
        add t3 t3 t2;
        addi t6 t6 (-1);
        bnez t6 trip;
        addi a2 a2 bytes;
        addi t1 t1 (-1);
        bnez t1 row;
        mv src s11;
      ])
