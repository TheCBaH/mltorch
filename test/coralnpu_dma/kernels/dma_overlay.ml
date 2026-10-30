(* Code streamed by the DMA into the instruction memory, then run: does a slot
   in the ITCM take new code, and how long does a load take?

   A resident loader owns a slot of {!slot_bytes} zero bytes in its own text
   (a zero word is an illegal instruction, so a beat that did not land traps
   rather than computes). Four loads, A, B, C and A again, each: start a chain
   that copies an overlay image from SRAM into the slot, wait for [STATUS.done],
   [fence.i], call the slot, and record the overlay's [a0], the final [STATUS]
   and the cycles from the start to the fence's end. The overlays are linked on
   their own at the slot's address ({!images}):

   - A returns 42;
   - B calls its own body through an address {i la} makes, so a wrong link
     address jumps elsewhere, and sums 11 + 10 + ... + 1 = 66;
   - C fills the whole slot with increments of [a0] and returns, so every beat
     of the slot must have landed: 511.

   A again loads a shorter image over the longer one. The table holds 0 [chains]
   (SRAM), 1 [images] (SRAM) and 2 [results] (DTCM, three words a load).

   [fence.i] is in the loader because the ISA requires it, but this core runs
   without it: the RTL gives the same results and cycles four fewer a load, even
   when the wait loop sits right before the slot, so no run here can show its
   omission. *)

open Peak_kernels
module D = Coral_dma

let slot_bytes = 2048
let results_slot = 2

(* The overlay each load brings in, in order. *)
let loads = [ "A"; "B"; "C"; "A" ]
let n = List.length loads
let expected_a0 = [ 42; 66; (slot_bytes / 4) - 1; 42 ]

let slots =
  L.Placement.
    [
      L.Slot.make ~home:Ram ~align:32L In (Int64.of_int (D.Descriptor.size * n));
      L.Slot.make ~home:Ram ~align:16L In (Int64.of_int (16 + 48 + slot_bytes));
      L.Slot.make ~home:Dtcm ~align:16L Out (Int64.of_int (12 * n));
    ]

let andi dst src n = Item.insn (Insn.Andi { dst; src; imm = imm12 n })
let beqz src target = Item.insn (Insn.Beqz { src; target })
let fence_i = Item.insn Insn.Fence_i
let jalr base = Item.insn (Insn.Jalr { base })
let la dst target = Item.insn (Insn.La { dst; target })
let mcycle dst = Item.insn (Insn.Csrr { dst; csr = Mcycle })
let sub dst lhs rhs = Item.insn (Insn.Sub { dst; lhs; rhs })
let slot_label = Label.named "ovl_slot"

(* [a0] the table; [s0] the chain cursor, [s1] the results cursor, [s2] the
   loads left. [ra] and the saved registers go to the stack, as the call into
   the slot overwrites [ra]. *)
let items () =
  let next = Label.named "next" and poll = Label.named "poll" in
  let open Err.Syntax in
  let* base = Insn.load_imm Gpr.a4 (Int64.to_int D.base) in
  Ok
    (start_items
    @ [
        addi Gpr.sp Gpr.sp (-16);
        sw Gpr.ra Gpr.sp 12;
        sw Gpr.s0 Gpr.sp 8;
        sw Gpr.s1 Gpr.sp 4;
        sw Gpr.s2 Gpr.sp 0;
        lw Gpr.s0 Gpr.a0 0;
        lw Gpr.s1 Gpr.a0 (4 * results_slot);
        li Gpr.s2 n;
      ]
    @ List.map Item.insn base
    @ [
        Item.label next;
        mcycle Gpr.t3;
        sw Gpr.s0 Gpr.a4 (D.Register.offset Desc_addr);
        li Gpr.a5 D.ctrl_start;
        sw Gpr.a5 Gpr.a4 (D.Register.offset Ctrl);
        Item.label poll;
        lw Gpr.t2 Gpr.a4 (D.Register.offset Status);
        andi Gpr.t0 Gpr.t2 D.status_done;
        beqz Gpr.t0 poll;
        fence_i;
        mcycle Gpr.t4;
        sub Gpr.t4 Gpr.t4 Gpr.t3;
        sw Gpr.t2 Gpr.s1 4;
        sw Gpr.t4 Gpr.s1 8;
        li Gpr.a0 0;
        la Gpr.t0 slot_label;
        jalr Gpr.t0;
        sw Gpr.a0 Gpr.s1 0;
        addi Gpr.s1 Gpr.s1 12;
        addi Gpr.s0 Gpr.s0 D.Descriptor.size;
        addi Gpr.s2 Gpr.s2 (-1);
        bnez Gpr.s2 next;
        lw Gpr.ra Gpr.sp 12;
        lw Gpr.s0 Gpr.sp 8;
        lw Gpr.s1 Gpr.sp 4;
        lw Gpr.s2 Gpr.sp 0;
        addi Gpr.sp Gpr.sp 16;
        li Gpr.a0 0;
        Item.insn Insn.Ret;
        Item.align 16;
        Item.export ~name:"ovl_slot" slot_label;
        Item.label slot_label;
        Item.zero slot_bytes;
      ])

(* An overlay: a self-contained module whose entry is its first item. *)
let overlay name body =
  let entry = Label.named name in
  ( name,
    [ Item.text; Item.align 4; Item.export ~name entry; Item.label entry ]
    @ body )

let overlay_a = overlay "ovl_a" [ li Gpr.a0 42; Item.insn Insn.Ret ]

let overlay_b =
  let body = Label.named "body" and loop = Label.named "loop" in
  overlay "ovl_b"
    [
      mv Gpr.t6 Gpr.ra;
      la Gpr.t0 body;
      jalr Gpr.t0;
      mv Gpr.ra Gpr.t6;
      Item.insn Insn.Ret;
      Item.label body;
      li Gpr.a0 0;
      li Gpr.t1 11;
      Item.label loop;
      add Gpr.a0 Gpr.a0 Gpr.t1;
      addi Gpr.t1 Gpr.t1 (-1);
      bnez Gpr.t1 loop;
      Item.insn Insn.Ret;
    ]

let overlay_c =
  overlay "ovl_c"
    (List.init ((slot_bytes / 4) - 1) (fun _ -> addi Gpr.a0 Gpr.a0 1)
    @ [ Item.insn Insn.Ret ])

(* The overlays, in the order of the image slot: A, B, C. *)
let overlays = [ overlay_a; overlay_b; overlay_c ]

(* D counts its calls in a bss word: 1 on a fresh load, then one more each call
   it stays resident, so a load that left the region dirty shows. *)
let overlay_d =
  let counter = Label.named "counter" in
  overlay "ovl_d"
    [
      la Gpr.t0 counter;
      lw Gpr.a0 Gpr.t0 0;
      addi Gpr.a0 Gpr.a0 1;
      sw Gpr.a0 Gpr.t0 0;
      Item.insn Insn.Ret;
      Item.bss;
      Item.align 16;
      Item.label counter;
      Item.zero 16;
    ]

(* {1 Through the manager}

   The same overlays reached as a code generator's output would: a caller that
   only [jal]s their names, with no knowledge of the slot, the DMA or the
   manager ({!Coral_overlay_manager}) that the image builder adds. Each call is
   timed whole; the table holds 0 [results] (DTCM, two words a call: the
   overlay's [a0] and the cycles). The last call finds its overlay resident. E
   calls back into the resident image: [calls_add7], in the caller's module,
   adds 7 to the 50 it is given. *)

module Calls = struct
  let overlays = overlays @ [ overlay_d ]

  let names =
    [
      "ovl_a";
      "ovl_b";
      "ovl_c";
      "ovl_a";
      "ovl_a";
      "ovl_d";
      "ovl_d";
      "ovl_a";
      "ovl_d";
      "ovl_e";
    ]

  let expected_a0 = [ 42; 66; (slot_bytes / 4) - 1; 42; 42; 1; 2; 42; 1; 57 ]
  let results_slot = 0

  let slots =
    L.Placement.
      [
        L.Slot.make ~home:Dtcm ~align:16L Out
          (Int64.of_int (8 * List.length names));
      ]

  let module_ () =
    let open Machine_rivet_riscv32.Rv32_asm in
    module_ "calls" (fun () ->
        [
          text;
          align 4;
          global entry;
          label entry;
          insn "addi" [ reg "sp"; reg "sp"; imm (-16) ];
          insn "sw" [ reg "ra"; mem "sp" 12 ];
          insn "sw" [ reg "s0"; mem "sp" 8 ];
          insn "sw" [ reg "s1"; mem "sp" 4 ];
          insn "lw" [ reg "s0"; mem "a0" (4 * results_slot) ];
        ]
        @ List.concat_map
            (fun name ->
              [
                insn "li" [ reg "a0"; imm 0 ];
                insn "csrr" [ reg "s1"; imm 0xB00 ];
                insn "jal" [ reg "ra"; sym name ];
                insn "csrr" [ reg "t3"; imm 0xB00 ];
                insn "sub" [ reg "t3"; reg "t3"; reg "s1" ];
                insn "sw" [ reg "a0"; mem "s0" 0 ];
                insn "sw" [ reg "t3"; mem "s0" 4 ];
                insn "addi" [ reg "s0"; reg "s0"; imm 8 ];
              ])
            names
        @ [
            insn "lw" [ reg "ra"; mem "sp" 12 ];
            insn "lw" [ reg "s0"; mem "sp" 8 ];
            insn "lw" [ reg "s1"; mem "sp" 4 ];
            insn "addi" [ reg "sp"; reg "sp"; imm 16 ];
            insn "li" [ reg "a0"; imm 0 ];
            insn "ret" [];
            global "calls_add7";
            label "calls_add7";
            insn "addi" [ reg "a0"; reg "a0"; imm 7 ];
            insn "ret" [];
          ])

  let overlay_e () =
    let open Machine_rivet_riscv32.Rv32_asm in
    module_ "ovl_e" (fun () ->
        [
          text;
          align 4;
          global "ovl_e";
          label "ovl_e";
          insn "addi" [ reg "sp"; reg "sp"; imm (-16) ];
          insn "sw" [ reg "ra"; mem "sp" 12 ];
          insn "li" [ reg "a0"; imm 50 ];
          insn "jal" [ reg "ra"; sym "calls_add7" ];
          insn "lw" [ reg "ra"; mem "sp" 12 ];
          insn "addi" [ reg "sp"; reg "sp"; imm 16 ];
          insn "ret" [];
        ])
end
