open Machine_rivet_riscv32.Rv32_asm

let slot_symbol = "coral_overlay_slot"
let ensure_label = "coral_overlay_ensure"

module Entry = struct
  type t = { name : string; entry_offset : int64 }
end

module Table = struct
  type entry = { chain : int64; bss_at : int64; bss_len : int64 }

  let entry_bytes = 16L

  let encode entries =
    let b = Buffer.create 64 in
    List.iter
      (fun e ->
        List.iter
          (fun v -> Buffer.add_int32_le b (Int64.to_int32 v))
          [ e.chain; e.bss_at; e.bss_len; 0L ])
      entries;
    Buffer.contents b
end

(* [r = v] in exactly two instructions, for any 32-bit [v], so that the module
   does not change size with the layout. *)
let addr r (v : int64) =
  let v = Int64.to_int (Int64.logand v 0xFFFF_FFFFL) in
  let lo = (v land 0xFFF lxor 0x800) - 0x800 in
  let hi = ((v - lo) asr 12) land 0xFFFFF in
  [ insn "lui" [ reg r; imm hi ]; insn "addi" [ reg r; reg r; imm lo ] ]

(* Polls of [STATUS] before a load is given up: far more than the largest
   slot's copy takes. *)
let polls = 0x100000L

(* Entered with the overlay's number in [t6] (from 1) and the return address in
   [t4]; clobbers [t0]-[t3] and [t5]. *)
let ensure ~state ~table =
  let ready = "coral_overlay_ready"
  and poll = "coral_overlay_poll"
  and loaded = "coral_overlay_loaded"
  and fail = "coral_overlay_fail"
  and zero_loop = "coral_overlay_zero"
  and zeroed = "coral_overlay_zeroed" in
  let entry =
    addr "t1" table
    @ [
        insn "addi" [ reg "t2"; reg "t6"; imm (-1) ];
        insn "slli" [ reg "t2"; reg "t2"; imm 4 ];
        insn "add" [ reg "t1"; reg "t1"; reg "t2" ];
      ]
  in
  let register r = Coral_dma.Register.offset r in
  [ label ensure_label ]
  @ addr "t0" state
  @ [
      insn "lw" [ reg "t1"; mem "t0" 0 ];
      insn "beq" [ reg "t1"; reg "t6"; sym ready ];
      (* nothing is callable while the slot is being written *)
      insn "sw" [ reg "zero"; mem "t0" 0 ];
    ]
  @ entry @ addr "t2" Coral_dma.base
  @ [
      insn "lw" [ reg "t3"; mem "t1" 0 ];
      insn "sw" [ reg "t3"; mem "t2" (register Desc_addr) ];
      insn "li" [ reg "t3"; imm Coral_dma.ctrl_start ];
      insn "sw" [ reg "t3"; mem "t2" (register Ctrl) ];
    ]
  @ addr "t5" polls
  @ [
      label poll;
      insn "lw" [ reg "t3"; mem "t2" (register Status) ];
      insn "andi" [ reg "t1"; reg "t3"; imm Coral_dma.status_done ];
      insn "bnez" [ reg "t1"; sym loaded ];
      insn "andi" [ reg "t1"; reg "t3"; imm Coral_dma.status_error ];
      insn "bnez" [ reg "t1"; sym fail ];
      insn "addi" [ reg "t5"; reg "t5"; imm (-1) ];
      insn "bnez" [ reg "t5"; sym poll ];
      label fail;
      insn "ecall" [];
      label loaded;
    ]
  @ entry
  @ [
      insn "lw" [ reg "t3"; mem "t1" 4 ];
      insn "lw" [ reg "t5"; mem "t1" 8 ];
      insn "beqz" [ reg "t5"; sym zeroed ];
      label zero_loop;
      insn "sw" [ reg "zero"; mem "t3" 0 ];
      insn "addi" [ reg "t3"; reg "t3"; imm 4 ];
      insn "addi" [ reg "t5"; reg "t5"; imm (-4) ];
      insn "bnez" [ reg "t5"; sym zero_loop ];
      label zeroed;
      insn "fence.i" [];
      insn "sw" [ reg "t6"; mem "t0" 0 ];
      label ready;
      insn "jr" [ reg "t4" ];
    ]

let stub k (e : Entry.t) =
  [
    align 4;
    global e.name;
    label e.name;
    insn "li" [ reg "t6"; imm (k + 1) ];
    insn "jal" [ reg "t4"; sym ensure_label ];
    insn "la" [ reg "t0"; sym slot_symbol ];
  ]
  @ addr "t1" e.entry_offset
  @ [ insn "add" [ reg "t0"; reg "t0"; reg "t1" ]; insn "jr" [ reg "t0" ] ]

let module_ ~slot_bytes ~state ~table entries =
  module_ "coral_overlay_manager" (fun () ->
      [ text; align 4 ]
      @ ensure ~state ~table
      @ List.concat (List.mapi stub entries)
      @ [
          align 16;
          global slot_symbol;
          label slot_symbol;
          zero (Int64.to_int slot_bytes);
        ])
