open Machine_rivet_riscv32.Rv32_asm
module Abi = Machine_rivet_riscv32.Rv32_harness.Abi

let refused = 0xFFFF_FFFFl

(* The reason a guest gives for refusing its input: [mepc] of the refusal
   frame. The codes are the wire's, in declaration order from 1. *)
module Refusal = struct
  type t = Digest | End_of_input | Length | Magic | Trailer | Version

  let code = function
    | Magic -> 1
    | Version -> 2
    | Digest -> 3
    | Length -> 4
    | End_of_input -> 5
    | Trailer -> 6

  let of_code = function
    | 1l -> Some Magic
    | 2l -> Some Version
    | 3l -> Some Digest
    | 4l -> Some Length
    | 5l -> Some End_of_input
    | 6l -> Some Trailer
    | _ -> None

  let to_string = function
    | Digest -> "layout digest"
    | End_of_input -> "end of input"
    | Length -> "payload length"
    | Magic -> "magic"
    | Trailer -> "trailer"
    | Version -> "version"
end

let entry_symbol = "_start"

(* CSR numbers. *)
let fcsr = 0x003
let mstatus = 0x300
let mtvec = 0x305
let mcause = 0x342
let mepc = 0x341
let mtval = 0x343
let mcycle = 0xB00
let minstret = 0xB02
let mcycleh = 0xB80
let minstreth = 0xB82
let mpause = 0x08000073

(* [r = v] in exactly two instructions, for any 32-bit [v]. *)
let addr r (v : int64) =
  let v = Int64.to_int (Int64.logand v 0xFFFF_FFFFL) in
  let lo = (v land 0xFFF lxor 0x800) - 0x800 in
  let hi = ((v - lo) asr 12) land 0xFFFFF in
  [ insn "lui" [ reg r; imm hi ]; insn "addi" [ reg r; reg r; imm lo ] ]

let word s at = Int64.of_int32 (String.get_int32_le s at)

(* Fresh labels, numbered from 1 for each module built. *)
let next_label = ref 0

let fresh name =
  incr next_label;
  Printf.sprintf "coral_%s_%d" name !next_label

let sys_label = "coral_sys"
let trap_label = "coral_trap"
let report_label = "coral_report"
let refuse_label k = Printf.sprintf "coral_refuse_%d" (Refusal.code k)

(* [sys(a0 = number, a1, a2, a3) -> a0] through the HTIF packet: the
   simulators' proxy-kernel calls. Clobbers [t0]-[t4]. *)
let sys (l : Coral_layout.t) =
  let at r off = mem r off in
  [ align 4; label sys_label ]
  @ addr "t0" l.packet
  @ [
      insn "sw" [ reg "a0"; at "t0" 0 ];
      insn "sw" [ reg "zero"; at "t0" 4 ];
      insn "sw" [ reg "a1"; at "t0" 8 ];
      insn "srai" [ reg "t1"; reg "a1"; imm 31 ];
      insn "sw" [ reg "t1"; at "t0" 12 ];
      insn "sw" [ reg "a2"; at "t0" 16 ];
      insn "sw" [ reg "zero"; at "t0" 20 ];
      insn "sw" [ reg "a3"; at "t0" 24 ];
    ]
  @ List.map
      (fun off -> insn "sw" [ reg "zero"; at "t0" off ])
      [ 28; 32; 36; 40; 44; 48; 52; 56; 60 ]
  @ [ insn "fence" [ sym "rw"; sym "w" ] ]
  @ addr "t1" l.htif
  @ [
      insn "sw" [ reg "zero"; at "t1" 4 ];
      insn "sw" [ reg "t0"; at "t1" 0 ];
      insn "fence" [ sym "rw"; sym "w" ];
      insn "li" [ reg "t2"; imm 1 ];
      insn "sw" [ reg "zero"; at "t1" 132 ];
      insn "sw" [ reg "t2"; at "t1" 128 ];
      label "coral_sys_wait";
      insn "lw" [ reg "t3"; at "t1" 64 ];
      insn "lw" [ reg "t4"; at "t1" 68 ];
      insn "or" [ reg "t3"; reg "t3"; reg "t4" ];
      insn "lw" [ reg "t4"; at "t1" 192 ];
      insn "or" [ reg "t3"; reg "t3"; reg "t4" ];
      insn "beqz" [ reg "t3"; sym "coral_sys_wait" ];
      insn "sw" [ reg "zero"; at "t1" 64 ];
      insn "sw" [ reg "zero"; at "t1" 68 ];
      insn "sw" [ reg "zero"; at "t1" 192 ];
      insn "sw" [ reg "zero"; at "t1" 196 ];
      insn "fence" [ sym "r"; sym "rw" ];
      insn "lw" [ reg "a0"; at "t0" 0 ];
      insn "ret" [];
    ]

(* HTIF exit with [code] in [a0], then [mpause] forever. *)
let exit_with (l : Coral_layout.t) =
  let halt = fresh "halt" in
  [
    insn "slli" [ reg "a0"; reg "a0"; imm 1 ];
    insn "ori" [ reg "a0"; reg "a0"; imm 1 ];
    insn "fence" [ sym "rw"; sym "w" ];
  ]
  @ addr "t1" l.htif
  @ [
      insn "sw" [ reg "zero"; mem "t1" 4 ];
      insn "sw" [ reg "a0"; mem "t1" 0 ];
      insn "fence" [ sym "rw"; sym "w" ];
      insn "li" [ reg "t2"; imm 1 ];
      insn "sw" [ reg "zero"; mem "t1" 132 ];
      insn "sw" [ reg "t2"; mem "t1" 128 ];
      label halt;
      words [ mpause ];
      insn "j" [ sym halt ];
    ]

(* Calls [sys n] on fd [fd] until [s7] bytes from [s6] have moved. A short
   transfer continues; zero, less, or more than was asked goes to [on_end].
   Clobbers [s6], [s7]. *)
let transfer ~n ~fd ~on_end =
  let again = fresh "xfer" and fin = fresh "xfer_done" in
  [
    label again;
    insn "beqz" [ reg "s7"; sym fin ];
    insn "li" [ reg "a0"; imm n ];
    insn "li" [ reg "a1"; imm fd ];
    insn "mv" [ reg "a2"; reg "s6" ];
    insn "mv" [ reg "a3"; reg "s7" ];
    insn "call" [ sym sys_label ];
    insn "blez" [ reg "a0"; sym on_end ];
    insn "bltu" [ reg "s7"; reg "a0"; sym on_end ];
    insn "add" [ reg "s6"; reg "s6"; reg "a0" ];
    insn "sub" [ reg "s7"; reg "s7"; reg "a0" ];
    insn "j" [ sym again ];
    label fin;
  ]

let read_into ~dst ~len =
  addr "s6" dst @ addr "s7" len
  @ transfer ~n:63 ~fd:0 ~on_end:(refuse_label Refusal.End_of_input)

(* A write that cannot complete leaves nothing to report on: exit 2. *)
let write_from_regs () = transfer ~n:64 ~fd:1 ~on_end:"coral_write_failed"
let write_from ~src ~len = addr "s6" src @ addr "s7" len @ write_from_regs ()

(* [t2] bytes from [t1] to [t0]: words, then the tail bytes. All three are
   advanced; [t3], [t4] clobbered. Every caller's addresses are word-aligned. *)
let copy () =
  let words_ = fresh "copy_w"
  and bytes = fresh "copy_b"
  and fin = fresh "copy_done" in
  [
    label words_;
    insn "li" [ reg "t3"; imm 4 ];
    insn "bltu" [ reg "t2"; reg "t3"; sym bytes ];
    insn "lw" [ reg "t4"; mem "t1" 0 ];
    insn "sw" [ reg "t4"; mem "t0" 0 ];
    insn "addi" [ reg "t0"; reg "t0"; imm 4 ];
    insn "addi" [ reg "t1"; reg "t1"; imm 4 ];
    insn "addi" [ reg "t2"; reg "t2"; imm (-4) ];
    insn "j" [ sym words_ ];
    label bytes;
    insn "beqz" [ reg "t2"; sym fin ];
    insn "lbu" [ reg "t4"; mem "t1" 0 ];
    insn "sb" [ reg "t4"; mem "t0" 0 ];
    insn "addi" [ reg "t0"; reg "t0"; imm 1 ];
    insn "addi" [ reg "t1"; reg "t1"; imm 1 ];
    insn "addi" [ reg "t2"; reg "t2"; imm (-1) ];
    insn "j" [ sym bytes ];
    label fin;
  ]

(* [size] bytes at [at] set to zero, by words then bytes. *)
let zero_fill ~at ~size =
  let words_ = fresh "zero_w"
  and bytes = fresh "zero_b"
  and fin = fresh "zero_done" in
  addr "t0" at @ addr "t2" size
  @ [
      label words_;
      insn "li" [ reg "t3"; imm 4 ];
      insn "bltu" [ reg "t2"; reg "t3"; sym bytes ];
      insn "sw" [ reg "zero"; mem "t0" 0 ];
      insn "addi" [ reg "t0"; reg "t0"; imm 4 ];
      insn "addi" [ reg "t2"; reg "t2"; imm (-4) ];
      insn "j" [ sym words_ ];
      label bytes;
      insn "beqz" [ reg "t2"; sym fin ];
      insn "sb" [ reg "zero"; mem "t0" 0 ];
      insn "addi" [ reg "t0"; reg "t0"; imm 1 ];
      insn "addi" [ reg "t2"; reg "t2"; imm (-1) ];
      insn "j" [ sym bytes ];
      label fin;
    ]

(* A slot through the staging buffer, chunk by chunk: [s8] the slot cursor,
   [s9] the bytes left, [s10] this chunk. *)
let staged (l : Coral_layout.t) (s : Coral_layout.Placed_slot.t) ~reading =
  let again = fresh "stage"
  and fin = fresh "stage_done"
  and full = fresh "stage_full" in
  addr "s8" s.address @ addr "s9" s.slot.size
  @ [ label again; insn "beqz" [ reg "s9"; sym fin ] ]
  @ addr "s10" l.staging_size
  @ [
      insn "bgeu" [ reg "s9"; reg "s10"; sym full ];
      insn "mv" [ reg "s10"; reg "s9" ];
      label full;
    ]
  @ (if reading then
       addr "s6" l.staging
       @ [ insn "mv" [ reg "s7"; reg "s10" ] ]
       @ transfer ~n:63 ~fd:0 ~on_end:(refuse_label Refusal.End_of_input)
       @ [ insn "mv" [ reg "t0"; reg "s8" ] ]
       @ addr "t1" l.staging
       @ [ insn "mv" [ reg "t2"; reg "s10" ] ]
       @ copy ()
     else
       addr "t0" l.staging
       @ [ insn "mv" [ reg "t1"; reg "s8" ]; insn "mv" [ reg "t2"; reg "s10" ] ]
       @ copy () @ addr "s6" l.staging
       @ [ insn "mv" [ reg "s7"; reg "s10" ] ]
       @ write_from_regs ())
  @ [
      insn "add" [ reg "s8"; reg "s8"; reg "s10" ];
      insn "sub" [ reg "s9"; reg "s9"; reg "s10" ];
      insn "j" [ sym again ];
      label fin;
    ]

let slot_io l (s : Coral_layout.Placed_slot.t) ~reading =
  if s.htif_reachable then
    if reading then read_into ~dst:s.address ~len:s.slot.size
    else write_from ~src:s.address ~len:s.slot.size
  else staged l s ~reading

(* [t0] compared with the 32-bit [expected]; a mismatch refuses for [k]. *)
let expect_word ~at off expected k =
  [ insn "lw" [ reg "t0"; mem at off ] ]
  @ addr "t1" expected
  @ [ insn "bne" [ reg "t0"; reg "t1"; sym (refuse_label k) ] ]

(* The 64-bit counter [lo]/[hi] into [rl]/[rh], [hi] read on both sides of
   [lo] and retried on a carry between them. *)
let read_counter ~lo ~hi ~rl ~rh =
  let again = fresh "ctr" in
  [
    label again;
    insn "csrr" [ reg rh; imm hi ];
    insn "csrr" [ reg rl; imm lo ];
    insn "csrr" [ reg "t2"; imm hi ];
    insn "bne" [ reg rh; reg "t2"; sym again ];
  ]

(* [end - start] of a counter into the frame at [s6 + off]: [t0]/[t1] hold the
   end, [sl]/[sh] the start. *)
let store_delta ~sl ~sh ~off =
  [
    insn "sltu" [ reg "t2"; reg "t0"; reg sl ];
    insn "sub" [ reg "t0"; reg "t0"; reg sl ];
    insn "sub" [ reg "t1"; reg "t1"; reg sh ];
    insn "sub" [ reg "t1"; reg "t1"; reg "t2" ];
    insn "sw" [ reg "t0"; mem "s6" off ];
    insn "sw" [ reg "t1"; mem "s6" (off + 4) ];
  ]

let magic_in = "CNPUIN01"
let magic_out = "CNPUOUT1"
let magic_trap = "CNPUTRP1"
let trailer = "CNPUEND1"
let store_const ~at off v = addr "t0" v @ [ insn "sw" [ reg "t0"; mem at off ] ]

let store_string ~at off s =
  store_const ~at off (word s 0) @ store_const ~at (off + 4) (word s 4)

let size_of slots p =
  List.fold_left
    (fun n (s : Coral_layout.Placed_slot.t) ->
      if p s.slot.placement then Int64.add n s.slot.size else n)
    0L slots

let table_label = "coral_slot_table"

let items ?(vector = false) ?(matrix = false) (l : Coral_layout.t) ~abi ~entry
    () =
  next_label := 0;
  let version = Int64.of_int32 Coral_wire.version in
  let ins =
    List.filter
      (fun (s : Coral_layout.Placed_slot.t) ->
        Coral_layout.Placement.read s.slot.placement)
      l.slots
  in
  let outs =
    List.filter
      (fun (s : Coral_layout.Placed_slot.t) ->
        Coral_layout.Placement.written s.slot.placement)
      l.slots
  in
  let start =
    [ text; align 4; global entry_symbol; label entry_symbol ]
    @ addr "sp" l.stack_top
    @ [
        insn "la" [ reg "t0"; sym trap_label ];
        insn "csrw" [ imm mtvec; reg "t0" ];
      ]
    (* FS (and, for vector code, VS; for matrix code, MS) Initial: F, V and
       Zvt instructions trap while their state is Off *)
    @ addr "t0"
        (Int64.logor
           (if vector then 0x2200L else 0x2000L)
           (if matrix then 0x2000_0000L else 0L))
    @ [
        insn "csrs" [ imm mstatus; reg "t0" ]; insn "csrwi" [ imm fcsr; imm 0 ];
      ]
    @ List.concat_map
        (fun (s : Coral_layout.Placed_slot.t) ->
          match s.slot.placement with
          | Out | Scratch -> zero_fill ~at:s.address ~size:s.slot.size
          | In | In_out -> [])
        l.slots
    (* The header: magic, version, digest, length. *)
    @ read_into ~dst:l.staging ~len:20L
    @ addr "s6" l.staging
    @ expect_word ~at:"s6" 0 (word magic_in 0) Magic
    @ expect_word ~at:"s6" 4 (word magic_in 4) Magic
    @ expect_word ~at:"s6" 8 version Version
    @ expect_word ~at:"s6" 12 (Int64.of_int32 l.digest) Digest
    @ expect_word ~at:"s6" 16
        (size_of l.slots Coral_layout.Placement.read)
        Length
    @ List.concat_map (slot_io l ~reading:true) ins
    @ read_into ~dst:l.staging ~len:8L
    @ addr "s6" l.staging
    @ expect_word ~at:"s6" 0 (word trailer 0) Trailer
    @ expect_word ~at:"s6" 4 (word trailer 4) Trailer
    @ read_counter ~lo:mcycle ~hi:mcycleh ~rl:"s2" ~rh:"s3"
    @ read_counter ~lo:minstret ~hi:minstreth ~rl:"s4" ~rh:"s5"
    @ (match (abi : Abi.t) with
      | Table -> [ insn "la" [ reg "a0"; sym table_label ] ]
      | Slot0 -> (
          match l.slots with
          | s :: _ -> addr "a0" s.address
          | [] -> [ insn "li" [ reg "a0"; imm 0 ] ]))
    @ [ insn "call" [ sym entry ]; insn "mv" [ reg "s1"; reg "a0" ] ]
    (* The result frame's header, built in staging. *)
    @ addr "s6" l.staging
    @ read_counter ~lo:mcycle ~hi:mcycleh ~rl:"t0" ~rh:"t1"
    @ store_delta ~sl:"s2" ~sh:"s3" ~off:20
    @ read_counter ~lo:minstret ~hi:minstreth ~rl:"t0" ~rh:"t1"
    @ store_delta ~sl:"s4" ~sh:"s5" ~off:28
    @ store_string ~at:"s6" 0 magic_out
    @ store_const ~at:"s6" 8 version
    @ store_const ~at:"s6" 12 (Int64.of_int32 l.digest)
    @ [ insn "sw" [ reg "s1"; mem "s6" 16 ] ]
    @ store_const ~at:"s6" 36 (size_of l.slots Coral_layout.Placement.written)
    @ write_from ~src:l.staging ~len:40L
    @ List.concat_map (slot_io l ~reading:false) outs
    @ addr "s6" l.staging
    @ store_string ~at:"s6" 0 trailer
    @ write_from ~src:l.staging ~len:8L
    @ [ insn "li" [ reg "a0"; imm 0 ] ]
    @ exit_with l
  in
  let write_failed =
    [ label "coral_write_failed"; insn "li" [ reg "a0"; imm 2 ] ] @ exit_with l
  in
  let refusals =
    List.concat_map
      (fun k ->
        [ label (refuse_label k) ]
        @ addr "a4" (Int64.of_int32 refused)
        @ [
            insn "li" [ reg "a5"; imm (Refusal.code k) ];
            insn "li" [ reg "a6"; imm 0 ];
          ]
        @ [ insn "j" [ sym report_label ] ])
      Refusal.[ Magic; Version; Digest; Length; End_of_input; Trailer ]
  in
  (* [a4], [a5], [a6] as the trap frame's [mcause], [mepc], [mtval]; written
     from the trap scratch, then exit 1. Uses no stack. *)
  let report =
    [ label report_label ]
    @ addr "s6" l.trap
    @ store_string ~at:"s6" 0 magic_trap
    @ store_const ~at:"s6" 8 version
    @ [
        insn "sw" [ reg "a4"; mem "s6" 12 ];
        insn "sw" [ reg "a5"; mem "s6" 16 ];
        insn "sw" [ reg "a6"; mem "s6" 20 ];
      ]
    @ store_string ~at:"s6" 24 trailer
    @ [ insn "li" [ reg "s7"; imm 32 ] ]
    @ write_from_regs ()
    @ [ insn "li" [ reg "a0"; imm 1 ] ]
    @ exit_with l
  in
  let trap =
    [
      align 4;
      label trap_label;
      insn "csrr" [ reg "a4"; imm mcause ];
      insn "csrr" [ reg "a5"; imm mepc ];
      insn "csrr" [ reg "a6"; imm mtval ];
      insn "j" [ sym report_label ];
    ]
  in
  let table =
    [ data; align 4; label table_label ]
    @
    match l.slots with
    | [] -> [ zero 4 ]
    | slots ->
        [
          words
            (List.map
               (fun (s : Coral_layout.Placed_slot.t) -> Int64.to_int s.address)
               slots);
        ]
  in
  start @ write_failed @ refusals @ report @ trap @ sys l @ table

let module_ ?vector ?matrix l ~abi ~entry =
  module_ "coral_runtime" (items ?vector ?matrix l ~abi ~entry)
