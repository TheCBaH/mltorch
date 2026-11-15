(* The process around a table-bound image: [_start] maps two files named on the
   command line and calls the image's entry with a table of region addresses.

   argv[1] is the input file, mapped read-write and private (the kernel may
   scribble on it; the file is untouched). argv[2] is the output file, mapped
   shared, which the host sized beforehand: its first {!status_bytes} bytes
   hold the entry's return value, then come the output regions. A region
   placed in [Bss] is zero-filled memory of the ELF itself. Nothing is read
   from or written to standard output.

   The entry is the System V RV32 ABI: the table in [a0], the status in [a0].
   [_start] holds the file bases in [s4]/[s5], which the entry preserves. *)

open Rv32_asm
module U = Core.Storage_units

let ( let* ) = Result.bind
let sure r = Err.or_raise ~pp_error:U.pp_error r
let checked r = Result.map_error (Fmt.str "%a" U.pp_error) (Err.payload r)

module Placement = struct
  type t = Bss | In | Out
end

(* What the entry is passed in [a0]: the table of region addresses, or the first
   region's address alone (a C [entry(void *io)]). *)
module Abi = struct
  type t = Slot0 | Table
end

type slot = {
  size : U.Byte_size.t;
  align : U.Byte_alignment.t;
  placement : Placement.t;
}

type layout = {
  slots : (slot * U.Byte_offset.t) list;
      (** with its offset in the file or block *)
  in_size : U.Byte_size.t;
  out_size : U.Byte_size.t;
  bss_size : U.Byte_size.t;
}

let status_bytes = sure (U.Byte_size.of_int64 16L)

(* Every region starts on at least a 16-byte boundary. *)
let line = sure (U.Byte_alignment.of_int64 16L)

(* Failure exits of [_start], distinct from any status a test kernel returns by
   being above it. Under the vector profile, 105 is a core whose vector
   registers are narrower than the 128 bits the code assumes. *)
module Exit = struct
  let arguments = 100
  let open_in = 101
  let map_in = 102
  let open_out = 103
  let map_out = 104
  let vlen = 105
end

let layout slots =
  let at = Hashtbl.create 3 in
  Hashtbl.replace at Placement.Out (U.Byte_offset.of_size status_bytes);
  let cursor p =
    Option.value (Hashtbl.find_opt at p) ~default:U.Byte_offset.zero
  in
  let* placed =
    List.fold_left
      (fun acc s ->
        let* placed = acc in
        let* off =
          checked
            (U.Byte_offset.align_up (cursor s.placement)
               (U.Byte_alignment.max line s.align))
        in
        let* next = checked (U.Byte_offset.advance off s.size) in
        Hashtbl.replace at s.placement next;
        Ok ((s, off) :: placed))
      (Ok []) slots
  in
  let size p =
    let* e = checked (U.Byte_offset.align_up (cursor p) line) in
    Ok (U.Byte_offset.to_size e)
  in
  let* in_size = size Placement.In in
  let* out_size = size Placement.Out in
  let* bss_size = size Placement.Bss in
  Ok { slots = List.rev placed; in_size; out_size; bss_size }

let table = "harness_table"
let bss_block = "harness_bss"
let syscall n = [ insn "li" [ reg "a7"; imm n ]; insn "ecall" [] ]

(* Opens the path in [path] and maps [len] bytes of it, leaving the base in
   [into]. Any failure exits with its own code. *)
let map ~path ~(len : U.Byte_size.t) ~flags ~prot ~into ~open_code ~map_code
    ~name =
  let bad = name ^ "_open_failed" and bad_map = name ^ "_map_failed" in
  let ok = name ^ "_ok" in
  [
    insn "li" [ reg "a0"; imm (-100) ];
    insn "mv" [ reg "a1"; reg path ];
    insn "li" [ reg "a2"; imm 2 ];
    insn "li" [ reg "a3"; imm 0 ];
  ]
  @ syscall 56
  @ [
      insn "bltz" [ reg "a0"; sym bad ];
      insn "mv" [ reg "a4"; reg "a0" ];
      insn "li" [ reg "a0"; imm 0 ];
    ]
  @ load_imm "a1" (Int64.to_int (U.Byte_size.to_int64 len))
  @ [
      insn "li" [ reg "a2"; imm prot ];
      insn "li" [ reg "a3"; imm flags ];
      insn "li" [ reg "a5"; imm 0 ];
    ]
  @ syscall 222
  @ [
      (* an error is a value in [-4095, -1] *)
      insn "li" [ reg "t0"; imm (-4096) ];
      insn "bgeu" [ reg "a0"; reg "t0"; sym bad_map ];
      insn "mv" [ reg into; reg "a0" ];
      insn "j" [ sym ok ];
      label bad;
      insn "li" [ reg "a0"; imm open_code ];
      insn "j" [ sym "harness_exit" ];
      label bad_map;
      insn "li" [ reg "a0"; imm map_code ];
      insn "j" [ sym "harness_exit" ];
      label ok;
    ]

(* [t1 = base + off], for any [off]: an [addi] immediate is 12 bits. *)
let at base (off : U.Byte_offset.t) =
  load_imm "t1" (Int64.to_int (U.Byte_offset.to_int64 off))
  @ [ insn "add" [ reg "t1"; reg base; reg "t1" ] ]

let harness ?(abi = Abi.Table) ?(min_vlenb = U.Byte_size.zero) ~entry lay =
  let has p = List.exists (fun (s, _) -> s.placement = p) lay.slots in
  let items () =
    let start =
      [
        text;
        align 4;
        global "_start";
        label "_start";
        (* argc, argv[0], argv[1], argv[2] *)
        insn "lw" [ reg "t0"; mem "sp" 0 ];
        insn "li" [ reg "t1"; imm 3 ];
        insn "li" [ reg "a0"; imm Exit.arguments ];
        insn "blt" [ reg "t0"; reg "t1"; sym "harness_exit" ];
        insn "lw" [ reg "s2"; mem "sp" 8 ];
        insn "lw" [ reg "s3"; mem "sp" 12 ];
      ]
      @ (if U.Byte_size.equal min_vlenb U.Byte_size.zero then []
         else
           (* the vector registers' bytes, vlenb (CSR 0xc22): code that
              assumes VLEN 128 must not run on a smaller one *)
           [
             insn "csrr" [ reg "t0"; imm 0xc22 ];
             insn "li"
               [ reg "t1"; imm (Int64.to_int (U.Byte_size.to_int64 min_vlenb)) ];
             insn "li" [ reg "a0"; imm Exit.vlen ];
             insn "bltu" [ reg "t0"; reg "t1"; sym "harness_exit" ];
           ])
      @ (if has Placement.In then
           map ~path:"s2" ~len:lay.in_size ~flags:2 ~prot:3 ~into:"s4"
             ~open_code:Exit.open_in ~map_code:Exit.map_in ~name:"in"
         else [])
      @ map ~path:"s3" ~len:lay.out_size ~flags:1 ~prot:3 ~into:"s5"
          ~open_code:Exit.open_out ~map_code:Exit.map_out ~name:"out"
      @ [ insn "la" [ reg "t0"; sym table ] ]
      @ List.concat
          (List.mapi
             (fun k (s, off) ->
               let base =
                 match s.placement with
                 | Placement.In -> at "s4" off
                 | Out -> at "s5" off
                 | Bss ->
                     [
                       insn "la"
                         [
                           reg "t1";
                           sym_at bss_block
                             (Int64.to_int (U.Byte_offset.to_int64 off));
                         ];
                     ]
               in
               base @ [ insn "sw" [ reg "t1"; mem "t0" (4 * k) ] ])
             lay.slots)
      @ [
          (match abi with
          | Abi.Table -> insn "mv" [ reg "a0"; reg "t0" ]
          | Abi.Slot0 -> insn "lw" [ reg "a0"; mem "t0" 0 ]);
          insn "call" [ sym entry ];
          insn "sw" [ reg "a0"; mem "s5" 0 ];
          insn "li" [ reg "a0"; imm 0 ];
          label "harness_exit";
        ]
      @ syscall 94
    in
    let tables =
      [
        data;
        align 4;
        global table;
        label table;
        zero (max 4 (4 * List.length lay.slots));
      ]
      @
      if U.Byte_size.equal lay.bss_size U.Byte_size.zero then []
      else
        [
          bss;
          align 16;
          global bss_block;
          label bss_block;
          zero (Int64.to_int (U.Byte_size.to_int64 lay.bss_size));
        ]
    in
    start @ tables
  in
  module_ "harness" items

(* The bytes of a 128-bit vector register: what the vector profile's code
   assumes of the core. *)
let vlenb_128 = sure (U.Byte_size.of_int64 16L)
let base = 0x10000L

let elf_of ?profile ~entry modules =
  let render e = Fmt.str "%a" Rv32_image.Error.pp e in
  match Err.payload (Rv32_image.plan ?profile ~entry modules) with
  | Error e -> Error (render e)
  | Ok laid -> (
      match Err.payload (Rv32_image.bind ~base laid) with
      | Error e -> Error (render e)
      | Ok image -> Rv32_elf.write ?profile image)

(* An image and its harness, built once: the code does not change between
   runs. The profile it was built for picks the emulator's CPU. *)
type prepared = { elf : string; lay : layout; profile : Rv32_profile.t }

let prepare ?(profile = Rv32_profile.default) ?abi ~entry ~kernel slots =
  let* lay = layout slots in
  let min_vlenb =
    if Rv32_profile.vector profile then vlenb_128 else U.Byte_size.zero
  in
  match harness ?abi ~min_vlenb ~entry lay with
  | Error what -> Error (Fmt.str "the harness has no form for %s" what)
  | Ok h -> (
      match elf_of ~profile ~entry:"_start" (h :: kernel) with
      | Error e -> Error e
      | Ok elf -> Ok { elf; lay; profile })

type outcome = { status : int32; outputs : (Core.Table_slot.t * string) list }

(* One run: [bound k] is slot [k]'s initial bytes. The result holds every
   [Out] slot's bytes by index. [cpu] replaces the profile's qemu CPU (another
   VLEN, another tail policy). *)
let run ?timeout ?cpu (p : prepared) ~bound =
  let tmp suffix = Filename.temp_file "mltorch_rv32_" suffix in
  let in_path = tmp ".in" and out_path = tmp ".out" in
  Fun.protect
    ~finally:(fun () ->
      List.iter
        (fun f -> try Sys.remove f with Sys_error _ -> ())
        [ in_path; out_path ])
    (fun () ->
      let fill size slots_of =
        let b = Bytes.make (Int64.to_int (U.Byte_size.to_int64 size)) '\000' in
        List.iteri
          (fun k ((s : slot), off) ->
            if slots_of s then
              Option.iter
                (fun bytes ->
                  Bytes.blit_string bytes 0 b
                    (Int64.to_int (U.Byte_offset.to_int64 off))
                    (min (String.length bytes)
                       (Int64.to_int (U.Byte_size.to_int64 s.size))))
                (bound (Core.Table_slot.of_int k)))
          p.lay.slots;
        Bytes.unsafe_to_string b
      in
      Out_channel.with_open_bin in_path (fun oc ->
          output_string oc (fill p.lay.in_size (fun s -> s.placement = In)));
      Out_channel.with_open_bin out_path (fun oc ->
          output_string oc (fill p.lay.out_size (fun s -> s.placement = Out)));
      let* r =
        Rv32_process.execute ?timeout
          ?cpu:
            (match cpu with
            | Some _ -> cpu
            | None -> Rv32_profile.qemu_cpu p.profile)
          ~args:[ in_path; out_path ] p.elf
      in
      let* () =
        if r.Rv32_process.exit_code = 0 then Ok ()
        else Error (Fmt.str "the process exited %d" r.Rv32_process.exit_code)
      in
      let out = In_channel.with_open_bin out_path In_channel.input_all in
      if Int64.of_int (String.length out) <> U.Byte_size.to_int64 p.lay.out_size
      then
        Error
          (Fmt.str "the output file is %d bytes, expected %a"
             (String.length out) U.Byte_size.pp p.lay.out_size)
      else
        Ok
          {
            status = String.get_int32_le out 0;
            outputs =
              List.concat
                (List.mapi
                   (fun k ((s : slot), off) ->
                     if s.placement = Out then
                       [
                         ( Core.Table_slot.of_int k,
                           String.sub out
                             (Int64.to_int (U.Byte_offset.to_int64 off))
                             (Int64.to_int (U.Byte_size.to_int64 s.size)) );
                       ]
                     else [])
                   p.lay.slots);
          })
