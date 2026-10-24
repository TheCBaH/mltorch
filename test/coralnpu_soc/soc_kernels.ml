(* Typed kernels on the subsystem simulators: binary32 negation from one slot
   into another and copied to a third, the slots in DTCM and in SRAM, the result
   checked by the mailbox's hashes; then guests that fail one way each. Prints
   one line per run; a wrong answer is a changed line. *)

open Machine_rivet_riscv32
open Rv32_asm
module L = Coral_layout

let dir = Sys.argv.(1)

let or_exit pp = function
  | Ok v -> v
  | Error e ->
      Fmt.epr "%a@." pp e;
      exit 1

let profile name =
  or_exit Coral_profile.pp_error (Err.payload (Coral_profile.load ~dir name))

(* [mir_entry(table)]: table[0] negated into table[1], copied to table[2];
   returns 7. [body] replaces the work for the failing guests. *)
let kernel ?body count =
  let copy ~src ~dst ~negate ~name =
    [
      insn "lw" [ reg "t0"; mem "a0" (4 * src) ];
      insn "lw" [ reg "t1"; mem "a0" (4 * dst) ];
    ]
    @ load_imm "t2" count
    @ [ label name; insn "flw" [ reg "ft0"; mem "t0" 0 ] ]
    @ (if negate then [ insn "fneg.s" [ reg "ft0"; reg "ft0" ] ] else [])
    @ [
        insn "fsw" [ reg "ft0"; mem "t1" 0 ];
        insn "addi" [ reg "t0"; reg "t0"; imm 4 ];
        insn "addi" [ reg "t1"; reg "t1"; imm 4 ];
        insn "addi" [ reg "t2"; reg "t2"; imm (-1) ];
        insn "bnez" [ reg "t2"; sym name ];
      ]
  in
  let work =
    match body with
    | Some b -> b
    | None ->
        copy ~src:0 ~dst:1 ~negate:true ~name:"negate"
        @ copy ~src:1 ~dst:2 ~negate:false ~name:"copy"
  in
  or_exit Fmt.string
    (module_ "kernel" (fun () ->
         [ text; align 4; global "mir_entry"; label "mir_entry" ]
         @ work
         @ [ insn "li" [ reg "a0"; imm 7 ]; insn "ret" [] ]))

let slots ~home count =
  let size = Int64.of_int (4 * count) in
  L.Placement.
    [
      L.Slot.make ~home ~align:16L In size;
      L.Slot.make ~home ~align:16L Scratch size;
      L.Slot.make ~home ~align:16L Out size;
    ]

let input count =
  let b = Bytes.create (4 * count) in
  for i = 0 to count - 1 do
    Bytes.set_int32_le b (4 * i) (Int32.bits_of_float (float_of_int (i + 1)))
  done;
  Bytes.to_string b

let negated count =
  let b = Bytes.create (4 * count) in
  for i = 0 to count - 1 do
    Bytes.set_int32_le b (4 * i) (Int32.bits_of_float (-.float_of_int (i + 1)))
  done;
  Bytes.to_string b

let image ?body ?(home = L.Home.Any) p count =
  or_exit Coral_image.pp_error
    (Err.payload
       (Coral_image.build p ~entry:"mir_entry"
          ~kernel:[ kernel ?body count ]
          (slots ~home count)))

let placement (i : Coral_image.t) =
  String.concat " "
    (List.map (fun (s : L.Placed_slot.t) -> s.region) i.layout.slots)

let run ?deadline ?(dir = dir) ?(expect = negated) label (i : Coral_image.t)
    count =
  match
    Err.payload
      (Coral_soc_run.run ?deadline ~dir i ~bound:(fun k ->
           if k = 0 then Some (input count) else None))
  with
  | Ok o ->
      Fmt.pr "%s: status %ld, %s, minstret %Ld, mcycle %Ld@." label o.status
        (if Coral_soc_run.matches o 2 (expect count) then "negated" else "WRONG")
        o.minstret o.mcycle
  | Error e -> Fmt.pr "%s: %a@." label Coral_soc_run.Run_error.pp e

let profiles =
  [
    "subsystem-matrix-highmem";
    "subsystem-matrix-normal";
    "subsystem-vector-highmem";
    "subsystem-vector-normal";
  ]

let () =
  List.iter
    (fun name ->
      let p = profile name in
      List.iter
        (fun (home, count) ->
          let i = image ~home p count in
          Fmt.pr "%s %d floats: %s@." name count (placement i);
          run
            (Printf.sprintf "%s %d %s" name count (L.Home.to_string home))
            i count)
        [ (L.Home.Any, 8); (L.Home.Ram, 8) ])
    profiles;
  let p = profile "subsystem-vector-normal" in
  let i = image ~home:Ram p 4096 in
  Fmt.pr "4096 floats: %s@." (placement i);
  run "4096 ram" i 4096;
  (* A check that cannot pass: the hash of other bytes. *)
  run ~expect:input "8 against its input" (image p 8) 8;
  (* One way to fail each. *)
  let fails what body = run what (image ~body p 8) 8 in
  fails "illegal instruction" [ words [ 0 ] ];
  fails "store to ITCM" [ insn "sw" [ reg "zero"; mem "zero" 0x100 ] ];
  run ~deadline:20. "spin"
    (image ~body:[ label "spin"; insn "j" [ sym "spin" ] ] p 8)
    8;
  (* A host that believes another layout. *)
  let i = image p 8 in
  run "wrong digest"
    { i with layout = { i.layout with digest = Int32.add i.layout.digest 1l } }
    8;
  run ~dir:"/nonexistent" "missing simulator" (image p 8) 8
