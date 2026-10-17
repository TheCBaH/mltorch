(* Typed kernels on the real simulators: binary32 negation through a scratch
   slot, small (every slot in DTCM) and large (64 KiB slots, staged through
   testbench RAM where DTCM is too small), the same in place in one [In_out]
   slot, and guests built to fail one way each. Prints one line per run; exits 0, a wrong answer is a changed line. *)

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

(* [mir_entry(region)]: the region's [count] floats negated in place; returns
   7. *)
let in_place count =
  or_exit Fmt.string
    (module_ "kernel" (fun () ->
         [ text; align 4; global "mir_entry"; label "mir_entry" ]
         @ load_imm "t2" count
         @ [
             label "negate";
             insn "flw" [ reg "ft0"; mem "a0" 0 ];
             insn "fneg.s" [ reg "ft0"; reg "ft0" ];
             insn "fsw" [ reg "ft0"; mem "a0" 0 ];
             insn "addi" [ reg "a0"; reg "a0"; imm 4 ];
             insn "addi" [ reg "t2"; reg "t2"; imm (-1) ];
             insn "bnez" [ reg "t2"; sym "negate" ];
             insn "li" [ reg "a0"; imm 7 ];
             insn "ret" [];
           ]))

let slots count =
  let size = Int64.of_int (4 * count) in
  L.Placement.
    [
      { L.Slot.align = 8L; placement = In; size };
      { L.Slot.align = 8L; placement = Scratch; size };
      { L.Slot.align = 8L; placement = Out; size };
    ]

let input count =
  let b = Bytes.create (4 * count) in
  for i = 0 to count - 1 do
    Bytes.set_int32_le b (4 * i)
      (Int32.bits_of_float (float_of_int (i + 1) *. 0.5))
  done;
  Bytes.to_string b

(* "negated" when every output is the negated input, else the first miss. *)
let check count out =
  let rec go i =
    if i = count then "negated"
    else
      let got = Int32.float_of_bits (String.get_int32_le out (4 * i)) in
      if got = -.(float_of_int (i + 1) *. 0.5) then go (i + 1)
      else Printf.sprintf "slot 2[%d] = %h" i got
  in
  go 0

let image ?body p count =
  or_exit Coral_image.pp_error
    (Err.payload
       (Coral_image.build p ~entry:"mir_entry"
          ~kernel:[ kernel ?body count ]
          (slots count)))

let in_place_image p count =
  or_exit Coral_image.pp_error
    (Err.payload
       (Coral_image.build ~abi:Slot0 p ~entry:"mir_entry"
          ~kernel:[ in_place count ]
          [
            {
              L.Slot.align = 8L;
              placement = In_out;
              size = Int64.of_int (4 * count);
            };
          ]))

let placement (i : Coral_image.t) =
  String.concat " "
    (List.map
       (fun (s : L.Placed_slot.t) ->
         s.region ^ if s.htif_reachable then "" else "(staged)")
       i.layout.slots)

let run ?deadline ?flags ?(dir = dir) ?(out = 2) label (i : Coral_image.t) count
    =
  match
    Err.payload
      (Coral_run.run ?deadline ?flags ~dir i ~bound:(fun k ->
           if k = 0 then Some (input count) else None))
  with
  | Ok o ->
      Fmt.pr "%s: status %ld, %s, minstret %Ld, mcycle %s@." label o.status
        (check count (List.assoc out o.outputs))
        o.minstret
        (if o.mcycle > 0L then "counted" else "0")
  | Error e -> Fmt.pr "%s: %a@." label Coral_run.Run_error.pp e

let () =
  List.iter
    (fun name ->
      let p = profile name in
      List.iter
        (fun count ->
          let i = image p count in
          Fmt.pr "%s %d floats: %s@." name count (placement i);
          run (Printf.sprintf "%s %d" name count) i count)
        [ 8; 16384 ])
    Coral_profile.baseline;
  List.iter
    (fun name ->
      let p = profile name in
      List.iter
        (fun count ->
          let i = in_place_image p count in
          Fmt.pr "%s %d floats in place: %s@." name count (placement i);
          run ~out:0 (Printf.sprintf "%s %d in place" name count) i count)
        [ 8; 16384 ])
    Coral_profile.baseline;
  (* One way to fail each, on one MPACT and one RTL profile. *)
  List.iter
    (fun name ->
      let p = profile name in
      let fails what body = run (name ^ " " ^ what) (image ~body p 8) 8 in
      fails "illegal instruction" [ words [ 0 ] ];
      fails "store to ITCM" [ insn "sw" [ reg "zero"; mem "zero" 0x100 ] ];
      run ~deadline:5. (name ^ " spin")
        (image ~body:[ label "spin"; insn "j" [ sym "spin" ] ] p 8)
        8)
    [ "core-mini-axi"; "mpact-v2" ];
  (* The testbench's own limit: it reports it and keeps waiting (a producer
     defect), so the deadline ends the run and the text names the cause. *)
  run ~deadline:10. ~flags:[ "--cycles=20000" ] "core-mini-axi cycle limit"
    (image
       ~body:[ label "spin"; insn "j" [ sym "spin" ] ]
       (profile "core-mini-axi") 8)
    8;
  (* A host that believes another layout: the guest checks the digest first. *)
  let i = image (profile "mpact-v2") 8 in
  run "mpact-v2 wrong digest"
    { i with layout = { i.layout with digest = Int32.add i.layout.digest 1l } }
    8;
  run ~dir:"/nonexistent" "missing simulator" (image (profile "mpact-v2") 8) 8
