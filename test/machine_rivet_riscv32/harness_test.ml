(* The mmap harness over a hand-written kernel: the input file's doubles are
   negated into a workspace in the ELF and copied to the shared output file. *)

open Machine_rivet_riscv32
module H = Rv32_harness
open Rv32_asm

let count = 8
let doubles = 8 * count

let slots =
  H.
    [
      { size = doubles; align = 8; placement = Placement.In };
      { size = doubles; align = 8; placement = Placement.Bss };
      { size = doubles; align = 8; placement = Placement.Out };
    ]

(* [mir_entry(table)]: table[0] input, table[1] workspace, table[2] output. *)
let kernel ~negate =
  let copy ~src ~dst ~op ~name =
    [
      insn "lw" [ reg "t0"; mem "a0" (4 * src) ];
      insn "lw" [ reg "t1"; mem "a0" (4 * dst) ];
      insn "li" [ reg "t2"; imm count ];
      label name;
      insn "fld" [ reg "ft0"; mem "t0" 0 ];
    ]
    @ (if op then [ insn "fneg.d" [ reg "ft0"; reg "ft0" ] ] else [])
    @ [
        insn "fsd" [ reg "ft0"; mem "t1" 0 ];
        insn "addi" [ reg "t0"; reg "t0"; imm 8 ];
        insn "addi" [ reg "t1"; reg "t1"; imm 8 ];
        insn "addi" [ reg "t2"; reg "t2"; imm (-1) ];
        insn "bnez" [ reg "t2"; sym name ];
      ]
  in
  module_ "kernel" (fun () ->
      [ text; align 4; global "mir_entry"; label "mir_entry" ]
      @ copy ~src:0 ~dst:1 ~op:negate ~name:"negate"
      @ copy ~src:1 ~dst:2 ~op:false ~name:"copy"
      @ [ insn "li" [ reg "a0"; imm 7 ]; insn "ret" [] ])

let input =
  String.concat ""
    (List.init count (fun i ->
         let b = Bytes.create 8 in
         Bytes.set_int64_le b 0
           (Int64.bits_of_float (float_of_int (i + 1) *. 0.5));
         Bytes.to_string b))

let show_doubles s =
  List.init count (fun i -> Int64.float_of_bits (String.get_int64_le s (8 * i)))
  |> Fmt.str "%a" Fmt.(list ~sep:(any " ") float)

let run ~negate =
  match kernel ~negate with
  | Error what -> Fmt.pr "no form for %s@." what
  | Ok k -> (
      match H.prepare ~entry:"mir_entry" ~kernel:[ k ] slots with
      | Error e -> Fmt.pr "prepare: %s@." e
      | Ok p -> (
          match
            H.run p ~bound:(fun k -> if k = 0 then Some input else None)
          with
          | Error e -> Fmt.pr "run: %s@." e
          | Ok { H.status; outputs } ->
              Fmt.pr "status %ld@." status;
              List.iter
                (fun (k, bytes) ->
                  Fmt.pr "slot %d: %s@." k (show_doubles bytes))
                outputs))

let%expect_test "negated through workspace" =
  run ~negate:true;
  [%expect {|
    status 7
    slot 2: -0.5 -1 -1.5 -2 -2.5 -3 -3.5 -4 |}]

let%expect_test "copied" =
  run ~negate:false;
  [%expect {|
    status 7
    slot 2: 0.5 1 1.5 2 2.5 3 3.5 4 |}]

(* Each failure of the harness is its own exit code, none a skip. *)
let prepared =
  lazy
    (match kernel ~negate:false with
    | Error what -> failwith what
    | Ok k -> (
        match H.prepare ~entry:"mir_entry" ~kernel:[ k ] slots with
        | Ok p -> p
        | Error e -> failwith e))

let exec args =
  match Rv32_process.execute ~args (Lazy.force prepared).H.elf with
  | Error e -> Fmt.pr "run: %s@." e
  | Ok { Rv32_process.exit_code; _ } -> Fmt.pr "exit %d@." exit_code

let%expect_test "harness failures" =
  let missing = Filename.concat (Filename.get_temp_dir_name ()) "rv32_absent" in
  let sized = Filename.temp_file "rv32_" ".sized" in
  Out_channel.with_open_bin sized (fun oc ->
      output_string oc (String.make 4096 '\000'));
  exec [];
  exec [ missing; sized ];
  exec [ sized; missing ];
  List.iter Sys.remove [ sized ];
  [%expect {|
    exit 100
    exit 101
    exit 103 |}]
