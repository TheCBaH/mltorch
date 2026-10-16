(* The mmap harness over a hand-written kernel: the input file's floats are
   negated into a workspace in the ELF and copied to the shared output file.
   The kernel is binary32 under the default profile, which has no D extension;
   its binary64 twin is refused there and runs under the binary64 profile. *)

open Machine_rivet_riscv32
module H = Rv32_harness
open Rv32_asm

let count = 8

(* One float format: its width and the kernel's load, negation and store. *)
module Precision = struct
  type t = { bytes : int; load : string; neg : string; store : string }

  let binary32 = { bytes = 4; load = "flw"; neg = "fneg.s"; store = "fsw" }
  let binary64 = { bytes = 8; load = "fld"; neg = "fneg.d"; store = "fsd" }
end

let slots (p : Precision.t) =
  let size = p.bytes * count in
  H.
    [
      { size; align = 8; placement = Placement.In };
      { size; align = 8; placement = Placement.Bss };
      { size; align = 8; placement = Placement.Out };
    ]

(* [mir_entry(table)]: table[0] input, table[1] workspace, table[2] output. *)
let kernel (p : Precision.t) ~negate =
  let copy ~src ~dst ~op ~name =
    [
      insn "lw" [ reg "t0"; mem "a0" (4 * src) ];
      insn "lw" [ reg "t1"; mem "a0" (4 * dst) ];
      insn "li" [ reg "t2"; imm count ];
      label name;
      insn p.load [ reg "ft0"; mem "t0" 0 ];
    ]
    @ (if op then [ insn p.neg [ reg "ft0"; reg "ft0" ] ] else [])
    @ [
        insn p.store [ reg "ft0"; mem "t1" 0 ];
        insn "addi" [ reg "t0"; reg "t0"; imm p.bytes ];
        insn "addi" [ reg "t1"; reg "t1"; imm p.bytes ];
        insn "addi" [ reg "t2"; reg "t2"; imm (-1) ];
        insn "bnez" [ reg "t2"; sym name ];
      ]
  in
  module_ "kernel" (fun () ->
      [ text; align 4; global "mir_entry"; label "mir_entry" ]
      @ copy ~src:0 ~dst:1 ~op:negate ~name:"negate"
      @ copy ~src:1 ~dst:2 ~op:false ~name:"copy"
      @ [ insn "li" [ reg "a0"; imm 7 ]; insn "ret" [] ])

let encode (p : Precision.t) x =
  let b = Bytes.create p.bytes in
  if p.bytes = 4 then Bytes.set_int32_le b 0 (Int32.bits_of_float x)
  else Bytes.set_int64_le b 0 (Int64.bits_of_float x);
  Bytes.to_string b

let decode (p : Precision.t) s i =
  if p.bytes = 4 then Int32.float_of_bits (String.get_int32_le s (4 * i))
  else Int64.float_of_bits (String.get_int64_le s (8 * i))

let input p =
  String.concat ""
    (List.init count (fun i -> encode p (float_of_int (i + 1) *. 0.5)))

let show p s =
  List.init count (decode p s) |> Fmt.str "%a" Fmt.(list ~sep:(any " ") float)

let run ?profile ?(precision = Precision.binary32) ~negate () =
  let p = precision in
  match kernel p ~negate with
  | Error what -> Fmt.pr "no form for %s@." what
  | Ok k -> (
      match H.prepare ?profile ~entry:"mir_entry" ~kernel:[ k ] (slots p) with
      | Error e -> Fmt.pr "prepare: %s@." e
      | Ok prepared -> (
          match
            H.run prepared ~bound:(fun k ->
                if k = 0 then Some (input p) else None)
          with
          | Error e -> Fmt.pr "run: %s@." e
          | Ok { H.status; outputs } ->
              Fmt.pr "status %ld@." status;
              List.iter
                (fun (k, bytes) -> Fmt.pr "slot %d: %s@." k (show p bytes))
                outputs))

let%expect_test "negated through workspace" =
  run ~negate:true ();
  [%expect {|
    status 7
    slot 2: -0.5 -1 -1.5 -2 -2.5 -3 -3.5 -4 |}]

let%expect_test "copied" =
  run ~negate:false ();
  [%expect {|
    status 7
    slot 2: 0.5 1 1.5 2 2.5 3 3.5 4 |}]

(* The binary64 kernel: Rivet refuses its D instructions under the default
   profile, and the binary64 profile runs it. *)
let%expect_test "binary64 only under the binary64 profile" =
  run ~precision:Precision.binary64 ~negate:true ();
  run ~profile:Rv32_profile.Rv32imafd_ilp32d ~precision:Precision.binary64
    ~negate:true ();
  [%expect
    {|
    prepare: <synthesized by riscv32>: error[riscv32.feature]: fld requires feature d, which is not enabled
    <synthesized by riscv32>: error[riscv32.feature]: fneg.d requires feature d, which is not enabled
    <synthesized by riscv32>: error[riscv32.feature]: fsd requires feature d, which is not enabled
    <synthesized by riscv32>: error[riscv32.feature]: fld requires feature d, which is not enabled
    <synthesized by riscv32>: error[riscv32.feature]: fsd requires feature d, which is not enabled
    status 7
    slot 2: -0.5 -1 -1.5 -2 -2.5 -3 -3.5 -4 |}]

(* Each failure of the harness is its own exit code, none a skip. *)
let prepared =
  lazy
    (match kernel Precision.binary32 ~negate:false with
    | Error what -> failwith what
    | Ok k -> (
        match
          H.prepare ~entry:"mir_entry" ~kernel:[ k ] (slots Precision.binary32)
        with
        | Ok p -> p
        | Error e -> failwith e))

let exec args =
  let p = Lazy.force prepared in
  match
    Rv32_process.execute ?cpu:(Rv32_profile.qemu_cpu p.H.profile) ~args p.H.elf
  with
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
