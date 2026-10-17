(* The binary32 C library under the default profile: each input is passed
   through [truncf] in qemu without the D extension, and the result compared
   bitwise with the host's [Float.trunc] of the same binary32 (any NaN equals
   any NaN). *)

open Machine_rivet_riscv32
module H = Rv32_harness
open Rv32_asm

let boundary =
  [
    0x00000000l (* 0 *);
    0x80000000l (* -0 *);
    0x00000001l (* least subnormal *);
    0x80000001l;
    0x3f000000l (* 0.5 *);
    0xbf000000l (* -0.5 *);
    0x3f7fffffl (* largest below 1 *);
    0x3fc00000l (* 1.5 *);
    0xbfc00000l (* -1.5 *);
    0x40200000l (* 2.5 *);
    0x42f78000l (* 123.75 *);
    0x4affffffl (* 2^23 - 0.5 *);
    0xcaffffffl (* -(2^23 - 0.5) *);
    0x4b000000l (* 2^23 *);
    0x4b000001l (* 2^23 + 1 *);
    0x4f000000l (* 2^31 *);
    0xcf000000l (* -2^31 *);
    0x7f7fffffl (* largest finite *);
    0x7f800000l (* inf *);
    0xff800000l (* -inf *);
    0x7fc00000l (* quiet NaN *);
    0x7f800001l (* signalling NaN *);
  ]

let sweep =
  let s = Random.State.make [| 42 |] in
  List.init 4096 (fun _ -> Int32.of_int (Random.State.bits s land 0xFFFF_FFFF))

(* [mir_entry(table)]: [truncf] of each binary32 in table[0], into table[1]. *)
let kernel count =
  module_ "kernel" (fun () ->
      [
        text;
        align 4;
        global "mir_entry";
        label "mir_entry";
        insn "addi" [ reg "sp"; reg "sp"; imm (-16) ];
        insn "sw" [ reg "ra"; mem "sp" 12 ];
        insn "sw" [ reg "s0"; mem "sp" 8 ];
        insn "sw" [ reg "s1"; mem "sp" 4 ];
        insn "sw" [ reg "s2"; mem "sp" 0 ];
        insn "lw" [ reg "s0"; mem "a0" 0 ];
        insn "lw" [ reg "s1"; mem "a0" 4 ];
      ]
      @ load_imm "s2" count
      @ [
          label "each";
          insn "flw" [ reg "fa0"; mem "s0" 0 ];
          insn "call" [ sym "truncf" ];
          insn "fsw" [ reg "fa0"; mem "s1" 0 ];
          insn "addi" [ reg "s0"; reg "s0"; imm 4 ];
          insn "addi" [ reg "s1"; reg "s1"; imm 4 ];
          insn "addi" [ reg "s2"; reg "s2"; imm (-1) ];
          insn "bnez" [ reg "s2"; sym "each" ];
          insn "lw" [ reg "ra"; mem "sp" 12 ];
          insn "lw" [ reg "s0"; mem "sp" 8 ];
          insn "lw" [ reg "s1"; mem "sp" 4 ];
          insn "lw" [ reg "s2"; mem "sp" 0 ];
          insn "addi" [ reg "sp"; reg "sp"; imm 16 ];
          insn "li" [ reg "a0"; imm 0 ];
          insn "ret" [];
        ])

let host x = Int32.bits_of_float (Float.trunc (Int32.float_of_bits x))

let same a b =
  let nan x = Float.is_nan (Int32.float_of_bits x) in
  Int32.equal a b || (nan a && nan b)

(* The machine's [truncf] of each input, in order. *)
let machine inputs =
  let n = List.length inputs in
  let size = 4 * n in
  let slots =
    H.
      [
        { size; align = 8; placement = Placement.In };
        { size; align = 8; placement = Placement.Out };
      ]
  in
  let bytes = Bytes.create size in
  List.iteri (fun i x -> Bytes.set_int32_le bytes (4 * i) x) inputs;
  let ( let* ) = Result.bind in
  let* k = Result.map_error (( ^ ) "no form for ") (kernel n) in
  let* libc = Result.map_error (( ^ ) "no form for ") (Rv32_libc.module_ ()) in
  let* prepared = H.prepare ~entry:"mir_entry" ~kernel:[ k; libc ] slots in
  let* { H.outputs; _ } =
    H.run prepared ~bound:(fun k ->
        if k = 0 then Some (Bytes.to_string bytes) else None)
  in
  match List.assoc_opt 1 outputs with
  | None -> Error "no output slot"
  | Some s -> Ok (List.init n (fun i -> String.get_int32_le s (4 * i)))

let%expect_test "truncf at the boundaries" =
  (match machine boundary with
  | Error e -> Fmt.pr "%s@." e
  | Ok got ->
      List.iter2
        (fun x y ->
          Fmt.pr "%08lx -> %08lx%s@." x y
            (if same y (host x) then "" else Fmt.str " (host %08lx)" (host x)))
        boundary got);
  [%expect
    {|
    00000000 -> 00000000
    80000000 -> 80000000
    00000001 -> 00000000
    80000001 -> 80000000
    3f000000 -> 00000000
    bf000000 -> 80000000
    3f7fffff -> 00000000
    3fc00000 -> 3f800000
    bfc00000 -> bf800000
    40200000 -> 40000000
    42f78000 -> 42f60000
    4affffff -> 4afffffe
    caffffff -> cafffffe
    4b000000 -> 4b000000
    4b000001 -> 4b000001
    4f000000 -> 4f000000
    cf000000 -> cf000000
    7f7fffff -> 7f7fffff
    7f800000 -> 7f800000
    ff800000 -> ff800000
    7fc00000 -> 7fc00000
    7f800001 -> 7f800001 |}]

let%expect_test "truncf over a seeded sweep" =
  (match machine sweep with
  | Error e -> Fmt.pr "%s@." e
  | Ok got ->
      let wrong =
        List.fold_left2
          (fun n x y -> if same y (host x) then n else n + 1)
          0 sweep got
      in
      Fmt.pr "%d inputs, %d differ from the host@." (List.length sweep) wrong);
  [%expect {| 4096 inputs, 0 differ from the host |}]
