(* Branch layout: a conditional branch whose target is beyond a B-type's
   reach becomes the inverted branch over a [jal], and the program still
   takes the right path under qemu, in both directions across a long body. *)

module A = Machine_rivet_riscv32.Rv32_asm
module F = Machine_rivet_riscv32_mir.Rv32_mir_form
module Mo = Machine_rivet_riscv32_mir.Rv32_mir_module
module H = Machine_rivet_riscv32.Rv32_harness
module P = Machine_rivet_riscv32.Rv32_process
module Profile = Machine_rivet_riscv32.Rv32_profile

let pad n =
  List.init n (fun _ -> F.i "addi" [ A.reg "t0"; A.reg "t0"; A.imm 1 ])

(* a0 = 1 + 2 when [taken], else 1: a forward branch over [n] instructions to
   a block that adds 2 and a backward one returning to the exit. *)
let program ~n ~taken =
  [
    F.i "li" [ A.reg "a0"; A.imm 1 ];
    F.i "li" [ A.reg "t1"; A.imm (if taken then 1 else 0) ];
    F.Item.Branch { mnemonic = "bnez"; regs = [ "t1" ]; label = "far" };
    F.Item.Label "back";
    F.i "li" [ A.reg "a7"; A.imm 94 ];
    F.i "ecall" [];
  ]
  @ pad n
  @ [
      F.Item.Label "far";
      F.i "addi" [ A.reg "a0"; A.reg "a0"; A.imm 2 ];
      F.i "li" [ A.reg "t2"; A.imm 1 ];
      F.Item.Branch { mnemonic = "bnez"; regs = [ "t2" ]; label = "back" };
    ]

let run ~n ~taken =
  let items = Mo.lay_out (program ~n ~taken) in
  let longs =
    List.length
      (List.filter
         (function F.Item.Insn { mnemonic = "j"; _ } -> true | _ -> false)
         items)
  in
  match
    A.module_ "layout" (fun () ->
        [ A.text; A.global "_start"; A.label "_start" ] @ Mo.render items)
  with
  | Error what -> Fmt.pr "no form for %s@." what
  | Ok m -> (
      match H.elf_of ~entry:"_start" [ m ] with
      | Error e -> Fmt.pr "elf: %s@." e
      | Ok elf -> (
          match P.execute ?cpu:(Profile.qemu_cpu Profile.default) elf with
          | Error e -> Fmt.pr "run: %s@." e
          | Ok { P.exit_code; _ } ->
              Fmt.pr "n=%d taken=%b: %d long, exit %d@." n taken longs exit_code
          ))

let%expect_test "branches across a long body" =
  List.iter
    (fun n -> List.iter (fun taken -> run ~n ~taken) [ false; true ])
    [ 10; 990; 1100; 3000 ];
  [%expect
    {|
    n=10 taken=false: 0 long, exit 1
    n=10 taken=true: 0 long, exit 3
    n=990 taken=false: 0 long, exit 1
    n=990 taken=true: 0 long, exit 3
    n=1100 taken=false: 2 long, exit 1
    n=1100 taken=true: 2 long, exit 3
    n=3000 taken=false: 2 long, exit 1
    n=3000 taken=true: 2 long, exit 3 |}]
