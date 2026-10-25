open Machine_rivet_riscv32
module K = Rv32_kernel
module Item = Rv32_kernel_item
module Insn = Rv32_kernel_insn
module Imm = Rv32_kernel_imm
module Label = Rv32_kernel_label
module Vtype = Rv32_kernel_vtype
open Rv32_kernel_reg

let finalize ?(name = "k") items =
  match Err.payload (K.finalize ~name items) with
  | Ok k -> Fmt.pr "ok: %d items@." (List.length (K.items k))
  | Error e -> Fmt.pr "%a@." K.pp_error e

let loop = Label.named "loop"
let finish = Label.named "finish"

let counted =
  [
    Item.text;
    Item.export ~name:"peak_body" loop;
    Item.label loop;
    Item.insn
      (Insn.Addi { dst = Gpr.a1; src = Gpr.a1; imm = Imm.Imm12.minus_one });
    Item.insn (Insn.Bnez { src = Gpr.a1; target = loop });
    Item.insn Insn.Ret;
  ]

let%expect_test "a well-formed kernel finalizes" =
  finalize counted;
  [%expect {| ok: 6 items |}]

let%expect_test "each row, once" =
  let at_x = [ Item.note "setup" ] in
  finalize ~name:"1k" [];
  finalize (counted @ [ Item.label loop ]);
  finalize
    (counted @ [ Item.export ~name:"peak_body" finish; Item.label finish ]);
  finalize (counted @ [ Item.export ~name:"other" loop ]);
  finalize (counted @ [ Item.export ~name:"never" (Label.named "nowhere") ]);
  finalize (at_x @ [ Item.insn (Insn.Bnez { src = Gpr.a0; target = finish }) ]);
  finalize
    (at_x
    @ [
        Item.insn
          (Insn.Srli { dst = Gpr.t3; src = Gpr.a4; shamt = Imm.Shamt.of_int 32 });
      ]);
  finalize
    [
      Item.insn
        (Insn.Vsetvli
           {
             avl = Vtype.Avl.Vlmax { vl = Gpr.zero };
             vtype =
               {
                 Vtype.sew = Vtype.Sew.E8;
                 lmul = Vtype.Lmul.M4;
                 tail = Vtype.Policy.Agnostic;
                 mask = Vtype.Policy.Agnostic;
               };
           });
    ];
  finalize [ Item.label (Label.named "bad name") ];
  [%expect
    {|
    kernel name "1k" is not [A-Za-z_][A-Za-z0-9_]*
    item 6: label loop is already defined at item 2
    item 6: export peak_body is already exported at item 1
    item 6: label loop already has an export name, at item 1
    item 6: export never names label nowhere, which is never defined
    item 1 (setup): label finish is never defined
    item 1 (setup): 32 is not a shift amount
    item 0: vsetvli with rd = rs1 = x0 keeps vl; VLMAX needs a destination
    item 0: label name "bad name" is not [A-Za-z_][A-Za-z0-9_]* |}]

let%expect_test "several failures are reported together, in item order" =
  finalize ~name:"k"
    [
      Item.insn (Insn.Bnez { src = Gpr.a0; target = Label.named "missing" });
      Item.insn
        (Insn.Addi { dst = Gpr.a0; src = Gpr.a0; imm = Imm.Imm12.of_int 4096 });
      Item.label loop;
      Item.label loop;
    ];
  [%expect
    {|
    item 0: label missing is never defined
    item 1: 4096 is not a signed 12-bit immediate
    item 3: label loop is already defined at item 2 |}]

let%expect_test "a named label has one identity wherever it was made" =
  finalize
    [
      Item.label (Label.named "loop");
      Item.insn (Insn.Bnez { src = Gpr.a0; target = Label.named "loop" });
    ];
  finalize [ Item.label (Label.named "loop"); Item.label (Label.named "loop") ];
  [%expect
    {|
    ok: 2 items
    item 1: label loop is already defined at item 0 |}]

let%expect_test "two kernels may use the same label names" =
  finalize ~name:"a" counted;
  finalize ~name:"b" counted;
  [%expect {|
    ok: 6 items
    ok: 6 items |}]

let%expect_test "an annotation is cleared, not inherited" =
  let bad =
    Item.insn
      (Insn.Srli { dst = Gpr.t3; src = Gpr.a4; shamt = Imm.Shamt.of_int 99 })
  in
  finalize [ Item.note "first"; bad; Item.clear_note; bad ];
  [%expect
    {|
    item 1 (first): 99 is not a shift amount
    item 3: 99 is not a shift amount |}]

let%expect_test "one failure is its own row, several are a list" =
  let shape items =
    match Err.payload (K.finalize ~name:"k" items) with
    | Ok _ -> Fmt.pr "ok@."
    | Error (`Invalid _) -> Fmt.pr "Invalid@."
    | Error (`Kernel_invalid is) ->
        Fmt.pr "Kernel_invalid %d@." (List.length is)
  in
  shape [ Item.label loop; Item.label loop ];
  shape [ Item.label loop; Item.label loop; Item.label loop ];
  [%expect {|
    Invalid
    Kernel_invalid 2 |}]

let%expect_test "an alignment is a positive power of two" =
  List.iter (fun n -> finalize [ Item.align n ]) [ 1; 4; 8192; 3; 0; -4 ];
  [%expect
    {|
    ok: 1 items
    ok: 1 items
    ok: 1 items
    item 0: alignment 3 is not a positive power of two
    item 0: alignment 0 is not a positive power of two
    item 0: alignment -4 is not a positive power of two |}]

let%expect_test
    "a zero fill is not negative, and a label address needs its label" =
  List.iter (fun n -> finalize [ Item.bss; Item.zero n ]) [ 0; 4096; -1 ];
  finalize
    [
      Item.text;
      Item.insn (Insn.La { dst = Gpr.s8; target = Label.named "nowhere" });
    ];
  finalize
    [
      Item.text;
      Item.insn (Insn.La { dst = Gpr.s8; target = loop });
      Item.bss;
      Item.label loop;
      Item.zero 16;
    ];
  [%expect
    {|
    ok: 2 items
    ok: 2 items
    item 1: a zero fill of -1 bytes
    item 1: label nowhere is never defined
    ok: 5 items |}]
