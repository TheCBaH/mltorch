(* Branch ranges. AArch64 declares no relaxation ladder, so a branch that cannot
   reach is a bind error, never a truncated word. *)

open Asm_core
module Image = Machine_rivet_aarch64.Rivet_a64_image

let origin = Foundation.Origin.synthesized ~pass:"range_test" ()

(* [b.eq far] over [bytes] bytes of filler, and a return at [far]. *)
let across ~cond bytes =
  let i op ops =
    Normalized_ast.Instruction
      { insn = { Aarch64.Instruction.op; ops }; origin }
  in
  let d directive = Normalized_ast.Directive { directive; origin } in
  (* zero fill: a quarter of a million instructions would exhaust the stack of
     a list-recursive pass long before any branch was out of reach *)
  let filler = [ d (Directive.Zero { length = bytes }) ] in
  {
    Normalized_ast.unit_name = "range";
    items =
      [
        d
          (Directive.Section
             { name = ".text"; perms = Perms.rx; nobits = false });
        d (Directive.Align { boundary = 4 });
        d (Directive.Global { name = "f" });
        Normalized_ast.Label { name = "f"; origin };
        (if cond then
           i (Aarch64.Opcode.Bcond Aarch64.Cond.Eq)
             [ Aarch64.Operand.Sym (Expr.Symbol "far") ]
         else i Aarch64.Opcode.B [ Aarch64.Operand.Sym (Expr.Symbol "far") ]);
      ]
      @ filler
      @ [
          Normalized_ast.Label { name = "far"; origin }; i Aarch64.Opcode.Ret [];
        ];
  }

(* A fixup's value needs its address, so the range is checked when the image is
   bound to the addresses the host chose: by the loader. *)
let verdict ~cond bytes =
  match Err.payload (Image.plan ~entry:"f" [ across ~cond bytes ]) with
  | Error e -> Fmt.str "not planned: %a" Image.Error.pp e
  | Ok laid -> (
      match Err.payload (Image.load laid) with
      | Ok loaded ->
          Image.close loaded;
          "loaded"
      | Error e -> Fmt.str "refused: %a" Image.Error.pp e)

let%expect_test
    "a conditional branch reaches 1 MiB and an unconditional 128 MiB" =
  Fmt.pr "b.eq over 64 KiB: %s@." (verdict ~cond:true 0x1_0000);
  Fmt.pr "b.eq over 1 MiB - 8: %s@." (verdict ~cond:true (0x10_0000 - 8));
  Fmt.pr "b.eq over 1 MiB + 8: %s@." (verdict ~cond:true (0x10_0000 + 8));
  Fmt.pr "b over 4 MiB: %s@." (verdict ~cond:false 0x40_0000);
  [%expect
    {|
    b.eq over 64 KiB: loaded
    b.eq over 1 MiB - 8: loaded
    b.eq over 1 MiB + 8: refused: <synthesized by image>: error[aarch64.fixup]: branch target is out of range
    b over 4 MiB: loaded |}]
