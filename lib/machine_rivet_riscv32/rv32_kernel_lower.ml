module N = Asm_core.Normalized_ast
module D = Asm_core.Directive
module E = Riscv32_encode
module Item = Rv32_kernel_item
module Insn = Rv32_kernel_insn
module Label = Rv32_kernel_label
module Imm = Rv32_kernel_imm
module Vtype = Rv32_kernel_vtype
module Gpr = Rv32_kernel_reg.Gpr
module Fpr = Rv32_kernel_reg.Fpr
module Vreg = Rv32_kernel_reg.Vreg
module Label_map = Map.Make (Label)

module Rivet_form = struct
  type t = { at : Item.At.t; error : E.error }

  let pp ppf { at; error } = Fmt.pf ppf "%a: %a" Item.At.pp at E.pp_error error
end

type error = [ `Rivet_form of Rivet_form.t ]

let pp_error ppf : error -> unit = function
  | `Rivet_form f -> Rivet_form.pp ppf f

let gpr r = E.Operand.Reg (Gpr.to_rivet r)
let fpr r = E.Operand.Reg (Fpr.to_rivet r)
let vreg r = E.Operand.Reg (Vreg.to_rivet r)
let imm n = E.Operand.Imm (Foundation.Bigint.of_int n)
let sym name = E.Operand.Sym (Asm_core.Expr.Symbol name)

(* [offset(base)]. *)
let mem base offset =
  E.Operand.Mem
    {
      E.Mem.offset = Asm_core.Expr.Const (Foundation.Bigint.of_int offset);
      base = Gpr.to_rivet base;
    }

(* The operands in assembly order. *)
let operands ~spelling : Insn.t -> E.Operand.t list = function
  | Add { dst; lhs; rhs } -> [ gpr dst; gpr lhs; gpr rhs ]
  | Addi { dst; src; imm = i } -> [ gpr dst; gpr src; imm (Imm.Imm12.to_int i) ]
  | Bnez { src; target } -> [ gpr src; sym (spelling target) ]
  | Flh { dst; base; offset } | Flw { dst; base; offset } ->
      [ fpr dst; mem base (Imm.Imm12.to_int offset) ]
  | Li { dst; imm = i } -> [ gpr dst; imm (Imm.Li.to_int i) ]
  | Lui { dst; imm = i } -> [ gpr dst; imm (Imm.U20.to_int i) ]
  | Lw { dst; base; offset } -> [ gpr dst; mem base (Imm.Imm12.to_int offset) ]
  | Mv { dst; src } -> [ gpr dst; gpr src ]
  | Ret -> []
  | Srli { dst; src; shamt } ->
      [ gpr dst; gpr src; imm (Imm.Shamt.to_int shamt) ]
  | Sw { src; base; offset } -> [ gpr src; mem base (Imm.Imm12.to_int offset) ]
  | Vfmacc_vf { acc; scalar; src } | Vfwmaccbf16_vf { acc; scalar; src } ->
      [ vreg acc; fpr scalar; vreg src ]
  | Vle { dst; base; _ } -> [ vreg dst; mem base 0 ]
  | Vmacc_vx { acc; scalar; src } | Vwmacc_vx { acc; scalar; src } ->
      [ vreg acc; gpr scalar; vreg src ]
  | Vmv_v_i { dst; imm = i } -> [ vreg dst; imm (Imm.Simm5.to_int i) ]
  | Vmv_v_x { dst; src } -> [ vreg dst; gpr src ]
  | Vse { src; base; _ } -> [ vreg src; mem base 0 ]
  | Vsetvli { avl; vtype } ->
      let rd, rs1 =
        match avl with Reg r -> (Gpr.zero, r) | Vlmax { vl } -> (vl, Gpr.zero)
      in
      let word pp x = sym (Fmt.str "%a" pp x) in
      [
        gpr rd;
        gpr rs1;
        word Vtype.Sew.pp vtype.Vtype.sew;
        word Vtype.Lmul.pp vtype.Vtype.lmul;
        word (Vtype.Policy.pp ~prefix:"t") vtype.Vtype.tail;
        word (Vtype.Policy.pp ~prefix:"m") vtype.Vtype.mask;
      ]
  | Vv { dst; a; b; _ } -> [ vreg dst; vreg a; vreg b ]

let module_ ~profile kernel =
  let name = Rv32_kernel.name kernel in
  let items = Rv32_kernel.items kernel in
  let exports =
    List.fold_left
      (fun acc -> function
        | Item.Export { name; label } -> Label_map.add label name acc | _ -> acc)
      Label_map.empty items
  in
  let spelling label =
    match Label_map.find_opt label exports with
    | Some export -> export
    | None -> Fmt.str ".L%s_%s" name (Label.name label)
  in
  let state = Rv32_profile.state profile in
  let lower_item (index, note, rev) item =
    let at = { Item.At.index; note } in
    let note' = match item with Item.Note n -> n | _ -> note in
    let origin =
      Foundation.Origin.producer ~tool:"rv32_kernel"
        ~item:
          (Fmt.str "%s#%d%a" name index Fmt.(option (any " " ++ string)) note)
        ()
    in
    let directive directive = N.Directive { directive; origin } in
    let next items = Ok (index + 1, note', items @ rev) in
    match item with
    | Item.Align boundary -> next [ directive (D.Align { boundary }) ]
    | Export { name; _ } -> next [ directive (D.Global { name }) ]
    | Label l -> next [ N.Label { name = spelling l; origin } ]
    | Note _ -> next []
    | Text ->
        next
          [
            directive
              (D.Section
                 { name = ".text"; perms = Asm_core.Perms.rx; nobits = false });
          ]
    | Insn i -> (
        let form =
          match
            E.make_surface_instruction ~mnemonic:(Insn.mnemonic i) ~origin
              (operands ~spelling i)
          with
          | Ok s -> E.simplify_instruction state s
          | Error _ as e -> e
        in
        match form with
        | Ok insn -> next [ N.Instruction { insn; origin } ]
        | Error e ->
            Err.map_error ~pos:__POS__
              (fun error -> `Rivet_form { Rivet_form.at; error })
              (Error e))
  in
  let rec go state = function
    | [] -> Ok state
    | item :: rest -> (
        match lower_item state item with
        | Ok state -> go state rest
        | Error _ as e -> e)
  in
  Err.map
    (fun (_, _, rev) -> { N.unit_name = name; items = List.rev rev })
    (go (0, None, []) items)
