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
module Mtreg = Rv32_kernel_reg.Mtreg
module Vreg = Rv32_kernel_reg.Vreg
module Label_map = Map.Make (Label)

module Rivet_form = struct
  type t = { at : Item.At.t; error : E.error }

  let pp ppf { at; error } = Fmt.pf ppf "%a: %a" Item.At.pp at E.pp_error error
end

module Extension_form = struct
  type t = { at : Item.At.t; feature : string; message : string }

  let pp ppf { at; feature; message } =
    Fmt.pf ppf "%a: %s: %s" Item.At.pp at feature message
end

module Extension_profile = struct
  type t = {
    at : Item.At.t;
    feature : string;
    profile : Rv32_profile.t;
    missing : string list;
  }

  let pp ppf { at; feature; profile; missing } =
    Fmt.pf ppf "%a: %s needs %a, which %s lacks" Item.At.pp at feature
      Fmt.(list ~sep:comma string)
      missing
      (Rv32_profile.name profile)
end

module Extension_missing = struct
  type t = { at : Item.At.t; feature : string }

  let pp ppf { at; feature } =
    Fmt.pf ppf "%a: needs the %s extension, which was not given" Item.At.pp at
      feature
end

type error =
  [ `Extension_form of Extension_form.t
  | `Extension_missing of Extension_missing.t
  | `Extension_profile of Extension_profile.t
  | `Rivet_form of Rivet_form.t ]

let pp_error ppf : error -> unit = function
  | `Extension_form f -> Extension_form.pp ppf f
  | `Extension_missing m -> Extension_missing.pp ppf m
  | `Extension_profile p -> Extension_profile.pp ppf p
  | `Rivet_form f -> Rivet_form.pp ppf f

(* An operand as the kernel spells it, before an encoder's own operand type:
   the standard encoder and an extension's are different instances. Alphabetical. *)
module Operand = struct
  type t =
    | Fpr of Fpr.t
    | Gpr of Gpr.t
    | Imm of int
    | Mem of { base : Gpr.t; offset : int }  (** [offset(base)] *)
    | Sym of string
    | Vreg of Vreg.t
end

let gpr r = Operand.Gpr r
let fpr r = Operand.Fpr r
let vreg r = Operand.Vreg r
let imm n = Operand.Imm n
let sym name = Operand.Sym name
let mem base offset = Operand.Mem { base; offset }

(* Rivet spells a tile register as an indexed name. *)
let mtreg r = sym (Fmt.str "%a" Mtreg.pp r)

(* The operands in assembly order. *)
let operands ~spelling : Insn.t -> Operand.t list = function
  | Add { dst; lhs; rhs } | Sub { dst; lhs; rhs } ->
      [ gpr dst; gpr lhs; gpr rhs ]
  | Addi { dst; src; imm = i } | Andi { dst; src; imm = i } ->
      [ gpr dst; gpr src; imm (Imm.Imm12.to_int i) ]
  | Beqz { src; target } | Bnez { src; target } ->
      [ gpr src; sym (spelling target) ]
  | Csrr { dst; csr } -> [ gpr dst; imm (Insn.Csr.number csr) ]
  | Fence -> [ sym "rw"; sym "rw" ]
  | J { target } -> [ sym (spelling target) ]
  | La { dst; target } -> [ gpr dst; sym (spelling target) ]
  | Flh { dst; base; offset } | Flw { dst; base; offset } ->
      [ fpr dst; mem base (Imm.Imm12.to_int offset) ]
  | Li { dst; imm = i } -> [ gpr dst; imm (Imm.Li.to_int i) ]
  | Lui { dst; imm = i } -> [ gpr dst; imm (Imm.U20.to_int i) ]
  | Lw { dst; base; offset } -> [ gpr dst; mem base (Imm.Imm12.to_int offset) ]
  | Msetmtype { dst; mtype; vtype } -> [ gpr dst; gpr mtype; gpr vtype ]
  | Msettn { dst; avl } -> [ gpr dst; gpr avl ]
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
  | Vnsrl_wi { dst; src; shamt } ->
      [ vreg dst; vreg src; imm (Imm.Shamt.to_int shamt) ]
  | Vse { src; base; _ } -> [ vreg src; mem base 0 ]
  | Vsse { src; base; stride; _ } -> [ vreg src; mem base 0; gpr stride ]
  | Vlse { dst; base; stride; _ } -> [ vreg dst; mem base 0; gpr stride ]
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
  | Vtmm { acc; a; b; _ } -> [ mtreg acc; vreg a; vreg b ]
  | Vtle { slice; base; _ } | Vtse { slice; base; _ } ->
      [ gpr slice; mem base 0 ]
  | Vtzero { dst } -> [ mtreg dst ]
  | Vv { dst; a; b; _ } -> [ vreg dst; vreg a; vreg b ]

(* The standard encoder's spelling of an operand. *)
let to_rivet : Operand.t -> E.Operand.t = function
  | Fpr r -> E.Operand.Reg (Fpr.to_rivet r)
  | Gpr r -> E.Operand.Reg (Gpr.to_rivet r)
  | Imm n -> E.Operand.Imm (Foundation.Bigint.of_int n)
  | Mem { base; offset } ->
      E.Operand.Mem
        {
          E.Mem.offset = Asm_core.Expr.Const (Foundation.Bigint.of_int offset);
          base = Gpr.to_rivet base;
        }
  | Sym name -> E.Operand.Sym (Asm_core.Expr.Symbol name)
  | Vreg r -> E.Operand.Reg (Vreg.to_rivet r)

module Extension = struct
  type t = {
    feature : string;
    encode :
      Rv32_profile.t ->
      origin:Foundation.Origin.t ->
      mnemonic:string ->
      Operand.t list ->
      (string, [ `Profile_lacks of string list | `Rivet of string ]) result;
  }

  let feature t = t.feature

  (* An encoder over the standard extensions and [definition]'s rows, under the
     profile's features and [feature]: one instruction at a time, to its bytes.
     Rivet's diagnostics are rendered, being another encoder instance's. *)
  let make ~feature definition =
    let module Z =
      Riscv_family_encode.Make_with
        (Riscv32_encode.Profile)
        (struct
          let definition = definition
        end)
    in
    let operand : Operand.t -> Z.Operand.t = function
      | Fpr r -> Z.Operand.Reg (Z.Reg.F (Fpr.index r))
      | Gpr r -> Z.Operand.Reg (Z.Reg.X (Gpr.index r))
      | Imm n -> Z.Operand.Imm (Foundation.Bigint.of_int n)
      | Mem { base; offset } ->
          Z.Operand.Mem
            {
              Z.Mem.offset =
                Asm_core.Expr.Const (Foundation.Bigint.of_int offset);
              base = Z.Reg.X (Gpr.index base);
            }
      | Sym name -> Z.Operand.Sym (Asm_core.Expr.Symbol name)
      | Vreg r -> Z.Operand.Reg (Z.Reg.V (Vreg.index r))
    in
    let render r =
      Result.map_error
        (fun e -> `Rivet (Fmt.str "%a" Z.pp_error e))
        (Err.payload r)
    in
    let rivet r = Result.map_error (fun m -> `Rivet m) r in
    (* The profile's features and [feature], which must bring nothing else:
       Rivet enables what a feature requires, and a profile without it does not
       have it. *)
    let configure profile =
      let ( let* ) = Result.bind in
      let* base =
        rivet
          (Target_config.resolve Z.components (Rv32_profile.features profile))
      in
      let* config =
        rivet
          (Target_config.apply Z.components base
             [ Target_config.Enable feature ])
      in
      match
        List.filter
          (fun f -> f <> feature && not (Target_config.enabled base f))
          (Target_config.features config)
      with
      | [] -> Ok config
      | missing -> Error (`Profile_lacks missing)
    in
    let encode profile ~origin ~mnemonic ops =
      let ( let* ) = Result.bind in
      let* config = configure profile in
      let state = Z.initial_state config in
      let* surface =
        render
          (Z.make_surface_instruction ~mnemonic ~origin (List.map operand ops))
      in
      let* insn = render (Z.simplify_instruction state surface) in
      let* lowered = render (Z.lower_instruction state insn) in
      List.fold_left
        (fun acc l ->
          let* acc = acc in
          match render (Z.encode_in state l) with
          | Ok (`Fixed { Asm_core.Lowered_ast.bytes; fixups = []; _ }) ->
              Ok (acc ^ bytes)
          | Ok (`Fixed _ | `Relax _) ->
              Error
                (`Rivet
                   (mnemonic
                  ^ " needs a fixup, which an extension form cannot carry"))
          | Error _ as e -> e)
        (Ok "") lowered
    in
    { feature; encode }
end

(* The little-endian words of [bytes], a whole number of them. *)
let words bytes =
  List.init
    (String.length bytes / 4)
    (fun k ->
      Asm_core.Expr.Const
        (Foundation.Bigint.of_int64
           (Int64.logand
              (Int64.of_int32 (String.get_int32_le bytes (4 * k)))
              0xFFFF_FFFFL)))

let module_ ?(extensions = []) ~profile kernel =
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
    | Bss ->
        next
          [
            directive
              (D.Section
                 { name = ".bss"; perms = Asm_core.Perms.rw; nobits = true });
          ]
    | Zero length -> next [ directive (D.Zero { length }) ]
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
        match Insn.extension i with
        | Some feature -> (
            match
              List.find_opt
                (fun (x : Extension.t) -> String.equal x.feature feature)
                extensions
            with
            | None ->
                Err.fail ~pos:__POS__
                  (`Extension_missing { Extension_missing.at; feature })
            | Some x -> (
                match
                  x.encode profile ~origin ~mnemonic:(Insn.mnemonic i)
                    (operands ~spelling i)
                with
                | Ok bytes ->
                    next
                      [ directive (D.Data { width = 4; values = words bytes }) ]
                | Error (`Rivet message) ->
                    Err.fail ~pos:__POS__
                      (`Extension_form { Extension_form.at; feature; message })
                | Error (`Profile_lacks missing) ->
                    Err.fail ~pos:__POS__
                      (`Extension_profile
                         { Extension_profile.at; feature; profile; missing })))
        | None -> (
            let form =
              match
                E.make_surface_instruction ~mnemonic:(Insn.mnemonic i) ~origin
                  (List.map to_rivet (operands ~spelling i))
              with
              | Ok s -> E.simplify_instruction state s
              | Error _ as e -> e
            in
            match form with
            | Ok insn -> next [ N.Instruction { insn; origin } ]
            | Error e ->
                Err.map_error ~pos:__POS__
                  (fun error -> `Rivet_form { Rivet_form.at; error })
                  (Error e)))
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
