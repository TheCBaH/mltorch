(* One physical RV32 instruction as the Rivet instructions that encode it. The
   selected form says the operation and its immediates; the locations the
   allocator chose say the registers, by operand position. Every instruction
   is four bytes (no compressed extension is enabled) except [la] and [call],
   which are [auipc] and one more; a form's sequence spells out its
   instructions with the labels it branches to. A form Rivet cannot encode is
   a refusal naming it, never an approximation. *)

open Machine_ir
open Machine_target_riscv32
module Loc = Mir_phys.Loc
module A = Machine_rivet_riscv32.Rv32_asm
module O = Rv32_op

(* Fault injection for the evidence suite: each is one deliberate mapping
   defect a run under qemu must detect. No consumer passes one. *)
module Mutation = struct
  type t =
    | Branch_sense  (** a fall-through branch tested with the taken sense *)
    | Commuted_sub  (** a binary32 subtraction's operands swapped *)
    | Shifted_slot  (** a region's base read from the next slot of the table *)
    | Select_arms  (** a select's two arms exchanged *)
    | Unset_vtype
        (** no [vsetivli] before the vector forms: they run under whatever
            configuration the process starts with *)
end

module Refusal = struct
  type t =
    | External of string  (** a helper this route cannot link *)
    | Location of string  (** an operand where the form needs a register *)
    | No_reference of string  (** a reference the artifact does not record *)
    | Unencodable of string  (** a mnemonic Rivet does not take *)

  let pp fmt = function
    | External n -> Fmt.pf fmt "helper %s has no RV32 implementation" n
    | Location w -> Fmt.pf fmt "%s is not in a register" w
    | No_reference w -> Fmt.pf fmt "no relocation for %s" w
    | Unencodable m -> Fmt.pf fmt "rivet has no RV32 form for %s" m
end

(* What one emitted item is. A conditional branch stays symbolic until the
   function is laid out: one instruction when its target is within reach,
   else the inverted branch over a [jal]. *)
module Item = struct
  type t =
    | Branch of { mnemonic : string; regs : string list; label : string }
    | Insn of { mnemonic : string; ops : A.E.Operand.t list }
    | Label of string

  let bytes = function
    | Branch _ -> 4
    | Insn { mnemonic = "la" | "call"; _ } -> 8
    | Insn _ -> 4
    | Label _ -> 0
end

type env = {
  mutation : Mutation.t option;
  esc : Refusal.t Err.Escape.t;
  reference : Mir_target.Reference.t -> (string * int64) option;
      (** the symbol and addend the artifact records for this reference *)
  table_slot : string -> int option;
      (** the slot of the caller's table a region symbol is addressed through;
          [None]: the symbol is resident in the image *)
  fresh_label : unit -> string;
}

let refuse env r = Err.Escape.throw env.esc r
let mutated env m = env.mutation = Some m
let i mnemonic ops = Item.Insn { mnemonic; ops }
let imm k = A.imm (Int64.to_int k)
let table = Rv32_reg.table.Mir_target.View.name
let scratch = Rv32_reg.scratch.Mir_target.View.name

let name env what (l : Loc.t) =
  match l with
  | Loc.Reg v -> v.Mir_target.View.name
  | Loc.Slot _ | Loc.Mem _ -> refuse env (Refusal.Location what)

let r env what l = A.reg (name env what l)

let nth env what ls k =
  match List.nth_opt ls k with
  | Some l -> l
  | None -> refuse env (Refusal.Location (Fmt.str "%s %d" what k))

let imm12 k = Int64.compare k (-2048L) >= 0 && Int64.compare k 2047L <= 0

(* [k] as the [lui] and [addi] immediates that sum to it modulo 2^32. *)
let split k =
  let low = Int64.sub (Int64.logxor (Int64.logand k 0xFFFL) 0x800L) 0x800L in
  (Int64.logand (Int64.shift_right (Int64.sub k low) 12) 0xFFFFFL, low)

(* [rd = k] for any 32-bit [k]. *)
let load_imm rd k =
  if imm12 k then [ i "li" [ A.reg rd; imm k ] ]
  else
    let high, low = split k in
    i "lui" [ A.reg rd; imm high ]
    ::
    (if Int64.equal low 0L then []
     else [ i "addi" [ A.reg rd; A.reg rd; imm low ] ])

let address env reference what =
  match env.reference reference with
  | Some x -> x
  | None -> refuse env (Refusal.No_reference what)

(* A memory operand [k(base)] for any [k] up to 4094: above 2047 the reserved
   scratch first holds [base + 2047]. *)
let mem_at base k =
  if imm12 k then ([], A.mem base (Int64.to_int k))
  else
    ( [ i "addi" [ A.reg scratch; A.reg base; imm 2047L ] ],
      A.mem scratch (Int64.to_int (Int64.sub k 2047L)) )

(* A select as branches around moves: [p] is read before [rd] is written, so
   any register sharing is sound. *)
let select env ~move ~rd ~p ~a ~b =
  let a, b = if mutated env Mutation.Select_arms then (b, a) else (a, b) in
  let l1 = env.fresh_label () in
  if String.equal rd a then
    [ Item.Branch { mnemonic = "bnez"; regs = [ p ]; label = l1 } ]
    @ move rd b @ [ Item.Label l1 ]
  else if String.equal rd b then
    [ Item.Branch { mnemonic = "beqz"; regs = [ p ]; label = l1 } ]
    @ move rd a @ [ Item.Label l1 ]
  else
    let l2 = env.fresh_label () in
    [ Item.Branch { mnemonic = "bnez"; regs = [ p ]; label = l1 } ]
    @ move rd b
    @ [ i "j" [ A.sym l2 ]; Item.Label l1 ]
    @ move rd a @ [ Item.Label l2 ]

let gmove rd s =
  if String.equal rd s then [] else [ i "mv" [ A.reg rd; A.reg s ] ]

let fmove rd s =
  if String.equal rd s then []
  else [ i "fsgnj.s" [ A.reg rd; A.reg s; A.reg s ] ]

(* {1 Vectors}

   Every vector form runs under SEW 32, LMUL 1, vl 4, tail and mask agnostic,
   which the module writer establishes ({!vsetivli}) before the first vector
   item of a block and after a call. A form needing another configuration
   (the tail-undisturbed slide of a lane insert) restores it. *)

let vsetivli ?(vl = 4) ?(tail = "ta") () =
  i "vsetivli"
    [ A.reg "zero"; A.imm vl; A.sym "e32"; A.sym "m1"; A.sym tail; A.sym "ma" ]

let v0 = Rv32_reg.vector_scratch.Mir_target.View.name

let vmove rd s =
  if String.equal rd s then [] else [ i "vmv.v.v" [ A.reg rd; A.reg s ] ]

let vitems env (op : Rv32_vop.t) ~(uses : Loc.t list) ~(defs : Loc.t list) :
    Item.t list =
  let u k = name env "use" (nth env "use" uses k)
  and d k = name env "def" (nth env "def" defs k) in
  let ru k = A.reg (u k) and rd k = A.reg (d k) in
  let module V = Rv32_vop in
  match op with
  | V.Vcast _ | V.Vmv _ -> vmove (d 0) (u 0)
  | V.Vext { signed; from; first; _ } ->
      let ext =
        Printf.sprintf "v%cext.vf%d"
          (if signed then 's' else 'z')
          (32 / Mir_width.bits from)
      in
      let sew = Printf.sprintf "e%d" (Mir_width.bits from) in
      if first = 0 then [ i ext [ rd 0; ru 0 ] ]
      else
        [
          i "vsetivli"
            [
              A.reg "zero";
              A.imm (128 / Mir_width.bits from);
              A.sym sew;
              A.sym "m1";
              A.sym "ta";
              A.sym "ma";
            ];
          i "vslidedown.vi" [ A.reg v0; ru 0; A.imm first ];
          vsetivli ();
          i ext [ rd 0; A.reg v0 ];
        ]
  | V.Vle_int (w, _) ->
      let b = Mir_width.bits w in
      [
        i "vsetivli"
          [
            A.reg "zero";
            A.imm (128 / b);
            A.sym (Printf.sprintf "e%d" b);
            A.sym "m1";
            A.sym "ta";
            A.sym "ma";
          ];
        i (Printf.sprintf "vle%d.v" b) [ rd 0; A.mem (u 0) 0 ];
        vsetivli ();
      ]
  | V.Vshift (o, k, _) -> [ i (Rv32_kind.Shift.name o) [ rd 0; ru 0; A.imm k ] ]
  | V.Vfbin (Rv32_kind.Fop.Sub, _, _) when mutated env Mutation.Commuted_sub ->
      [ i "vfsub.vv" [ rd 0; ru 1; ru 0 ] ]
  | V.Vfbin (o, _, _) -> [ i (Rv32_kind.Fop.vname o) [ rd 0; ru 0; ru 1 ] ]
  | V.Vfcmp (o, _, _) ->
      [
        i (Rv32_kind.Fcmp.vname o) [ A.reg v0; ru 0; ru 1 ];
        i "vmv.v.i" [ rd 0; A.imm 0 ];
        i "vmerge.vim" [ rd 0; rd 0; A.imm (-1); A.reg v0 ];
      ]
  | V.Vfcvt_f_x _ -> [ i "vfcvt.f.x.v" [ rd 0; ru 0 ] ]
  | V.Vfcvt_x_f _ -> [ i "vfcvt.rtz.x.f.v" [ rd 0; ru 0 ] ]
  | V.Vfmacc _ -> [ i "vfmacc.vv" [ rd 0; ru 1; ru 2 ] ]
  | V.Vfmv_f_s (0, _) -> [ i "vfmv.f.s" [ rd 0; ru 0 ] ]
  | V.Vfmv_f_s (k, _) ->
      [
        i "vslidedown.vi" [ A.reg v0; ru 0; A.imm k ];
        i "vfmv.f.s" [ rd 0; A.reg v0 ];
      ]
  | V.Vfmv_v_f _ -> [ i "vfmv.v.f" [ rd 0; ru 0 ] ]
  | V.Vfsqrt _ -> [ i "vfsqrt.v" [ rd 0; ru 0 ] ]
  | V.Vinsert (k, _, _) ->
      (* the scalar to v0's first lane, then slid up to lane [k] with only
         lanes [0, k] active and the rest undisturbed *)
      vmove (d 0) (u 0)
      @ [
          i "vfmv.s.f" [ A.reg v0; ru 1 ];
          vsetivli ~vl:(k + 1) ~tail:"tu" ();
          i "vslideup.vi" [ rd 0; A.reg v0; A.imm k ];
          vsetivli ();
        ]
  | V.Vle32 _ -> [ i "vle32.v" [ rd 0; A.mem (u 0) 0 ] ]
  | V.Vlogic (o, _, _) -> [ i (Rv32_kind.Logic.name o) [ rd 0; ru 0; ru 1 ] ]
  | V.Vlse32 _ -> [ i "vlse32.v" [ rd 0; A.mem (u 0) 0; ru 1 ] ]
  | V.Vmerge _ ->
      let o, t =
        if mutated env Mutation.Select_arms then (ru 1, ru 0) else (ru 0, ru 1)
      in
      [
        i "vmsne.vi" [ A.reg v0; ru 2; A.imm 0 ];
        i "vmerge.vvm" [ rd 0; o; t; A.reg v0 ];
      ]
  | V.Vmv_v_x _ -> [ i "vmv.v.x" [ rd 0; ru 0 ] ]
  | V.Vnot _ -> [ i "vxor.vi" [ rd 0; ru 0; A.imm (-1) ] ]
  | V.Vse32 _ -> [ i "vse32.v" [ ru 1; A.mem (u 0) 0 ] ]
  | V.Vsgnj (o, _, _) -> [ i (Rv32_kind.Sgnj.vname o) [ rd 0; ru 0; ru 1 ] ]
  | V.Vsse32 _ -> [ i "vsse32.v" [ ru 2; A.mem (u 0) 0; ru 1 ] ]

let items env (op : O.t) ~(uses : Loc.t list) ~(defs : Loc.t list) : Item.t list
    =
  let u k = name env "use" (nth env "use" uses k)
  and d k = name env "def" (nth env "def" defs k) in
  let ru k = A.reg (u k) and rd k = A.reg (d k) in
  match op with
  | O.Add_wide (_, k) ->
      let high, low = split k in
      if Int64.equal high 0L then [ i "addi" [ rd 0; ru 0; imm low ] ]
      else
        i "lui" [ rd 0; imm high ]
        ::
        (if Int64.equal low 0L then [] else [ i "addi" [ rd 0; rd 0; imm low ] ])
        @ [ i "add" [ rd 0; ru 0; rd 0 ] ]
  | O.Addi (_, k) -> [ i "addi" [ rd 0; ru 0; imm k ] ]
  | O.Alu (o, _, _) -> [ i (O.Alu.name o) [ rd 0; ru 0; ru 1 ] ]
  | O.Alui (o, _, k) -> [ i (O.Alui.name o) [ rd 0; ru 0; imm k ] ]
  | O.Call { callee; _ } ->
      let symbol, _ =
        address env (Mir_target.Reference.Call callee) "a call's target"
      in
      [ i "call" [ A.sym symbol ] ]
  | O.Fbin (O.Fop.Sub, _, _) when mutated env Mutation.Commuted_sub ->
      [ i "fsub.s" [ rd 0; ru 1; ru 0 ] ]
  | O.Fbin (o, _, _) -> [ i (O.Fop.name o) [ rd 0; ru 0; ru 1 ] ]
  | O.Fcmp (o, _, _) -> [ i (O.Fcmp.name o) [ rd 0; ru 0; ru 1 ] ]
  | O.Fcvt_s_w _ -> [ i "fcvt.s.w" [ rd 0; ru 0 ] ]
  | O.Fcvt_s_wu _ -> [ i "fcvt.s.wu" [ rd 0; ru 0 ] ]
  | O.Fcvt_w_s _ -> [ i "fcvt.w.s" [ rd 0; ru 0; A.sym "rtz" ] ]
  | O.Flw (_, k) -> [ i "flw" [ rd 0; A.mem (u 0) (Int64.to_int k) ] ]
  | O.Fmadd _ -> [ i "fmadd.s" [ rd 0; ru 0; ru 1; ru 2 ] ]
  | O.Fmv_w_x _ -> [ i "fmv.w.x" [ rd 0; ru 0 ] ]
  | O.Fmv_x_w _ -> [ i "fmv.x.w" [ rd 0; ru 0 ] ]
  | O.Fsel _ -> select env ~move:fmove ~rd:(d 0) ~p:(u 0) ~a:(u 1) ~b:(u 2)
  | O.Fsgnj (o, _, _) -> [ i (O.Sgnj.name o) [ rd 0; ru 0; ru 1 ] ]
  | O.Fsqrt _ -> [ i "fsqrt.s" [ rd 0; ru 0 ] ]
  | O.Fsw (_, k, _) -> [ i "fsw" [ ru 1; A.mem (u 0) (Int64.to_int k) ] ]
  | O.La view -> (
      let reference =
        Mir_target.Reference.View (view, Mir_target.Reference.Form.Pc_relative)
      in
      let symbol, addend = address env reference "a view's address" in
      match env.table_slot symbol with
      | Some slot ->
          (* a table-bound region: its base from the caller's table, then
             the view's offset in it *)
          let slot =
            if mutated env Mutation.Shifted_slot then slot + 1 else slot
          in
          i "lw" [ rd 0; A.mem table (4 * slot) ]
          ::
          (if Int64.equal addend 0L then []
           else if imm12 addend then [ i "addi" [ rd 0; rd 0; imm addend ] ]
           else
             load_imm scratch addend @ [ i "add" [ rd 0; rd 0; A.reg scratch ] ])
      | None -> [ i "la" [ rd 0; A.sym_at symbol (Int64.to_int addend) ] ])
  | O.Li (_, k) -> [ i "li" [ rd 0; imm k ] ]
  | O.Load (m, _, k) ->
      [ i (O.Msz.load m) [ rd 0; A.mem (u 0) (Int64.to_int k) ] ]
  | O.Lui k -> [ i "lui" [ rd 0; imm k ] ]
  | O.Mv _ | O.Zext _ -> gmove (d 0) (u 0)
  | O.Sel _ -> select env ~move:gmove ~rd:(d 0) ~p:(u 0) ~a:(u 1) ~b:(u 2)
  | O.Sext (w, _) ->
      let k = Int64.of_int (32 - Mir_width.bits w) in
      [ i "slli" [ rd 0; ru 0; imm k ]; i "srai" [ rd 0; rd 0; imm k ] ]
  | O.Store (m, _, k, _) ->
      [ i (O.Msz.store m) [ ru 1; A.mem (u 0) (Int64.to_int k) ] ]
  | O.Sw_pair (_, k, _, _) ->
      [
        i "sw" [ ru 1; A.mem (u 0) (Int64.to_int k) ];
        i "sw" [ ru 2; A.mem (u 0) (Int64.to_int (Int64.add k 4L)) ];
      ]
  | O.Trunc (Mir_width.W8, _) -> [ i "andi" [ rd 0; ru 0; imm 255L ] ]
  | O.Trunc (_, _) ->
      [ i "slli" [ rd 0; ru 0; imm 16L ]; i "srli" [ rd 0; rd 0; imm 16L ] ]
  | O.Vec x -> vitems env x ~uses ~defs

(* A transfer between locations: a register copy in its bank, or a word
   between a register and frame memory. *)
(* [scratch = base + offset], the address a vector access takes whole. A frame
   offset is at most 4094, so two additions reach it, and the base may be the
   scratch itself. *)
let address ~(base : Mir_target.View.t) offset =
  let b = A.reg base.Mir_target.View.name in
  if imm12 offset then [ i "addi" [ A.reg scratch; b; imm offset ] ]
  else
    [
      i "addi" [ A.reg scratch; b; imm 2047L ];
      i "addi" [ A.reg scratch; A.reg scratch; imm (Int64.sub offset 2047L) ];
    ]

let transfer env ~(dst : Loc.t) ~(src : Loc.t) : Item.t list =
  let fp (v : Mir_target.View.t) =
    v.Mir_target.View.bank = Mir_target.Bank.Fpr
  in
  match (dst, src) with
  | Loc.Reg d, Loc.Reg s when d.Mir_target.View.bank = Mir_target.Bank.Vector ->
      vmove d.Mir_target.View.name s.Mir_target.View.name
  | Loc.Reg d, Loc.Reg s ->
      if fp d then fmove d.Mir_target.View.name s.Mir_target.View.name
      else gmove d.Mir_target.View.name s.Mir_target.View.name
  | Loc.Reg d, Loc.Mem { base; offset; bytes = 16L } ->
      (* sixteen bytes, whatever VLEN: a whole-register load would read
         VLEN/8 *)
      address ~base offset
      @ [ i "vle32.v" [ A.reg d.Mir_target.View.name; A.mem scratch 0 ] ]
  | Loc.Mem { base; offset; bytes = 16L }, Loc.Reg s ->
      address ~base offset
      @ [ i "vse32.v" [ A.reg s.Mir_target.View.name; A.mem scratch 0 ] ]
  | Loc.Reg d, Loc.Mem { base; offset; bytes = 4L } ->
      let pre, m = mem_at base.Mir_target.View.name offset in
      pre
      @ [ i (if fp d then "flw" else "lw") [ A.reg d.Mir_target.View.name; m ] ]
  | Loc.Mem { base; offset; bytes = 4L }, Loc.Reg s ->
      let pre, m = mem_at base.Mir_target.View.name offset in
      pre
      @ [ i (if fp s then "fsw" else "sw") [ A.reg s.Mir_target.View.name; m ] ]
  | _ -> refuse env (Refusal.Location "a transfer's operand")

let stack_step delta = [ i "addi" [ A.reg "sp"; A.reg "sp"; imm delta ] ]

(* A conditional branch to [label], or its inverse. *)
let branch env (test : O.test) ~(uses : Loc.t list) ~label ~inverted =
  let u k = name env "branch operand" (nth env "use" uses k) in
  let inverted =
    if mutated env Mutation.Branch_sense then not inverted else inverted
  in
  match test with
  | O.Bcmp (c, _, _) ->
      let c = if inverted then O.Bcond.invert c else c in
      Item.Branch { mnemonic = O.Bcond.name c; regs = [ u 0; u 1 ]; label }
  | O.Beqz _ ->
      Item.Branch
        {
          mnemonic = (if inverted then "bnez" else "beqz");
          regs = [ u 0 ];
          label;
        }
  | O.Bnez _ ->
      Item.Branch
        {
          mnemonic = (if inverted then "beqz" else "bnez");
          regs = [ u 0 ];
          label;
        }

let invert_mnemonic = function
  | "beq" -> "bne"
  | "bne" -> "beq"
  | "blt" -> "bge"
  | "bge" -> "blt"
  | "bltu" -> "bgeu"
  | "bgeu" -> "bltu"
  | "beqz" -> "bnez"
  | "bnez" -> "beqz"
  | m -> invalid_arg ("Rv32_mir_form.invert_mnemonic: " ^ m)
