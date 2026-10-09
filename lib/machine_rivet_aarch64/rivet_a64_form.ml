(* One physical AArch64 instruction as the Rivet instructions that encode it.
   The selected form says the operation and its immediates; the locations the
   allocator chose say the registers, by operand position. A form Rivet cannot
   encode is a refusal naming it, never an approximation. *)

open Machine_ir
open Machine_target_aarch64
module Loc = Mir_phys.Loc
module R = Rivet_a64_refusal
module A = A64_op
module I = Aarch64.Instruction
module O = Aarch64.Operand
module Op = Aarch64.Opcode

(* Fault injection for the evidence suite: each is one deliberate mapping
   defect a native comparison must detect. No consumer passes one. *)
module Mutation = struct
  type t =
    | Branch_sense  (** a fall-through branch tested with the taken sense *)
    | Commuted_sub  (** a subtraction's operands swapped *)
    | Dropped_lo12  (** a symbol's low twelve bits left out of its address *)
    | Narrow_spill  (** a 64-bit spill or reload moved as 32 bits *)
end

type env = {
  mutation : Mutation.t option;
  esc : R.t Err.Escape.t;
  reference : Mir_target.Reference.t -> (string * int64) option;
      (** the symbol and addend the artifact records for this reference *)
  table_slot : string -> int option;
      (** the slot of the caller's table a region symbol is addressed through;
          [None]: the symbol is resident in the image *)
}

let refuse env r = Err.Escape.throw env.esc r
let origin = Foundation.Origin.synthesized ~pass:"machine_rivet_aarch64" ()
let ins op ops = { I.op; ops }
let imm n = O.Imm (Foundation.Bigint.of_int64 n)
let imm_int n = imm (Int64.of_int n)
let sym name = O.Sym (Asm_core.Expr.Symbol name)

(* The constructor name of a selected form: the unit of the coverage report. *)
let form_name : A.t -> string = function
  | A.Add _ -> "Add"
  | A.Add_imm _ -> "Add_imm"
  | A.Add_lo12 _ -> "Add_lo12"
  | A.Adrp _ -> "Adrp"
  | A.Bl _ -> "Bl"
  | A.Cmp _ -> "Cmp"
  | A.Cmp_imm _ -> "Cmp_imm"
  | A.Csel _ -> "Csel"
  | A.Cset _ -> "Cset"
  | A.Dup_elem _ -> "Dup_elem"
  | A.Dup_half _ -> "Dup_half"
  | A.Dup_lane _ -> "Dup_lane"
  | A.Ext _ -> "Ext"
  | A.Fbin _ -> "Fbin"
  | A.Fcmp _ -> "Fcmp"
  | A.Fcsel _ -> "Fcsel"
  | A.Fcvt _ -> "Fcvt"
  | A.Fcvtl _ -> "Fcvtl"
  | A.Fcvtn _ -> "Fcvtn"
  | A.Fcvtzs _ -> "Fcvtzs"
  | A.Fmadd _ -> "Fmadd"
  | A.Fmov _ -> "Fmov"
  | A.Fmov_from_gpr _ -> "Fmov_from_gpr"
  | A.Fmov_to_gpr _ -> "Fmov_to_gpr"
  | A.Funary _ -> "Funary"
  | A.Ins_half _ -> "Ins_half"
  | A.Ins_lane _ -> "Ins_lane"
  | A.Ld1_lane _ -> "Ld1_lane"
  | A.Ld1r _ -> "Ld1r"
  | A.Ldr _ -> "Ldr"
  | A.Ldr_vec _ -> "Ldr_vec"
  | A.Logic _ -> "Logic"
  | A.Logic_imm _ -> "Logic_imm"
  | A.Mov _ -> "Mov"
  | A.Movk _ -> "Movk"
  | A.Movn _ -> "Movn"
  | A.Movz _ -> "Movz"
  | A.Mrs_fpcr _ -> "Mrs_fpcr"
  | A.Msr_fpcr _ -> "Msr_fpcr"
  | A.Msub _ -> "Msub"
  | A.Mul _ -> "Mul"
  | A.Scvtf _ -> "Scvtf"
  | A.Sdiv _ -> "Sdiv"
  | A.Shift_imm _ -> "Shift_imm"
  | A.St1_lane _ -> "St1_lane"
  | A.Str _ -> "Str"
  | A.Str_vec _ -> "Str_vec"
  | A.Sub _ -> "Sub"
  | A.Sub_imm _ -> "Sub_imm"
  | A.Sxtw _ -> "Sxtw"
  | A.Trunc _ -> "Trunc"
  | A.Uxtw _ -> "Uxtw"
  | A.Vfbin _ -> "Vfbin"
  | A.Vfmla _ -> "Vfmla"
  | A.Vfunary _ -> "Vfunary"
  | A.Vmov _ -> "Vmov"
  | A.Vwiden _ -> "Vwiden"
  | A.Wtrunc _ -> "Wtrunc"

(* {1 Registers} *)

let sp_unit = Mir_id.Unit.to_int A64_reg.sp.Mir_target.View.unit
let width_bits = function A.Sz.W -> 32 | A.Sz.X -> 64

(* A general register view as a register of [width] bits. The stack pointer's
   unit is register 31 with the stack-pointer meaning; no other view is 31. *)
let gpr env ~width (v : Mir_target.View.t) =
  let unit = Mir_id.Unit.to_int v.Mir_target.View.unit in
  match v.Mir_target.View.bank with
  | Mir_target.Bank.Gpr when unit = sp_unit ->
      { Aarch64.Reg.num = 31; width; is_sp = true }
  | Mir_target.Bank.Gpr when unit >= 0 && unit <= 30 ->
      { Aarch64.Reg.num = unit; width; is_sp = false }
  | _ -> refuse env (R.Register v.Mir_target.View.name)

let fpr env ~double (v : Mir_target.View.t) =
  let unit = Mir_id.Unit.to_int v.Mir_target.View.unit in
  match v.Mir_target.View.bank with
  | Mir_target.Bank.Fpr when unit >= 32 && unit <= 63 ->
      { Aarch64.Freg.num = unit - 32; double }
  | _ -> refuse env (R.Register v.Mir_target.View.name)

let loc_view env = function
  | Loc.Reg v -> v
  | Loc.Slot _ -> refuse env (R.Location "a frame slot")
  | Loc.Mem _ -> refuse env (R.Location "memory")

let g env sz l = O.Reg (gpr env ~width:(width_bits sz) (loc_view env l))
let f env fsz l = O.Freg (fpr env ~double:(fsz = A.Fsz.D) (loc_view env l))
let dbl = function A.Fsz.D -> true | A.Fsz.S -> false

(* The nth use or definition. *)
let nth env what l k =
  match List.nth_opt l k with
  | Some x -> x
  | None -> refuse env (R.Location (Fmt.str "a missing %s %d" what k))

let cond_name (c : A.Cond.t) = A.Cond.name c

(* {1 Memory} *)

let mem env ~base ~offset =
  O.Mem
    {
      Aarch64.Mem.base = gpr env ~width:64 (loc_view env base);
      offset = Aarch64.Disp.Const offset;
      writeback = false;
      pre = true;
    }

(* A load or store of the access size [m] against [base, #offset]. *)
let access env ~load (m : A.Msz.t) ~rt ~base ~offset =
  let mem = mem env ~base ~offset in
  let gp ~width = O.Reg (gpr env ~width (loc_view env rt)) in
  let fp ~double = O.Freg (fpr env ~double (loc_view env rt)) in
  let op, reg =
    match (m, load) with
    | A.Msz.B, true -> (Op.Ldrb, gp ~width:32)
    | A.Msz.B, false -> (Op.Strb, gp ~width:32)
    | A.Msz.H, true -> (Op.Ldrh, gp ~width:32)
    | A.Msz.H, false -> (Op.Strh, gp ~width:32)
    | A.Msz.W, true -> (Op.Ldr, gp ~width:32)
    | A.Msz.W, false -> (Op.Str, gp ~width:32)
    | A.Msz.X, true -> (Op.Ldr, gp ~width:64)
    | A.Msz.X, false -> (Op.Str, gp ~width:64)
    | A.Msz.S, true -> (Op.Ldr, fp ~double:false)
    | A.Msz.S, false -> (Op.Str, fp ~double:false)
    | A.Msz.D, true -> (Op.Ldr, fp ~double:true)
    | A.Msz.D, false -> (Op.Str, fp ~double:true)
  in
  ins op [ reg; mem ]

(* {1 Selected forms} *)

let address env reference =
  match env.reference reference with
  | Some (symbol, addend) ->
      let s = Asm_core.Expr.Symbol symbol in
      if Int64.equal addend 0L then s
      else
        Asm_core.Expr.Binary
          ( Asm_core.Expr.Add,
            s,
            Asm_core.Expr.Const (Foundation.Bigint.of_int64 addend) )
  | None -> refuse env (R.Form "an unrecorded reference")

let bitfield ~signed ~rd ~rn ~lsb ~width =
  ins
    (if signed then Op.Sbfx else Op.Ubfx)
    [ rd; rn; imm_int lsb; imm_int width ]

(* x18, which the table binding's entry sets to the caller's table. *)
let table_base = { Aarch64.Reg.num = 18; width = 64; is_sp = false }

(* [value] in [reg] by a move and as many keeps as it has nonzero halfwords. *)
let materialize reg value =
  let quarter k =
    Int64.logand (Int64.shift_right_logical value (16 * k)) 0xFFFFL
  in
  let shift k = O.Shift { Aarch64.Shift.kind = "lsl"; amount = 16 * k } in
  ins Op.Movz [ O.Reg reg; imm (quarter 0) ]
  :: List.filter_map
       (fun k ->
         if Int64.equal (quarter k) 0L then None
         else Some (ins Op.Movk [ O.Reg reg; imm (quarter k); shift k ]))
       [ 1; 2; 3 ]

let ( ++ ) = List.append

let instructions env (op : A.t) ~(uses : Loc.t list) ~(defs : Loc.t list) :
    I.t list =
  let u = nth env "use" uses and d = nth env "def" defs in
  let binary opcode sz =
    [ ins opcode [ g env sz (d 0); g env sz (u 0); g env sz (u 1) ] ]
  in
  let unsupported () = refuse env (R.Form (form_name op)) in
  let mutated m = env.mutation = Some m in
  match op with
  | A.Add (sz, _, _) -> binary Op.Add sz
  | A.Add_imm (sz, _, k) ->
      [ ins Op.Add [ g env sz (d 0); g env sz (u 0); imm k ] ]
  | A.Add_lo12 (_, _) when mutated Mutation.Dropped_lo12 ->
      [ ins Op.Add [ g env A.Sz.X (d 0); g env A.Sz.X (u 0); imm 0L ] ]
  | A.Add_lo12 (_, view)
    when match
           env.reference
             (Mir_target.Reference.View
                (view, Mir_target.Reference.Form.Page_offset))
         with
         | Some (symbol, _) -> Option.is_some (env.table_slot symbol)
         | None -> false ->
      (* a table-bound region: its base is already in the register, so the
         low twelve bits of the view's offset are all that remain *)
      let _, addend =
        Option.get
          (env.reference
             (Mir_target.Reference.View
                (view, Mir_target.Reference.Form.Page_offset)))
      in
      [
        ins Op.Add
          [
            g env A.Sz.X (d 0);
            g env A.Sz.X (u 0);
            imm (Int64.logand addend 0xFFFL);
          ];
      ]
  | A.Add_lo12 (_, view) ->
      let target =
        address env
          (Mir_target.Reference.View
             (view, Mir_target.Reference.Form.Page_offset))
      in
      [
        ins Op.Add
          [
            g env A.Sz.X (d 0);
            g env A.Sz.X (u 0);
            O.Sym (Asm_core.Expr.Modifier ("lo12", target));
          ];
      ]
  | A.Adrp view
    when match
           env.reference
             (Mir_target.Reference.View (view, Mir_target.Reference.Form.Page))
         with
         | Some (symbol, _) -> Option.is_some (env.table_slot symbol)
         | None -> false ->
      (* a table-bound region: the base from the caller's table, then the
         whole pages of the view's offset *)
      let symbol, addend =
        Option.get
          (env.reference
             (Mir_target.Reference.View (view, Mir_target.Reference.Form.Page)))
      in
      let slot = Option.get (env.table_slot symbol) in
      let rd = gpr env ~width:64 (loc_view env (d 0)) in
      let page = Int64.logand addend (Int64.lognot 0xFFFL) in
      ins Op.Ldr
        [
          O.Reg rd;
          O.Mem
            {
              Aarch64.Mem.base = table_base;
              offset = Aarch64.Disp.Const (Int64.of_int (8 * slot));
              writeback = false;
              pre = true;
            };
        ]
      ::
      (if Int64.equal page 0L then []
       else if Int64.compare page 0xFFF000L <= 0 then
         [ ins Op.Add [ O.Reg rd; O.Reg rd; imm page ] ]
       else
         let scratch = { Aarch64.Reg.num = 16; width = 64; is_sp = false } in
         materialize scratch page
         @ [ ins Op.Add [ O.Reg rd; O.Reg rd; O.Reg scratch ] ])
  | A.Adrp view ->
      let target =
        address env
          (Mir_target.Reference.View (view, Mir_target.Reference.Form.Page))
      in
      [ ins Op.Adrp [ g env A.Sz.X (d 0); O.Sym target ] ]
  | A.Bl { callee; _ } ->
      let target = address env (Mir_target.Reference.Call callee) in
      [ ins Op.Bl [ O.Sym target ] ]
  | A.Cmp (sz, _, _) -> [ ins Op.Cmp [ g env sz (u 0); g env sz (u 1) ] ]
  | A.Cmp_imm (sz, _, k) -> [ ins Op.Cmp [ g env sz (u 0); imm k ] ]
  | A.Csel (sz, c, _, _, _) ->
      [
        ins Op.Csel
          [ g env sz (d 0); g env sz (u 1); g env sz (u 2); sym (cond_name c) ];
      ]
  | A.Cset (c, _) -> [ ins Op.Cset [ g env A.Sz.W (d 0); sym (cond_name c) ] ]
  | A.Ext { signed; from; _ } ->
      [
        bitfield ~signed
          ~rd:(g env A.Sz.W (d 0))
          ~rn:(g env A.Sz.W (u 0))
          ~lsb:0 ~width:(Mir_width.bits from);
      ]
  | A.Fbin (o, fsz, _, _) -> (
      let fr l = f env fsz l in
      match o with
      | A.Fop.Add -> [ ins Op.Fadd [ fr (d 0); fr (u 0); fr (u 1) ] ]
      | A.Fop.Div -> [ ins Op.Fdiv [ fr (d 0); fr (u 0); fr (u 1) ] ]
      | A.Fop.Mul -> [ ins Op.Fmul [ fr (d 0); fr (u 0); fr (u 1) ] ]
      | A.Fop.Sub when mutated Mutation.Commuted_sub ->
          [ ins Op.Fsub [ fr (d 0); fr (u 1); fr (u 0) ] ]
      | A.Fop.Sub -> [ ins Op.Fsub [ fr (d 0); fr (u 0); fr (u 1) ] ]
      | A.Fop.Max -> [ ins Op.Fmax [ fr (d 0); fr (u 0); fr (u 1) ] ])
  | A.Fcmp (fsz, _, _) -> [ ins Op.Fcmp [ f env fsz (u 0); f env fsz (u 1) ] ]
  | A.Fcsel (fsz, c, _, _, _) ->
      [
        ins Op.Fcsel
          [
            f env fsz (d 0); f env fsz (u 1); f env fsz (u 2); sym (cond_name c);
          ];
      ]
  | A.Fcvt (to_, _) ->
      let from = match to_ with A.Fsz.D -> A.Fsz.S | A.Fsz.S -> A.Fsz.D in
      [ ins Op.Fcvt [ f env to_ (d 0); f env from (u 0) ] ]
  | A.Fcvtzs (fsz, _) ->
      [ ins Op.Fcvtzs [ g env A.Sz.X (d 0); f env fsz (u 0) ] ]
  | A.Fmov (fsz, _) -> [ ins Op.Fmov [ f env fsz (d 0); f env fsz (u 0) ] ]
  | A.Fmov_from_gpr (fsz, _) ->
      let sz = match fsz with A.Fsz.D -> A.Sz.X | A.Fsz.S -> A.Sz.W in
      [ ins Op.Fmov [ f env fsz (d 0); g env sz (u 0) ] ]
  | A.Fmadd (fsz, _, _, _) ->
      [
        ins Op.Fmadd
          [ f env fsz (d 0); f env fsz (u 0); f env fsz (u 1); f env fsz (u 2) ];
      ]
  | A.Fmov_to_gpr (fsz, _) ->
      let sz = match fsz with A.Fsz.D -> A.Sz.X | A.Fsz.S -> A.Sz.W in
      [ ins Op.Fmov [ g env sz (d 0); f env fsz (u 0) ] ]
  | A.Funary (u_, fsz, _) ->
      let opcode =
        match u_ with
        | A.Funary.Fneg -> Op.Fneg
        | A.Funary.Frintz -> Op.Frintz
        | A.Funary.Fsqrt -> Op.Fsqrt
      in
      [ ins opcode [ f env fsz (d 0); f env fsz (u 0) ] ]
  | A.Ldr (m, _, k) ->
      [ access env ~load:true m ~rt:(d 0) ~base:(u 0) ~offset:k ]
  | A.Logic (o, sz, _, _) ->
      binary
        (match o with
        | A.Logic.And -> Op.And
        | A.Logic.Eor -> Op.Eor
        | A.Logic.Orr -> Op.Orr)
        sz
  | A.Logic_imm (o, sz, _, k) ->
      [
        ins
          (match o with
          | A.Logic.And -> Op.And
          | A.Logic.Eor -> Op.Eor
          | A.Logic.Orr -> Op.Orr)
          [ g env sz (d 0); g env sz (u 0); imm k ];
      ]
  | A.Mov (sz, _) -> [ ins Op.Mov [ g env sz (d 0); g env sz (u 0) ] ]
  | A.Movk (sz, _, v, sh) ->
      [
        ins Op.Movk
          [
            g env sz (d 0);
            imm_int v;
            O.Shift { Aarch64.Shift.kind = "lsl"; amount = sh };
          ];
      ]
  | A.Movn (sz, v, sh) ->
      [
        ins Op.Movn
          [
            g env sz (d 0);
            imm_int v;
            O.Shift { Aarch64.Shift.kind = "lsl"; amount = sh };
          ];
      ]
  | A.Movz (ty, v, sh) ->
      let sz =
        match ty with
        | Mir_type.Int Mir_width.W64 | Mir_type.Ptr -> A.Sz.X
        | _ -> A.Sz.W
      in
      [
        ins Op.Movz
          [
            g env sz (d 0);
            imm_int v;
            O.Shift { Aarch64.Shift.kind = "lsl"; amount = sh };
          ];
      ]
  | A.Mrs_fpcr _ -> [ ins Op.Mrs [ g env A.Sz.X (d 0); sym "fpcr" ] ]
  | A.Msr_fpcr _ -> [ ins Op.Msr [ sym "fpcr"; g env A.Sz.X (u 0) ] ]
  | A.Msub (sz, _, _, _) ->
      [
        ins Op.Msub
          [ g env sz (d 0); g env sz (u 0); g env sz (u 1); g env sz (u 2) ];
      ]
  | A.Mul (sz, _, _) -> binary Op.Mul sz
  | A.Scvtf (fsz, _) -> [ ins Op.Scvtf [ f env fsz (d 0); g env A.Sz.X (u 0) ] ]
  | A.Sdiv (sz, _, _) -> binary Op.Sdiv sz
  | A.Shift_imm (o, sz, _, k) ->
      let bits = width_bits sz in
      let rd = g env sz (d 0) and rn = g env sz (u 0) in
      [
        (match o with
        | A.Shift.Lsl -> ins Op.Ubfiz [ rd; rn; imm_int k; imm_int (bits - k) ]
        | A.Shift.Lsr -> ins Op.Ubfx [ rd; rn; imm_int k; imm_int (bits - k) ]
        | A.Shift.Asr -> ins Op.Sbfx [ rd; rn; imm_int k; imm_int (bits - k) ]);
      ]
  | A.Str (m, _, k, _) ->
      [ access env ~load:false m ~rt:(u 1) ~base:(u 0) ~offset:k ]
  | A.Sub (sz, _, _) when mutated Mutation.Commuted_sub ->
      [ ins Op.Sub [ g env sz (d 0); g env sz (u 1); g env sz (u 0) ] ]
  | A.Sub (sz, _, _) -> binary Op.Sub sz
  | A.Sub_imm (sz, _, k) ->
      [ ins Op.Sub [ g env sz (d 0); g env sz (u 0); imm k ] ]
  | A.Sxtw _ -> [ ins Op.Sxtw [ g env A.Sz.X (d 0); g env A.Sz.W (u 0) ] ]
  | A.Trunc (w, _) ->
      [
        bitfield ~signed:false
          ~rd:(g env A.Sz.W (d 0))
          ~rn:(g env A.Sz.W (u 0))
          ~lsb:0 ~width:(Mir_width.bits w);
      ]
  | A.Uxtw _ | A.Wtrunc _ ->
      [ ins Op.Mov [ g env A.Sz.W (d 0); g env A.Sz.W (u 0) ] ]
  | A.Dup_elem _ | A.Dup_half _ | A.Dup_lane _ | A.Fcvtl _ | A.Fcvtn _
  | A.Ins_half _ | A.Ins_lane _ | A.Ld1_lane _ | A.Ld1r _ | A.Ldr_vec _
  | A.St1_lane _ | A.Str_vec _ | A.Vfbin _ | A.Vfmla _ | A.Vfunary _ | A.Vmov _
  | A.Vwiden _ ->
      unsupported ()

(* {1 Allocation-added forms} *)

(* A register-to-register or frame transfer. *)
let transfer ?mutation ?(save = false) env ~(dst : Loc.t) ~(src : Loc.t) :
    I.t list =
  match (dst, src) with
  | Loc.Reg d, Loc.Reg s -> (
      match (d.Mir_target.View.bank, s.Mir_target.View.bank) with
      | Mir_target.Bank.Gpr, Mir_target.Bank.Gpr ->
          let width = min d.Mir_target.View.bits s.Mir_target.View.bits in
          [ ins Op.Mov [ O.Reg (gpr env ~width d); O.Reg (gpr env ~width s) ] ]
      | Mir_target.Bank.Fpr, Mir_target.Bank.Fpr
        when d.Mir_target.View.bits <= 64 && s.Mir_target.View.bits <= 64 ->
          let double = d.Mir_target.View.bits = 64 in
          [
            ins Op.Fmov
              [ O.Freg (fpr env ~double d); O.Freg (fpr env ~double s) ];
          ]
      | bank, _ ->
          refuse env
            (R.Frame_access
               { bytes = Int64.of_int d.Mir_target.View.bits; bank }))
  | Loc.Mem { base; offset; bytes }, Loc.Reg r
  | Loc.Reg r, Loc.Mem { base; offset; bytes } -> (
      let load = match dst with Loc.Reg _ -> true | _ -> false in
      let m =
        match (r.Mir_target.View.bank, bytes) with
        | Mir_target.Bank.Gpr, 8L ->
            Some
              (if mutation = Some Mutation.Narrow_spill && not save then A.Msz.W
               else A.Msz.X)
        | Mir_target.Bank.Gpr, 4L -> Some A.Msz.W
        | Mir_target.Bank.Fpr, 8L -> Some A.Msz.D
        | Mir_target.Bank.Fpr, 4L -> Some A.Msz.S
        | _ -> None
      in
      match m with
      | Some m ->
          [ access env ~load m ~rt:(Loc.Reg r) ~base:(Loc.Reg base) ~offset ]
      | None ->
          refuse env (R.Frame_access { bytes; bank = r.Mir_target.View.bank }))
  | Loc.Slot _, _ | _, Loc.Slot _ -> refuse env (R.Location "a frame slot")
  | Loc.Mem _, Loc.Mem _ -> refuse env (R.Location "memory to memory")

(* The stack pointer moved by [delta] bytes. *)
let stack_step env delta : I.t list =
  let sp = O.Reg { Aarch64.Reg.num = 31; width = 64; is_sp = true } in
  ignore env;
  if Int64.compare delta 0L < 0 then
    [ ins Op.Sub [ sp; sp; imm (Int64.neg delta) ] ]
  else [ ins Op.Add [ sp; sp; imm delta ] ]

(* {1 Terminators} *)

let invert (c : A.Cond.t) : A.Cond.t =
  match c with
  | A.Cond.Eq -> A.Cond.Ne
  | A.Cond.Ne -> A.Cond.Eq
  | A.Cond.Ge -> A.Cond.Lt
  | A.Cond.Lt -> A.Cond.Ge
  | A.Cond.Gt -> A.Cond.Le
  | A.Cond.Le -> A.Cond.Gt
  | A.Cond.Hi -> A.Cond.Ls
  | A.Cond.Ls -> A.Cond.Hi
  | A.Cond.Hs -> A.Cond.Lo
  | A.Cond.Lo -> A.Cond.Hs
  | A.Cond.Mi -> A.Cond.Pl
  | A.Cond.Pl -> A.Cond.Mi
  | A.Cond.Vc -> A.Cond.Vs
  | A.Cond.Vs -> A.Cond.Vc

let rivet_cond (c : A.Cond.t) =
  match Aarch64.Cond.of_name (A.Cond.name c) with
  | Some c -> c
  | None -> invalid_arg "Rivet_a64_form.rivet_cond"

(* The conditional branch to [label] taken when [test] holds, optionally
   inverted. *)
let branch env (test : A.test) ~(uses : Loc.t list) ~label ~inverted : I.t =
  match test with
  | A.B_cond (c, _) ->
      let c = if inverted then invert c else c in
      ins (Op.Bcond (rivet_cond c)) [ sym label ]
  | A.Cbz (sz, _) ->
      ins
        (if inverted then Op.Cbnz else Op.Cbz)
        [ g env sz (nth env "use" uses 0); sym label ]
  | A.Cbnz (sz, _) ->
      ins
        (if inverted then Op.Cbz else Op.Cbnz)
        [ g env sz (nth env "use" uses 0); sym label ]

let jump label = ins Op.B [ sym label ]
let ret = ins Op.Ret []
