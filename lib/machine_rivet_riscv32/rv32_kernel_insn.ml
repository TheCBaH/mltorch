(* The instructions the peak kernels use, as data. Operand fields are named for
   their role and documented in assembly operand order, which differs between
   forms ([vadd.vv vd, vs2, vs1] against [vmacc.vv vd, vs1, vs2]); each
   register class has its own type, so an integer scalar cannot be passed to a
   floating-point instruction. *)

module Gpr = Rv32_kernel_reg.Gpr
module Fpr = Rv32_kernel_reg.Fpr
module Vreg = Rv32_kernel_reg.Vreg
module Imm = Rv32_kernel_imm

(* Vector-vector operations, [op.vv dst, a, b] with [a] and [b] in assembly
   order. *)
module Vv_op = struct
  type t = Vadd | Vfadd | Vfmacc | Vfmul | Vfwmaccbf16 | Vmacc | Vmul | Vwmacc

  let mnemonic = function
    | Vadd -> "vadd.vv"
    | Vfadd -> "vfadd.vv"
    | Vfmacc -> "vfmacc.vv"
    | Vfmul -> "vfmul.vv"
    | Vfwmaccbf16 -> "vfwmaccbf16.vv"
    | Vmacc -> "vmacc.vv"
    | Vmul -> "vmul.vv"
    | Vwmacc -> "vwmacc.vv"
end

module Constant_range = struct
  (* A word that fits neither a signed nor an unsigned 32 bits, as written. *)
  type t = { value : int }

  let pp ppf { value } = Fmt.pf ppf "%d is not a 32-bit constant" value
end

type t =
  | Add of { dst : Gpr.t; lhs : Gpr.t; rhs : Gpr.t }
  | Addi of { dst : Gpr.t; src : Gpr.t; imm : Imm.Imm12.t }
  | Bnez of { src : Gpr.t; target : Rv32_kernel_label.t }
  | Flh of { dst : Fpr.t; base : Gpr.t; offset : Imm.Imm12.t }
  | Flw of { dst : Fpr.t; base : Gpr.t; offset : Imm.Imm12.t }
  | Li of { dst : Gpr.t; imm : Imm.Li.t }
  | Lui of { dst : Gpr.t; imm : Imm.U20.t }
  | Lw of { dst : Gpr.t; base : Gpr.t; offset : Imm.Imm12.t }
  | Mv of { dst : Gpr.t; src : Gpr.t }
  | Ret
  | Srli of { dst : Gpr.t; src : Gpr.t; shamt : Imm.Shamt.t }
  | Sw of { src : Gpr.t; base : Gpr.t; offset : Imm.Imm12.t }
  | Vfmacc_vf of { acc : Vreg.t; scalar : Fpr.t; src : Vreg.t }
  | Vfwmaccbf16_vf of { acc : Vreg.t; scalar : Fpr.t; src : Vreg.t }
  | Vle of { eew : Rv32_kernel_vtype.Sew.t; dst : Vreg.t; base : Gpr.t }
  | Vmacc_vx of { acc : Vreg.t; scalar : Gpr.t; src : Vreg.t }
  | Vmv_v_i of { dst : Vreg.t; imm : Imm.Simm5.t }
  | Vmv_v_x of { dst : Vreg.t; src : Gpr.t }
  | Vse of { eew : Rv32_kernel_vtype.Sew.t; src : Vreg.t; base : Gpr.t }
  | Vsetvli of { avl : Rv32_kernel_vtype.Avl.t; vtype : Rv32_kernel_vtype.t }
  | Vv of { op : Vv_op.t; dst : Vreg.t; a : Vreg.t; b : Vreg.t }
  | Vwmacc_vx of { acc : Vreg.t; scalar : Gpr.t; src : Vreg.t }

(* The one exhaustive spelling of each mnemonic; the printer and the lowering
   both read it. *)
let mnemonic = function
  | Add _ -> "add"
  | Addi _ -> "addi"
  | Bnez _ -> "bnez"
  | Flh _ -> "flh"
  | Flw _ -> "flw"
  | Li _ -> "li"
  | Lui _ -> "lui"
  | Lw _ -> "lw"
  | Mv _ -> "mv"
  | Ret -> "ret"
  | Srli _ -> "srli"
  | Sw _ -> "sw"
  | Vfmacc_vf _ -> "vfmacc.vf"
  | Vfwmaccbf16_vf _ -> "vfwmaccbf16.vf"
  | Vle { eew; _ } -> Fmt.str "vle%d.v" (Rv32_kernel_vtype.Sew.bits eew)
  | Vmacc_vx _ -> "vmacc.vx"
  | Vmv_v_i _ -> "vmv.v.i"
  | Vmv_v_x _ -> "vmv.v.x"
  | Vse { eew; _ } -> Fmt.str "vse%d.v" (Rv32_kernel_vtype.Sew.bits eew)
  | Vsetvli _ -> "vsetvli"
  | Vv { op; _ } -> Vv_op.mnemonic op
  | Vwmacc_vx _ -> "vwmacc.vx"

let pp ppf i =
  let m = mnemonic i in
  let pp_mem ppf (offset, base) =
    Fmt.pf ppf "%a(%a)" Imm.Imm12.pp offset Gpr.pp base
  in
  match i with
  | Add { dst; lhs; rhs } ->
      Fmt.pf ppf "%s %a, %a, %a" m Gpr.pp dst Gpr.pp lhs Gpr.pp rhs
  | Addi { dst; src; imm } ->
      Fmt.pf ppf "%s %a, %a, %a" m Gpr.pp dst Gpr.pp src Imm.Imm12.pp imm
  | Bnez { src; target } ->
      Fmt.pf ppf "%s %a, %a" m Gpr.pp src Rv32_kernel_label.pp target
  | Flh { dst; base; offset } | Flw { dst; base; offset } ->
      Fmt.pf ppf "%s %a, %a" m Fpr.pp dst pp_mem (offset, base)
  | Li { dst; imm } -> Fmt.pf ppf "%s %a, %a" m Gpr.pp dst Imm.Li.pp imm
  | Lui { dst; imm } -> Fmt.pf ppf "%s %a, %a" m Gpr.pp dst Imm.U20.pp imm
  | Lw { dst; base; offset } ->
      Fmt.pf ppf "%s %a, %a" m Gpr.pp dst pp_mem (offset, base)
  | Mv { dst; src } -> Fmt.pf ppf "%s %a, %a" m Gpr.pp dst Gpr.pp src
  | Ret -> Fmt.string ppf m
  | Srli { dst; src; shamt } ->
      Fmt.pf ppf "%s %a, %a, %a" m Gpr.pp dst Gpr.pp src Imm.Shamt.pp shamt
  | Sw { src; base; offset } ->
      Fmt.pf ppf "%s %a, %a" m Gpr.pp src pp_mem (offset, base)
  | Vfmacc_vf { acc; scalar; src } | Vfwmaccbf16_vf { acc; scalar; src } ->
      Fmt.pf ppf "%s %a, %a, %a" m Vreg.pp acc Fpr.pp scalar Vreg.pp src
  | Vle { dst; base; _ } -> Fmt.pf ppf "%s %a, (%a)" m Vreg.pp dst Gpr.pp base
  | Vmacc_vx { acc; scalar; src } | Vwmacc_vx { acc; scalar; src } ->
      Fmt.pf ppf "%s %a, %a, %a" m Vreg.pp acc Gpr.pp scalar Vreg.pp src
  | Vmv_v_i { dst; imm } ->
      Fmt.pf ppf "%s %a, %a" m Vreg.pp dst Imm.Simm5.pp imm
  | Vmv_v_x { dst; src } -> Fmt.pf ppf "%s %a, %a" m Vreg.pp dst Gpr.pp src
  | Vse { src; base; _ } -> Fmt.pf ppf "%s %a, (%a)" m Vreg.pp src Gpr.pp base
  | Vsetvli { avl; vtype } ->
      let rd, rs1 =
        match avl with Reg r -> (Gpr.zero, r) | Vlmax { vl } -> (vl, Gpr.zero)
      in
      Fmt.pf ppf "%s %a, %a, %a" m Gpr.pp rd Gpr.pp rs1 Rv32_kernel_vtype.pp
        vtype
  | Vv { dst; a; b; _ } ->
      Fmt.pf ppf "%s %a, %a, %a" m Vreg.pp dst Vreg.pp a Vreg.pp b

(* [dst = value] for a signed 32-bit [value] or an unsigned 32-bit bit pattern
   (-0x8000_0000 to 0xFFFF_FFFF on the host), in the split [Rv32_asm.load_imm]
   uses, so the bytes are the same: [li] for a signed 12-bit value, else [lui] of
   the high part adjusted for the low part's sign, then [addi]. The range is
   checked here: a masked value could no longer be told from a valid one. *)
let load_imm dst value =
  if value < -0x8000_0000 || value > 0xFFFF_FFFF then
    Err.fail (`Constant_range { Constant_range.value })
  else if value >= -2048 && value < 2048 then
    Ok [ Li { dst; imm = Imm.Li.of_int value } ]
  else
    let lo = (value land 0xFFF lxor 0x800) - 0x800 in
    let hi = ((value - lo) asr 12) land 0xFFFFF in
    Ok
      (Lui { dst; imm = Imm.U20.of_int hi }
      ::
      (if lo = 0 then []
       else [ Addi { dst; src = dst; imm = Imm.Imm12.of_int lo } ]))

(* The immediates of [i] that fall outside their domain, in operand order. *)
let out_of_range i =
  let check (type a) (module M : Imm.S with type t = a) (x : a) =
    if M.in_range x then [] else [ M.range x ]
  in
  match i with
  | Addi { imm; _ } -> check (module Imm.Imm12) imm
  | Flh { offset; _ } | Flw { offset; _ } | Lw { offset; _ } | Sw { offset; _ }
    ->
      check (module Imm.Imm12) offset
  | Li { imm; _ } -> check (module Imm.Li) imm
  | Lui { imm; _ } -> check (module Imm.U20) imm
  | Srli { shamt; _ } -> check (module Imm.Shamt) shamt
  | Vmv_v_i { imm; _ } -> check (module Imm.Simm5) imm
  | Add _ | Bnez _ | Mv _ | Ret | Vfmacc_vf _ | Vfwmaccbf16_vf _ | Vle _
  | Vmacc_vx _ | Vmv_v_x _ | Vse _ | Vsetvli _ | Vv _ | Vwmacc_vx _ ->
      []

(* The labels [i] branches to. *)
let targets = function
  | Bnez { target; _ } -> [ target ]
  | Add _ | Addi _ | Flh _ | Flw _ | Li _ | Lui _ | Lw _ | Mv _ | Ret | Srli _
  | Sw _ | Vfmacc_vf _ | Vfwmaccbf16_vf _ | Vle _ | Vmacc_vx _ | Vmv_v_i _
  | Vmv_v_x _ | Vse _ | Vsetvli _ | Vv _ | Vwmacc_vx _ ->
      []
