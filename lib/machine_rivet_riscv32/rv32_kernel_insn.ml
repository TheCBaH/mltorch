(* The instructions the peak kernels use, as data. Operand fields are named for
   their role and documented in assembly operand order, which differs between
   forms ([vadd.vv vd, vs2, vs1] against [vmacc.vv vd, vs1, vs2]); each
   register class has its own type, so an integer scalar cannot be passed to a
   floating-point instruction. *)

module Gpr = Rv32_kernel_reg.Gpr
module Fpr = Rv32_kernel_reg.Fpr
module Mtreg = Rv32_kernel_reg.Mtreg
module Vreg = Rv32_kernel_reg.Vreg
module Imm = Rv32_kernel_imm

(* Coral NPU's Zvt tile multiplies, [op acc, a, b]: the tile [acc] of [tm] rows
   and [vl] columns accumulates [a]^T [b] over [tk] slices of A and B. [Vtfmm]
   is FP32 (BF16 with the [Vtfmm_alt] form), [Vtmms] signed INT8 A, [Vtmmu]
   unsigned. *)
module Tile_op = struct
  type t = Vtfmm | Vtfmm_alt | Vtmms | Vtmmu

  let all = [ Vtfmm; Vtfmm_alt; Vtmms; Vtmmu ]

  let mnemonic = function
    | Vtfmm -> "vtfmm.tvv"
    | Vtfmm_alt -> "vtfmm.alt.tvv"
    | Vtmms -> "vtmms.tvv"
    | Vtmmu -> "vtmmu.tvv"

  let of_mnemonic m = List.find_opt (fun op -> String.equal (mnemonic op) m) all
end

(* The Rivet vendor feature that holds the Zvt forms. *)
let zvt_feature = "xcoralzvt"

(* Vector-vector operations, [op.vv dst, a, b] with [a] and [b] in assembly
   order. *)
module Vv_op = struct
  type t = Vadd | Vfadd | Vfmacc | Vfmul | Vfwmaccbf16 | Vmacc | Vmul | Vwmacc

  let all = [ Vadd; Vfadd; Vfmacc; Vfmul; Vfwmaccbf16; Vmacc; Vmul; Vwmacc ]

  let mnemonic = function
    | Vadd -> "vadd.vv"
    | Vfadd -> "vfadd.vv"
    | Vfmacc -> "vfmacc.vv"
    | Vfmul -> "vfmul.vv"
    | Vfwmaccbf16 -> "vfwmaccbf16.vv"
    | Vmacc -> "vmacc.vv"
    | Vmul -> "vmul.vv"
    | Vwmacc -> "vwmacc.vv"

  let of_mnemonic m = List.find_opt (fun op -> String.equal (mnemonic op) m) all
end

(* The CSRs a kernel reads. Alphabetical. *)
module Csr = struct
  type t = Mcycle

  let number = function Mcycle -> 0xB00
  let pp ppf = function Mcycle -> Fmt.string ppf "mcycle"
end

module Constant_range = struct
  (* A word that fits neither a signed nor an unsigned 32 bits, as written. *)
  type t = { value : int }

  let pp ppf { value } = Fmt.pf ppf "%d is not a 32-bit constant" value
end

(* A register of any class, as the effects of an instruction name it. A vector
   register stands for the base of its group: the extent depends on the active
   [vtype], which an instruction does not carry. *)
module Reg_ref = struct
  (* Alphabetical. *)
  type t = Fpr of Fpr.t | Gpr of Gpr.t | Mtreg of Mtreg.t | Vreg of Vreg.t

  let equal a b =
    match (a, b) with
    | Fpr a, Fpr b -> Fpr.equal a b
    | Gpr a, Gpr b -> Gpr.equal a b
    | Mtreg a, Mtreg b -> Mtreg.equal a b
    | Vreg a, Vreg b -> Vreg.equal a b
    | (Fpr _ | Gpr _ | Mtreg _ | Vreg _), _ -> false

  let pp ppf = function
    | Fpr r -> Fpr.pp ppf r
    | Gpr r -> Gpr.pp ppf r
    | Mtreg r -> Mtreg.pp ppf r
    | Vreg r -> Vreg.pp ppf r
end

type t =
  | Add of { dst : Gpr.t; lhs : Gpr.t; rhs : Gpr.t }
  | Addi of { dst : Gpr.t; src : Gpr.t; imm : Imm.Imm12.t }
  | Andi of { dst : Gpr.t; src : Gpr.t; imm : Imm.Imm12.t }
  | Beqz of { src : Gpr.t; target : Rv32_kernel_label.t }
  | Bnez of { src : Gpr.t; target : Rv32_kernel_label.t }
  | Csrr of { dst : Gpr.t; csr : Csr.t }
  | Fence
      (** [fence rw, rw]: every earlier load and store before any later one,
          device accesses included. *)
  | Flh of { dst : Fpr.t; base : Gpr.t; offset : Imm.Imm12.t }
  | J of { target : Rv32_kernel_label.t }
      (** [jal zero]: a reach of 1 MiB, where a branch has 4 KiB. *)
  | Flw of { dst : Fpr.t; base : Gpr.t; offset : Imm.Imm12.t }
  | Li of { dst : Gpr.t; imm : Imm.Li.t }
  | Lui of { dst : Gpr.t; imm : Imm.U20.t }
  | Lw of { dst : Gpr.t; base : Gpr.t; offset : Imm.Imm12.t }
  | Msetmtype of { dst : Gpr.t; mtype : Gpr.t; vtype : Gpr.t }
      (** Zvt: [mtype] and, as [vsetvl] would, [vtype] from registers; [vl]
          becomes 0. *)
  | Msettn of { dst : Gpr.t; avl : Gpr.t }
      (** Zvt: [vl], the tile's columns, from [avl], clamped as [vsetvl]. *)
  | Mv of { dst : Gpr.t; src : Gpr.t }
  | Ret
  | Srli of { dst : Gpr.t; src : Gpr.t; shamt : Imm.Shamt.t }
  | Sub of { dst : Gpr.t; lhs : Gpr.t; rhs : Gpr.t }
  | Sw of { src : Gpr.t; base : Gpr.t; offset : Imm.Imm12.t }
  | Vfmacc_vf of { acc : Vreg.t; scalar : Fpr.t; src : Vreg.t }
  | Vfwmaccbf16_vf of { acc : Vreg.t; scalar : Fpr.t; src : Vreg.t }
  | Vle of { eew : Rv32_kernel_vtype.Sew.t; dst : Vreg.t; base : Gpr.t }
  | Vmacc_vx of { acc : Vreg.t; scalar : Gpr.t; src : Vreg.t }
  | Vmv_v_i of { dst : Vreg.t; imm : Imm.Simm5.t }
  | Vmv_v_x of { dst : Vreg.t; src : Gpr.t }
  | Vse of { eew : Rv32_kernel_vtype.Sew.t; src : Vreg.t; base : Gpr.t }
  | Vsetvli of { avl : Rv32_kernel_vtype.Avl.t; vtype : Rv32_kernel_vtype.t }
  | Vtmm of { op : Tile_op.t; acc : Mtreg.t; a : Vreg.t; b : Vreg.t }
  | Vtse of { eew : Rv32_kernel_vtype.Sew.t; slice : Gpr.t; base : Gpr.t }
      (** Zvt: one row or column of a tile, chosen by [slice] (tile, direction
          and index), stored at [base]. *)
  | Vtzero of { dst : Mtreg.t }
  | Vv of { op : Vv_op.t; dst : Vreg.t; a : Vreg.t; b : Vreg.t }
  | Vwmacc_vx of { acc : Vreg.t; scalar : Gpr.t; src : Vreg.t }

(* The Rivet vendor feature [i] needs beyond the standard ones, if any. *)
let extension = function
  | Msetmtype _ | Msettn _ | Vtmm _ | Vtse _ | Vtzero _ -> Some zvt_feature
  | Add _ | Addi _ | Andi _ | Beqz _ | Bnez _ | Csrr _ | Fence | Flh _ | Flw _
  | J _ | Li _ | Lui _ | Lw _ | Mv _ | Ret | Srli _ | Sub _ | Sw _ | Vfmacc_vf _
  | Vfwmaccbf16_vf _ | Vle _ | Vmacc_vx _ | Vmv_v_i _ | Vmv_v_x _ | Vse _
  | Vsetvli _ | Vv _ | Vwmacc_vx _ ->
      None

(* The one exhaustive spelling of each mnemonic; the printer and the lowering
   both read it. *)
let mnemonic = function
  | Add _ -> "add"
  | Addi _ -> "addi"
  | Andi _ -> "andi"
  | Beqz _ -> "beqz"
  | Bnez _ -> "bnez"
  | Csrr _ -> "csrr"
  | Fence -> "fence"
  | Flh _ -> "flh"
  | Flw _ -> "flw"
  | J _ -> "j"
  | Li _ -> "li"
  | Lui _ -> "lui"
  | Lw _ -> "lw"
  | Msetmtype _ -> "msetmtype"
  | Msettn _ -> "msettn"
  | Mv _ -> "mv"
  | Ret -> "ret"
  | Srli _ -> "srli"
  | Sub _ -> "sub"
  | Sw _ -> "sw"
  | Vfmacc_vf _ -> "vfmacc.vf"
  | Vfwmaccbf16_vf _ -> "vfwmaccbf16.vf"
  | Vle { eew; _ } -> Fmt.str "vle%d.v" (Rv32_kernel_vtype.Sew.bits eew)
  | Vmacc_vx _ -> "vmacc.vx"
  | Vmv_v_i _ -> "vmv.v.i"
  | Vmv_v_x _ -> "vmv.v.x"
  | Vse { eew; _ } -> Fmt.str "vse%d.v" (Rv32_kernel_vtype.Sew.bits eew)
  | Vsetvli _ -> "vsetvli"
  | Vtmm { op; _ } -> Tile_op.mnemonic op
  | Vtse { eew; _ } -> Fmt.str "vtse%d" (Rv32_kernel_vtype.Sew.bits eew)
  | Vtzero _ -> "vtzero"
  | Vv { op; _ } -> Vv_op.mnemonic op
  | Vwmacc_vx _ -> "vwmacc.vx"

let pp ppf i =
  let m = mnemonic i in
  let pp_mem ppf (offset, base) =
    Fmt.pf ppf "%a(%a)" Imm.Imm12.pp offset Gpr.pp base
  in
  match i with
  | Add { dst; lhs; rhs } | Sub { dst; lhs; rhs } ->
      Fmt.pf ppf "%s %a, %a, %a" m Gpr.pp dst Gpr.pp lhs Gpr.pp rhs
  | Addi { dst; src; imm } | Andi { dst; src; imm } ->
      Fmt.pf ppf "%s %a, %a, %a" m Gpr.pp dst Gpr.pp src Imm.Imm12.pp imm
  | Beqz { src; target } | Bnez { src; target } ->
      Fmt.pf ppf "%s %a, %a" m Gpr.pp src Rv32_kernel_label.pp target
  | Csrr { dst; csr } -> Fmt.pf ppf "%s %a, %a" m Gpr.pp dst Csr.pp csr
  | J { target } -> Fmt.pf ppf "%s %a" m Rv32_kernel_label.pp target
  | Fence -> Fmt.pf ppf "%s rw, rw" m
  | Flh { dst; base; offset } | Flw { dst; base; offset } ->
      Fmt.pf ppf "%s %a, %a" m Fpr.pp dst pp_mem (offset, base)
  | Li { dst; imm } -> Fmt.pf ppf "%s %a, %a" m Gpr.pp dst Imm.Li.pp imm
  | Lui { dst; imm } -> Fmt.pf ppf "%s %a, %a" m Gpr.pp dst Imm.U20.pp imm
  | Lw { dst; base; offset } ->
      Fmt.pf ppf "%s %a, %a" m Gpr.pp dst pp_mem (offset, base)
  | Msetmtype { dst; mtype; vtype } ->
      Fmt.pf ppf "%s %a, %a, %a" m Gpr.pp dst Gpr.pp mtype Gpr.pp vtype
  | Msettn { dst; avl } -> Fmt.pf ppf "%s %a, %a" m Gpr.pp dst Gpr.pp avl
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
  | Vtmm { acc; a; b; _ } ->
      Fmt.pf ppf "%s %a, %a, %a" m Mtreg.pp acc Vreg.pp a Vreg.pp b
  | Vtse { slice; base; _ } ->
      Fmt.pf ppf "%s %a, (%a)" m Gpr.pp slice Gpr.pp base
  | Vtzero { dst } -> Fmt.pf ppf "%s %a" m Mtreg.pp dst
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
  | Addi { imm; _ } | Andi { imm; _ } -> check (module Imm.Imm12) imm
  | Flh { offset; _ } | Flw { offset; _ } | Lw { offset; _ } | Sw { offset; _ }
    ->
      check (module Imm.Imm12) offset
  | Li { imm; _ } -> check (module Imm.Li) imm
  | Lui { imm; _ } -> check (module Imm.U20) imm
  | Srli { shamt; _ } -> check (module Imm.Shamt) shamt
  | Vmv_v_i { imm; _ } -> check (module Imm.Simm5) imm
  | Add _ | Beqz _ | Bnez _ | Csrr _ | Fence | J _ | Msetmtype _ | Msettn _
  | Mv _ | Ret | Sub _ | Vfmacc_vf _ | Vfwmaccbf16_vf _ | Vle _ | Vmacc_vx _
  | Vmv_v_x _ | Vse _ | Vsetvli _ | Vtmm _ | Vtse _ | Vtzero _ | Vv _
  | Vwmacc_vx _ ->
      []

(* The labels [i] branches to. *)
let targets = function
  | Beqz { target; _ } | Bnez { target; _ } | J { target } -> [ target ]
  | Add _ | Addi _ | Andi _ | Csrr _ | Fence | Flh _ | Flw _ | Li _ | Lui _
  | Lw _ | Msetmtype _ | Msettn _ | Mv _ | Ret | Srli _ | Sub _ | Sw _
  | Vfmacc_vf _ | Vfwmaccbf16_vf _ | Vle _ | Vmacc_vx _ | Vmv_v_i _ | Vmv_v_x _
  | Vse _ | Vsetvli _ | Vtmm _ | Vtse _ | Vtzero _ | Vv _ | Vwmacc_vx _ ->
      []

(* The registers [i] reads, in operand order. [zero] is a constant, not a
   dependency, so it is left out. An accumulating instruction reads its
   accumulator. *)
let reads i =
  let gprs rs =
    List.filter_map
      (fun r -> if Gpr.equal r Gpr.zero then None else Some (Reg_ref.Gpr r))
      rs
  in
  let vregs rs = List.map (fun r -> Reg_ref.Vreg r) rs in
  match i with
  | Add { lhs; rhs; _ } | Sub { lhs; rhs; _ } -> gprs [ lhs; rhs ]
  | Addi { src; _ } | Andi { src; _ } | Mv { src; _ } | Srli { src; _ } ->
      gprs [ src ]
  | Beqz { src; _ } | Bnez { src; _ } -> gprs [ src ]
  | Csrr _ | Fence | J _ -> []
  | Flh { base; _ } | Flw { base; _ } | Lw { base; _ } | Vle { base; _ } ->
      gprs [ base ]
  | Li _ | Lui _ | Ret | Vmv_v_i _ | Vtzero _ -> []
  | Msetmtype { mtype; vtype; _ } -> gprs [ mtype; vtype ]
  | Msettn { avl; _ } -> gprs [ avl ]
  | Sw { src; base; _ } -> gprs [ src; base ]
  | Vfmacc_vf { acc; scalar; src } | Vfwmaccbf16_vf { acc; scalar; src } ->
      vregs [ acc ] @ [ Reg_ref.Fpr scalar ] @ vregs [ src ]
  | Vmacc_vx { acc; scalar; src } | Vwmacc_vx { acc; scalar; src } ->
      vregs [ acc ] @ gprs [ scalar ] @ vregs [ src ]
  | Vmv_v_x { src; _ } -> gprs [ src ]
  | Vse { src; base; _ } -> vregs [ src ] @ gprs [ base ]
  | Vsetvli { avl = Reg r; _ } -> gprs [ r ]
  | Vsetvli { avl = Vlmax _; _ } -> []
  | Vtmm { acc; a; b; _ } -> [ Reg_ref.Mtreg acc ] @ vregs [ a; b ]
  | Vtse { slice; base; _ } -> gprs [ slice; base ]
  | Vv { op; dst; a; b } -> (
      match op with
      | Vfmacc | Vfwmaccbf16 | Vmacc | Vwmacc -> vregs [ dst; a; b ]
      | Vadd | Vfadd | Vfmul | Vmul -> vregs [ a; b ])

(* The registers [i] writes. [zero] is never written. [vsetvli] sets [vl], which
   is not a register, and writes a register only for a VLMAX request. *)
let writes i =
  let gpr r = if Gpr.equal r Gpr.zero then [] else [ Reg_ref.Gpr r ] in
  match i with
  | Add { dst; _ }
  | Addi { dst; _ }
  | Andi { dst; _ }
  | Csrr { dst; _ }
  | Li { dst; _ }
  | Lui { dst; _ }
  | Lw { dst; _ }
  | Msetmtype { dst; _ }
  | Msettn { dst; _ }
  | Mv { dst; _ }
  | Srli { dst; _ }
  | Sub { dst; _ } ->
      gpr dst
  | Flh { dst; _ } | Flw { dst; _ } -> [ Reg_ref.Fpr dst ]
  | Beqz _ | Bnez _ | Fence | J _ | Ret | Sw _ | Vse _ | Vtse _ -> []
  | Vtmm { acc = dst; _ } | Vtzero { dst } -> [ Reg_ref.Mtreg dst ]
  | Vfmacc_vf { acc; _ }
  | Vfwmaccbf16_vf { acc; _ }
  | Vmacc_vx { acc; _ }
  | Vwmacc_vx { acc; _ } ->
      [ Reg_ref.Vreg acc ]
  | Vle { dst; _ } | Vmv_v_i { dst; _ } | Vmv_v_x { dst; _ } | Vv { dst; _ } ->
      [ Reg_ref.Vreg dst ]
  | Vsetvli { avl = Reg _; _ } -> []
  | Vsetvli { avl = Vlmax { vl }; _ } -> gpr vl
