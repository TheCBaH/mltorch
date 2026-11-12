(* The admitted AArch64 selected forms: the variant of [A64_form], its typing
   rule from [A64_typing], and its printing. *)

open Machine_ir
include A64_form
include A64_typing

let pp_op pv fmt op =
  let vs = Fmt.(list ~sep:(any ", ") pv) in
  match op with
  | Add (sz, a, b) -> Fmt.pf fmt "add.%s %a" (Sz.name sz) vs [ a; b ]
  | Add_imm (sz, a, k) -> Fmt.pf fmt "add.%s %a, #%Ld" (Sz.name sz) pv a k
  | Add_lo12 (a, view) ->
      Fmt.pf fmt "add.x %a, #:lo12:%a" pv a Mir_id.View.pp view
  | Adrp view -> Fmt.pf fmt "adrp %a" Mir_id.View.pp view
  | Bl { callee; args; _ } ->
      Fmt.pf fmt "bl %a(%a)" Mir_op.Callee.pp callee vs args
  | Cmp (sz, a, b) -> Fmt.pf fmt "cmp.%s %a" (Sz.name sz) vs [ a; b ]
  | Cmp_imm (sz, a, k) -> Fmt.pf fmt "cmp.%s %a, #%Ld" (Sz.name sz) pv a k
  | Csel (sz, c, f, a, b) ->
      Fmt.pf fmt "csel.%s.%s %a" (Sz.name sz) (Cond.name c) vs [ f; a; b ]
  | Cset (c, f) -> Fmt.pf fmt "cset.%s %a" (Cond.name c) pv f
  | Dup_elem (arr, a) -> Fmt.pf fmt "dup.%s %a" (Arr.name arr) pv a
  | Dup_half (k, a) -> Fmt.pf fmt "dup.d %a.d[%d]" pv a k
  | Dup_mask (arr, a) -> Fmt.pf fmt "dup.%s.gpr %a" (Arr.name arr) pv a
  | Dup_lane (fsz, k, a) ->
      Fmt.pf fmt "dup.%s %a.%s[%d]" (Fsz.name fsz) pv a (Fsz.name fsz) k
  | Dup_w a -> Fmt.pf fmt "dup.4s.gpr %a" pv a
  | Ext { signed; from; src } ->
      Fmt.pf fmt "%sxt%s %a"
        (if signed then "s" else "u")
        (match from with Mir_width.W8 -> "b" | _ -> "h")
        pv src
  | Fbin (o, fsz, a, b) ->
      Fmt.pf fmt "%s.%s %a" (Fop.name o) (Fsz.name fsz) vs [ a; b ]
  | Fcmp (fsz, a, b) -> Fmt.pf fmt "fcmp.%s %a" (Fsz.name fsz) vs [ a; b ]
  | Fcsel (fsz, c, f, a, b) ->
      Fmt.pf fmt "fcsel.%s.%s %a" (Fsz.name fsz) (Cond.name c) vs [ f; a; b ]
  | Fcvt (fsz, a) -> Fmt.pf fmt "fcvt.%s %a" (Fsz.name fsz) pv a
  | Fcvtl a -> Fmt.pf fmt "fcvtl.2d %a" pv a
  | Fcvtn a -> Fmt.pf fmt "fcvtn.2s %a" pv a
  | Fcvtzs (fsz, a) -> Fmt.pf fmt "fcvtzs.x.%s %a" (Fsz.name fsz) pv a
  | Fmadd (fsz, a, b, c) ->
      Fmt.pf fmt "fmadd.%s %a" (Fsz.name fsz) vs [ a; b; c ]
  | Fmov (fsz, a) -> Fmt.pf fmt "fmov.%s %a" (Fsz.name fsz) pv a
  | Fmov_from_gpr (fsz, a) -> Fmt.pf fmt "fmov.%s.gpr %a" (Fsz.name fsz) pv a
  | Fmov_to_gpr (fsz, a) -> Fmt.pf fmt "fmov.gpr.%s %a" (Fsz.name fsz) pv a
  | Funary (u, fsz, a) ->
      Fmt.pf fmt "%s.%s %a" (Funary.name u) (Fsz.name fsz) pv a
  | Ins_half (a, b) -> Fmt.pf fmt "ins %a.d[1], %a.d[0]" pv a pv b
  | Ins_lane (fsz, k, a, x) ->
      Fmt.pf fmt "ins %a.%s[%d], %a" pv a (Fsz.name fsz) k pv x
  | Ld1_lane (fsz, k, a, base) ->
      Fmt.pf fmt "ld1 %a.%s[%d], [%a]" pv a (Fsz.name fsz) k pv base
  | Ld1r (arr, base) -> Fmt.pf fmt "ld1r.%s [%a]" (Arr.name arr) pv base
  | Ldr (m, base, k) -> Fmt.pf fmt "ldr.%s [%a, #%Ld]" (Msz.name m) pv base k
  | Ldr_idx (m, base, i) ->
      Fmt.pf fmt "ldr.%s [%a, %a, sxtw]" (Msz.name m) pv base pv i
  | Ldr_ivec (t, base, k) ->
      Fmt.pf fmt "ldr.q %a [%a, #%Ld]" Mir_type.pp t pv base k
  | Ldr_vec (arr, base, k) ->
      Fmt.pf fmt "ldr.%s [%a, #%Ld]" (Arr.name arr) pv base k
  | Logic (o, sz, a, b) ->
      Fmt.pf fmt "%s.%s %a" (Logic.name o) (Sz.name sz) vs [ a; b ]
  | Logic_imm (o, sz, a, k) ->
      Fmt.pf fmt "%s.%s %a, #0x%Lx" (Logic.name o) (Sz.name sz) pv a k
  | Mov (sz, a) -> Fmt.pf fmt "mov.%s %a" (Sz.name sz) pv a
  | Mrs_fpcr a -> Fmt.pf fmt "mrs %a, fpcr" pv a
  | Msr_fpcr a -> Fmt.pf fmt "msr fpcr, %a" pv a
  | Movk (sz, a, imm, sh) ->
      Fmt.pf fmt "movk.%s %a, #0x%x, lsl %d" (Sz.name sz) pv a imm sh
  | Movn (sz, imm, sh) -> Fmt.pf fmt "movn.%s #0x%x, lsl %d" (Sz.name sz) imm sh
  | Movz (t, imm, sh) -> Fmt.pf fmt "movz.%a #0x%x, lsl %d" Mir_type.pp t imm sh
  | Msub (sz, a, b, c) -> Fmt.pf fmt "msub.%s %a" (Sz.name sz) vs [ a; b; c ]
  | Mul (sz, a, b) -> Fmt.pf fmt "mul.%s %a" (Sz.name sz) vs [ a; b ]
  | Scvtf (fsz, a) -> Fmt.pf fmt "scvtf.%s.x %a" (Fsz.name fsz) pv a
  | Scvtf_vec a -> Fmt.pf fmt "scvtf.4s %a" pv a
  | Sdiv (sz, a, b) -> Fmt.pf fmt "sdiv.%s %a" (Sz.name sz) vs [ a; b ]
  | Shift_imm (o, sz, a, k) ->
      Fmt.pf fmt "%s.%s %a, #%d" (Shift.name o) (Sz.name sz) pv a k
  | St1_lane (fsz, k, a, base) ->
      Fmt.pf fmt "st1 %a.%s[%d], [%a]" pv a (Fsz.name fsz) k pv base
  | Str (m, base, k, x) ->
      Fmt.pf fmt "str.%s %a, [%a, #%Ld]" (Msz.name m) pv x pv base k
  | Str_idx (m, base, i, x) ->
      Fmt.pf fmt "str.%s %a, [%a, %a, sxtw]" (Msz.name m) pv x pv base pv i
  | Str_ivec (t, base, k, x) ->
      Fmt.pf fmt "str.%a %a, [%a, #%Ld]" Mir_type.pp t pv x pv base k
  | Str_vec (arr, base, k, x) ->
      Fmt.pf fmt "str.%s %a, [%a, #%Ld]" (Arr.name arr) pv x pv base k
  | Sub (sz, a, b) -> Fmt.pf fmt "sub.%s %a" (Sz.name sz) vs [ a; b ]
  | Sub_imm (sz, a, k) -> Fmt.pf fmt "sub.%s %a, #%Ld" (Sz.name sz) pv a k
  | Sxtw a -> Fmt.pf fmt "sxtw %a" pv a
  | Uxtw a -> Fmt.pf fmt "uxtw %a" pv a
  | Uzp1_h (a, b) -> Fmt.pf fmt "uzp1.8h %a" vs [ a; b ]
  | Vadd_w (a, b) -> Fmt.pf fmt "add.4s %a" vs [ a; b ]
  | Trunc (w, a) ->
      Fmt.pf fmt "uxt%s.trunc %a"
        (match w with Mir_width.W8 -> "b" | _ -> "h")
        pv a
  | Vfbin (o, arr, a, b) ->
      Fmt.pf fmt "%s.%s %a" (Fop.name o) (Arr.name arr) vs [ a; b ]
  | Vbit (arr, other, a, m) ->
      Fmt.pf fmt "bit.%s %a" (Arr.name arr) vs [ other; a; m ]
  | Vfcmp (c, arr, a, b) ->
      Fmt.pf fmt "%s.%s %a" (Vcmp.name c) (Arr.name arr) vs [ a; b ]
  | Vlogic (o, a, b) -> Fmt.pf fmt "%s.16b %a" (Logic.name o) vs [ a; b ]
  | Vnot a -> Fmt.pf fmt "not.16b %a" pv a
  | Vfmla (arr, a, b, c) ->
      Fmt.pf fmt "fmla.%s %a" (Arr.name arr) vs [ a; b; c ]
  | Vfunary (u, arr, a) ->
      Fmt.pf fmt "%s.%s %a" (Funary.name u) (Arr.name arr) pv a
  | Vcast (t, a) -> Fmt.pf fmt "mov.16b.%a %a" Mir_type.pp t pv a
  | Vshift (o, k, a) -> Fmt.pf fmt "%s.4s %a, #%d" (Shift.name o) pv a k
  | Xtl_w { signed; from; first; src } ->
      Fmt.pf fmt "%cxtl.4s.%s[%d] %a"
        (if signed then 's' else 'u')
        (Mir_width.name from) first pv src
  | Vmov (arr, a) -> Fmt.pf fmt "mov.%s %a" (Arr.name arr) pv a
  | Vwiden a -> Fmt.pf fmt "fmov.d %a" pv a
  | Wtrunc a -> Fmt.pf fmt "mov.w.x %a" pv a

let pp_test pv fmt = function
  | B_cond (c, f) -> Fmt.pf fmt "b.%s %a" (Cond.name c) pv f
  | Cbnz (sz, a) -> Fmt.pf fmt "cbnz.%s %a" (Sz.name sz) pv a
  | Cbz (sz, a) -> Fmt.pf fmt "cbz.%s %a" (Sz.name sz) pv a

let name = "aarch64"

let source =
  { Mir_target.Source.document = "Arm ARM DDI 0487"; revision = "K.a" }
