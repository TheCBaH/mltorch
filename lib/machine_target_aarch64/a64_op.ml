(* The admitted AArch64 selected forms over virtual values. Each constructor is
   one instruction form of DDI 0487 (revision K.a); a register operand's width
   is the form's ([W] or [X], [S] or [D]). Every consumer matches this type
   exhaustively and alphabetically. Immediates carry only values the form can
   encode; the typing rule refuses any other. *)

open Machine_ir

module Sz = struct
  type t = W | X

  let bits = function W -> 32 | X -> 64
  let name = function W -> "w" | X -> "x"
end

module Fsz = struct
  type t = D | S

  let name = function D -> "d" | S -> "s"
end

(* A memory access size and the register file it moves. *)
module Msz = struct
  type t = B | D | H | S | W | X

  let bytes = function B -> 1L | H -> 2L | S | W -> 4L | D | X -> 8L

  let name = function
    | B -> "b"
    | D -> "d"
    | H -> "h"
    | S -> "s"
    | W -> "w"
    | X -> "x"
end

(* Condition codes over NZCV (bits N=8, Z=4, C=2, V=1). *)
module Cond = struct
  type t = Eq | Ge | Gt | Hi | Hs | Le | Lo | Ls | Lt | Mi | Ne | Pl | Vc | Vs

  let n = 8L
  and z = 4L
  and c = 2L
  and v = 1L

  (* the bits a condition reads *)
  let reads = function
    | Eq | Ne -> z
    | Hs | Lo -> c
    | Mi | Pl -> n
    | Vc | Vs -> v
    | Hi | Ls -> Int64.logor c z
    | Ge | Lt -> Int64.logor n v
    | Gt | Le -> Int64.logor n (Int64.logor v z)

  let holds t nzcv =
    let bit b = not (Int64.equal (Int64.logand nzcv b) 0L) in
    match t with
    | Eq -> bit z
    | Ne -> not (bit z)
    | Hs -> bit c
    | Lo -> not (bit c)
    | Mi -> bit n
    | Pl -> not (bit n)
    | Vs -> bit v
    | Vc -> not (bit v)
    | Hi -> bit c && not (bit z)
    | Ls -> not (bit c && not (bit z))
    | Ge -> bit n = bit v
    | Lt -> bit n <> bit v
    | Gt -> (not (bit z)) && bit n = bit v
    | Le -> not ((not (bit z)) && bit n = bit v)

  let name = function
    | Eq -> "eq"
    | Ge -> "ge"
    | Gt -> "gt"
    | Hi -> "hi"
    | Hs -> "hs"
    | Le -> "le"
    | Lo -> "lo"
    | Ls -> "ls"
    | Lt -> "lt"
    | Mi -> "mi"
    | Ne -> "ne"
    | Pl -> "pl"
    | Vc -> "vc"
    | Vs -> "vs"
end

module Fop = struct
  type t = Add | Div | Max | Mul | Sub

  let name = function
    | Add -> "fadd"
    | Div -> "fdiv"
    | Max -> "fmax"
    | Mul -> "fmul"
    | Sub -> "fsub"
end

module Logic = struct
  type t = And | Eor | Orr

  let name = function And -> "and" | Eor -> "eor" | Orr -> "orr"
end

module Shift = struct
  type t = Asr | Lsl | Lsr

  let name = function Asr -> "asr" | Lsl -> "lsl" | Lsr -> "lsr"
end

module Funary = struct
  type t = Fneg | Frintz | Fsqrt

  let name = function Fneg -> "fneg" | Frintz -> "frintz" | Fsqrt -> "fsqrt"
end

(* An Advanced SIMD arrangement: two binary64 lanes of a Q register, two
   binary32 lanes of a D register, or four binary32 lanes of a Q register. *)
module Arr = struct
  type t = D2 | S2 | S4

  let fsz = function D2 -> Fsz.D | S2 | S4 -> Fsz.S
  let lanes = function D2 | S2 -> 2 | S4 -> 4

  let ty t =
    Mir_type.Vec
      ( (match fsz t with
        | Fsz.D -> Mir_type.Elem.F64
        | Fsz.S -> Mir_type.Elem.F32),
        Mir_type.Lanes.of_int (lanes t) )

  let name = function D2 -> "2d" | S2 -> "2s" | S4 -> "4s"
end

type v = Mir_value.t

type t =
  | Add of Sz.t * v * v
  | Add_imm of Sz.t * v * int64  (** ADD (immediate): imm12, optionally <<12 *)
  | Add_lo12 of v * Mir_id.View.t  (** ADD Xd, Xn, #:lo12:view *)
  | Adrp of Mir_id.View.t  (** the 4 KiB page holding the view's first byte *)
  | Bl of { callee : Mir_op.Callee.t; args : v list; results : Mir_type.t list }
      (** BL under AAPCS64: arguments and results in their fixed registers, then
          the [i32] status; every caller-saved bit clobbered *)
  | Cmp of Sz.t * v * v  (** SUBS ZR: NZCV of [a - b] *)
  | Cmp_imm of Sz.t * v * int64  (** SUBS ZR, #imm12 *)
  | Csel of Sz.t * Cond.t * v * v * v  (** flags, then, else *)
  | Cset of Cond.t * v  (** CSINC Wd, WZR, WZR, invert(cond): 0 or 1 *)
  | Dup_elem of Arr.t * v  (** DUP Vd.T, Vn.Ts[0]: every lane the scalar *)
  | Dup_half of int * v
      (** DUP Dd, Vn.D[k]: half [k] of a 4S vector as a 2S vector *)
  | Dup_lane of Fsz.t * int * v  (** DUP Sd/Dd, Vn.Ts[k]: a lane as a scalar *)
  | Ext of { signed : bool; from : Mir_width.t; src : v }
      (** SXTB, SXTH, UXTB, UXTH: a W register from a byte or halfword *)
  | Fbin of Fop.t * Fsz.t * v * v
  | Fcmp of Fsz.t * v * v
  | Fcsel of Fsz.t * Cond.t * v * v * v
  | Fcvt of Fsz.t * v  (** to the given precision from the other *)
  | Fcvtl of v  (** FCVTL Vd.2D, Vn.2S *)
  | Fcvtn of v  (** FCVTN Vd.2S, Vn.2D: one rounding a lane, upper half zero *)
  | Fcvtzs of Fsz.t * v  (** to X, toward zero, saturating *)
  | Fmadd of Fsz.t * v * v * v  (** [a * b + c], one rounding *)
  | Fmov of Fsz.t * v  (** register copy *)
  | Fmov_from_gpr of Fsz.t * v  (** S from W, D from X: the bits *)
  | Fmov_to_gpr of Fsz.t * v  (** W from S, X from D *)
  | Funary of Funary.t * Fsz.t * v
  | Ins_half of v * v
      (** INS Vd.D[1], Vn.D[0]: the 2S vector as the upper half, tied to the 4S
          vector *)
  | Ins_lane of Fsz.t * int * v * v
      (** INS Vd.Ts[k], Vn.Ts[0]: tied to the vector *)
  | Ld1_lane of Fsz.t * int * v * v
      (** LD1 \{Vt.Ts\}[k], [Xn]: one lane from memory, tied to the vector *)
  | Ld1r of Arr.t * v  (** LD1R \{Vt.T\}, [Xn]: one element to every lane *)
  | Ldr of Msz.t * v * int64
      (** [base, #imm]: unsigned, a multiple of the size, at most 4095 of them;
          B and H zero-extend into a W register *)
  | Ldr_vec of Arr.t * v * int64  (** LDR Qt or Dt, [Xn, #imm] *)
  | Logic of Logic.t * Sz.t * v * v
  | Logic_imm of Logic.t * Sz.t * v * int64  (** a bitmask immediate *)
  | Mov of Sz.t * v  (** ORR Rd, ZR, Rm *)
  | Movk of Sz.t * v * int * int  (** keeps the other halfwords: tied *)
  | Mrs_fpcr of v  (** an X register from FPCR (the operand: FPCR's value) *)
  | Msr_fpcr of v  (** FPCR from an X register (the result: FPCR's value) *)
  | Movn of Sz.t * int * int  (** NOT (imm16 << shift) *)
  | Movz of Mir_type.t * int * int
      (** imm16 << shift into a W (i8, i16, i32, pred) or X (i64) register,
          typed as the value it makes canonical *)
  | Msub of Sz.t * v * v * v  (** [c - a * b] *)
  | Mul of Sz.t * v * v
  | Scvtf of Fsz.t * v  (** from X, one rounding *)
  | Sdiv of Sz.t * v * v  (** truncating; [x / 0 = 0], [min / -1 = min] *)
  | Shift_imm of Shift.t * Sz.t * v * int
  | St1_lane of Fsz.t * int * v * v  (** ST1 \{Vt.Ts\}[k], [Xn]: vector, base *)
  | Str of Msz.t * v * int64 * v  (** base, imm, value *)
  | Str_vec of Arr.t * v * int64 * v
      (** STR Qt or Dt, [Xn, #imm]: base, value *)
  | Sub of Sz.t * v * v
  | Sxtw of v  (** X from W, sign-extended *)
  | Trunc of Mir_width.t * v
      (** UXTB, UXTH: a byte or halfword value from the low bits of a W *)
  | Uxtw of v  (** X from W, zero-extended (a W move) *)
  | Vfbin of Fop.t * Arr.t * v * v
  | Vfmla of Arr.t * v * v * v
      (** FMLA: [acc + a * b] a lane, one rounding, tied to the accumulator *)
  | Vfunary of Funary.t * Arr.t * v
  | Vmov of Arr.t * v  (** MOV Vd, Vn: the whole vector *)
  | Vwiden of v
      (** FMOV Dd, Dn: a 2S vector as the lower half of a 4S, upper half zero *)
  | Wtrunc of v  (** W from the low half of an X *)

type test = B_cond of Cond.t * v | Cbnz of Sz.t * v | Cbz of Sz.t * v

let uses = function
  | Add (_, a, b)
  | Cmp (_, a, b)
  | Fbin (_, _, a, b)
  | Fcmp (_, a, b)
  | Logic (_, _, a, b)
  | Mul (_, a, b)
  | Sdiv (_, a, b)
  | Sub (_, a, b)
  | Ins_half (a, b)
  | Ins_lane (_, _, a, b)
  | Ld1_lane (_, _, a, b)
  | St1_lane (_, _, a, b)
  | Vfbin (_, _, a, b) ->
      [ a; b ]
  | Add_imm (_, a, _)
  | Add_lo12 (a, _)
  | Cmp_imm (_, a, _)
  | Cset (_, a)
  | Dup_elem (_, a)
  | Dup_half (_, a)
  | Dup_lane (_, _, a)
  | Ext { src = a; _ }
  | Fcvt (_, a)
  | Fcvtl a
  | Fcvtn a
  | Fcvtzs (_, a)
  | Fmov (_, a)
  | Fmov_from_gpr (_, a)
  | Fmov_to_gpr (_, a)
  | Funary (_, _, a)
  | Ldr (_, a, _)
  | Ld1r (_, a)
  | Ldr_vec (_, a, _)
  | Logic_imm (_, _, a, _)
  | Mov (_, a)
  | Movk (_, a, _, _)
  | Mrs_fpcr a
  | Msr_fpcr a
  | Scvtf (_, a)
  | Shift_imm (_, _, a, _)
  | Sxtw a
  | Trunc (_, a)
  | Uxtw a
  | Vfunary (_, _, a)
  | Vmov (_, a)
  | Vwiden a
  | Wtrunc a ->
      [ a ]
  | Adrp _ | Movn _ | Movz _ -> []
  | Bl { args; _ } -> args
  | Csel (_, _, f, a, b) | Fcsel (_, _, f, a, b) -> [ f; a; b ]
  | Fmadd (_, a, b, c) | Msub (_, a, b, c) | Vfmla (_, a, b, c) -> [ a; b; c ]
  | Str (_, base, _, x) | Str_vec (_, base, _, x) -> [ base; x ]

let test_uses = function
  | B_cond (_, f) -> [ f ]
  | Cbnz (_, a) | Cbz (_, a) -> [ a ]

let test_flags_read = function
  | B_cond (c, _) -> Cond.reads c
  | Cbnz _ | Cbz _ -> 0L

let ordered = function
  | Bl _ | Ld1_lane _ | Ld1r _ | Ldr _ | Ldr_vec _ | St1_lane _ | Str _
  | Str_vec _ ->
      true
  | _ -> false

let constraints = function
  | Bl { args; results; _ } ->
      List.mapi
        (fun use view -> Mir_target.Constraint.Fixed_use { use; view })
        (A64_regs.args (List.map (fun (a : v) -> a.Mir_value.ty) args))
      @ List.mapi
          (fun result view ->
            Mir_target.Constraint.Fixed_result { result; view })
          (A64_regs.results (results @ [ Mir_type.i32 ]))
  | Ins_half _ | Ins_lane _ | Ld1_lane _ | Movk _ | Vfmla _ ->
      [ Mir_target.Constraint.Tied { result = 0; use = 0 } ]
  | _ -> []

(* Every admitted form writes a W, X, S or D view and zeroes the rest of the
   unit: bits 63:32 for W, bits 127:N of the vector register for S and D
   (DDI 0487 K.a, C1.2.5 and C2.1). *)
let result_write _ = Mir_target.Write.Zero_upper

let unit_bits = function
  | Mir_target.Bank.Control -> 64
  | Mir_target.Bank.Flags -> 4
  | Mir_target.Bank.Fpr -> 128
  | Mir_target.Bank.Gpr -> 64

(* A call leaves undefined every bit AAPCS64 does not preserve: x0-x18, the
   link register, v0-v7 and v16-v31 whole, the upper 64 bits of v8-v15, and
   NZCV. *)
let call_clobbers =
  List.map A64_reg.x (A64_reg.range 0 18 @ [ 30 ])
  @ List.map A64_reg.q (A64_reg.range 0 7 @ A64_reg.range 16 31)
  @ List.map
      (fun k ->
        {
          (A64_reg.q k) with
          Mir_target.View.name = Printf.sprintf "v%d.d[1]" k;
          lo = 64;
          bits = 64;
        })
      (A64_reg.range 8 15)
  @ [ A64_reg.nzcv ]

let clobbers = function Bl _ -> call_clobbers | _ -> []

let references = function
  | Add_lo12 (_, view) ->
      [
        Mir_target.Reference.View (view, Mir_target.Reference.Form.Page_offset);
      ]
  | Adrp view ->
      [ Mir_target.Reference.View (view, Mir_target.Reference.Form.Page) ]
  | Bl { callee; _ } -> [ Mir_target.Reference.Call callee ]
  | _ -> []

let writes_flags = function
  | Bl _ | Cmp _ | Cmp_imm _ | Fcmp _ -> true
  | _ -> false

let flags_defined = function Cmp _ | Cmp_imm _ | Fcmp _ -> 15L | _ -> 0L

let flags_read = function
  | Csel (_, c, _, _, _) | Fcsel (_, c, _, _, _) | Cset (c, _) -> Cond.reads c
  | _ -> 0L

let op_features = function
  | Bl { args; results; _ } ->
      let fp (t : Mir_type.t) = t = Mir_type.F32 || t = Mir_type.F64 in
      if List.exists fp (results @ List.map (fun (a : v) -> a.Mir_value.ty) args)
      then [ Mir_target.Feature.Fp ]
      else []
  | Dup_elem _ | Dup_half _ | Dup_lane _ | Fbin _ | Fcmp _ | Fcsel _ | Fcvt _
  | Fcvtl _ | Fcvtn _ | Fcvtzs _ | Fmadd _ | Fmov _ | Fmov_from_gpr _
  | Fmov_to_gpr _ | Funary _ | Ins_half _ | Ins_lane _ | Ld1_lane _ | Ld1r _
  | Ldr_vec _ | Scvtf _ | St1_lane _ | Str_vec _ | Vfbin _ | Vfmla _ | Vfunary _
  | Vmov _ | Vwiden _ ->
      [ Mir_target.Feature.Fp ]
  | Ldr ((Msz.S | Msz.D), _, _) | Str ((Msz.S | Msz.D), _, _, _) ->
      [ Mir_target.Feature.Fp ]
  | _ -> []

(* Whether [v] is an AArch64 bitmask immediate of [bits] bits: a repetition of
   a rotated run of ones that is neither empty nor full. *)
let bitmask_immediate ~bits v =
  let mask = if bits = 64 then -1L else Int64.pred (Int64.shift_left 1L bits) in
  let v = Int64.logand v mask in
  if Int64.equal v 0L || Int64.equal v mask then false
  else
    let rec period e =
      if e >= bits then bits
      else
        let m = Int64.pred (Int64.shift_left 1L e) in
        let p = Int64.logand v m in
        let rec same k =
          k >= bits
          || Int64.equal (Int64.logand (Int64.shift_right_logical v k) m) p
             && same (k + e)
        in
        if same e then e else period (2 * e)
    in
    let e = period 2 in
    let m = if e = 64 then -1L else Int64.pred (Int64.shift_left 1L e) in
    let p = Int64.logand v m in
    let rot =
      Int64.logand
        (Int64.logor
           (Int64.shift_right_logical p 1)
           (Int64.shift_left p (e - 1)))
        m
    in
    let rec popcount x n =
      if Int64.equal x 0L then n
      else popcount (Int64.logand x (Int64.pred x)) (n + 1)
    in
    popcount (Int64.logxor p rot) 0 = 2

let imm12 x =
  (Int64.compare x 0L >= 0 && Int64.compare x 4095L <= 0)
  || Int64.equal (Int64.rem x 4096L) 0L
     && Int64.compare x 0L > 0
     && Int64.compare x (Int64.shift_left 4095L 12) <= 0

let ( let* ) = Result.bind
let ty (v : v) = v.Mir_value.ty

let gpr sz (t : Mir_type.t) =
  match (sz, t) with
  | Sz.X, (Mir_type.Int Mir_width.W64 | Mir_type.Ptr) -> true
  | ( Sz.W,
      ( Mir_type.Int (Mir_width.W8 | Mir_width.W16 | Mir_width.W32)
      | Mir_type.Pred ) ) ->
      true
  | _ -> false

(* An arithmetic operand: W forms compute on i32 only (a predicate or a byte
   would leave its canonical range); X forms on i64, and a pointer where the
   rule says so. *)
let arith sz (t : Mir_type.t) =
  match (sz, t) with
  | Sz.X, Mir_type.Int Mir_width.W64 | Sz.W, Mir_type.Int Mir_width.W32 -> true
  | _ -> false

let fpr fsz (t : Mir_type.t) =
  match (fsz, t) with
  | Fsz.D, Mir_type.F64 | Fsz.S, Mir_type.F32 -> true
  | _ -> false

let fty = function Fsz.D -> Mir_type.F64 | Fsz.S -> Mir_type.F32
let need ok why = if ok then Ok () else Error why

(* The Q arrangement a lane form of [fsz] addresses. *)
let full = function Fsz.D -> Arr.D2 | Fsz.S -> Arr.S4
let is_arr arr (t : Mir_type.t) = Mir_type.equal t (Arr.ty arr)
let lane_ok fsz k = k >= 0 && k < Arr.lanes (full fsz)

(* A vector register's offset: a multiple of its size, at most 4095 of them. *)
let vec_offset arr k =
  let size = match arr with Arr.S2 -> 8L | Arr.D2 | Arr.S4 -> 16L in
  Int64.compare k 0L >= 0
  && Int64.equal (Int64.rem k size) 0L
  && Int64.compare (Int64.div k size) 4095L <= 0

(* The same integer type in and out; pointers stay pointers through X adds. *)
let typing op =
  match op with
  | Add (sz, a, b) -> (
      let ptr_or t = Mir_type.equal t Mir_type.Ptr && sz = Sz.X in
      let* () =
        need
          ((arith sz (ty a) || ptr_or (ty a))
          && (arith sz (ty b) || ptr_or (ty b)))
          "add operands"
      in
      match (ty a, ty b) with
      | Mir_type.Ptr, Mir_type.Ptr -> Error "add of two pointers"
      | Mir_type.Ptr, _ | _, Mir_type.Ptr -> Ok [ Mir_type.Ptr ]
      | t, u ->
          if Mir_type.equal t u then Ok [ t ] else Error "add operand types")
  | Add_imm (sz, a, k) ->
      let* () =
        need
          (arith sz (ty a) || (sz = Sz.X && Mir_type.equal (ty a) Mir_type.Ptr))
          "add operand"
      in
      let* () = need (imm12 k) "add immediate" in
      Ok [ ty a ]
  | Add_lo12 (a, _) ->
      let* () = need (Mir_type.equal (ty a) Mir_type.Ptr) "add lo12 base" in
      Ok [ Mir_type.Ptr ]
  | Adrp _ -> Ok [ Mir_type.Ptr ]
  | Bl { args; results; _ } ->
      let ok (t : Mir_type.t) =
        match t with
        | Mir_type.Int _ | Mir_type.Ptr | Mir_type.Pred | Mir_type.F32
        | Mir_type.F64 ->
            true
        | _ -> false
      in
      let* () =
        need (List.for_all ok (results @ List.map ty args)) "call operand types"
      in
      let fits tys =
        match A64_regs.args tys with
        | _ -> true
        | exception Invalid_argument _ -> false
      in
      let* () =
        need
          (fits (List.map ty args) && fits (results @ [ Mir_type.i32 ]))
          "more call operands than registers"
      in
      Ok (results @ [ Mir_type.i32 ])
  | Cmp (sz, a, b) ->
      let* () =
        need (gpr sz (ty a) && Mir_type.equal (ty a) (ty b)) "cmp operands"
      in
      Ok [ Mir_type.Flags ]
  | Cmp_imm (sz, a, k) ->
      let* () = need (gpr sz (ty a)) "cmp operand" in
      let* () = need (imm12 k) "cmp immediate" in
      Ok [ Mir_type.Flags ]
  | Csel (sz, _, f, a, b) ->
      let* () = need (Mir_type.equal (ty f) Mir_type.Flags) "csel flags" in
      let* () =
        need (gpr sz (ty a) && Mir_type.equal (ty a) (ty b)) "csel operands"
      in
      Ok [ ty a ]
  | Ext { from; src; _ } ->
      let* () =
        need
          ((from = Mir_width.W8 || from = Mir_width.W16)
          && Mir_type.equal (ty src) (Mir_type.Int from))
          "extension source"
      in
      Ok [ Mir_type.i32 ]
  | Cset (_, f) ->
      let* () = need (Mir_type.equal (ty f) Mir_type.Flags) "cset flags" in
      Ok [ Mir_type.Pred ]
  | Dup_elem (arr, a) ->
      let* () = need (fpr (Arr.fsz arr) (ty a)) "dup source" in
      Ok [ Arr.ty arr ]
  | Dup_half (k, a) ->
      let* () = need ((k = 0 || k = 1) && is_arr Arr.S4 (ty a)) "dup half" in
      Ok [ Arr.ty Arr.S2 ]
  | Dup_lane (fsz, k, a) ->
      let* () =
        need (lane_ok fsz k && is_arr (full fsz) (ty a)) "dup lane source"
      in
      Ok [ fty fsz ]
  | Fbin (_, fsz, a, b) ->
      let* () = need (fpr fsz (ty a) && fpr fsz (ty b)) "fp operands" in
      Ok [ fty fsz ]
  | Fcmp (fsz, a, b) ->
      let* () = need (fpr fsz (ty a) && fpr fsz (ty b)) "fcmp operands" in
      Ok [ Mir_type.Flags ]
  | Fcsel (fsz, _, f, a, b) ->
      let* () = need (Mir_type.equal (ty f) Mir_type.Flags) "fcsel flags" in
      let* () = need (fpr fsz (ty a) && fpr fsz (ty b)) "fcsel operands" in
      Ok [ fty fsz ]
  | Fcvt (fsz, a) ->
      let src = match fsz with Fsz.D -> Fsz.S | Fsz.S -> Fsz.D in
      let* () = need (fpr src (ty a)) "fcvt operand" in
      Ok [ fty fsz ]
  | Fcvtl a ->
      let* () = need (is_arr Arr.S2 (ty a)) "fcvtl source" in
      Ok [ Arr.ty Arr.D2 ]
  | Fcvtn a ->
      let* () = need (is_arr Arr.D2 (ty a)) "fcvtn source" in
      Ok [ Arr.ty Arr.S2 ]
  | Fcvtzs (fsz, a) ->
      let* () = need (fpr fsz (ty a)) "fcvtzs operand" in
      Ok [ Mir_type.i64 ]
  | Fmadd (fsz, a, b, c) ->
      let* () =
        need
          (fpr fsz (ty a) && fpr fsz (ty b) && fpr fsz (ty c))
          "fmadd operands"
      in
      Ok [ fty fsz ]
  | Fmov (fsz, a) ->
      let* () = need (fpr fsz (ty a)) "fmov operand" in
      Ok [ fty fsz ]
  | Fmov_from_gpr (fsz, a) ->
      let src =
        match fsz with Fsz.D -> Mir_type.i64 | Fsz.S -> Mir_type.i32
      in
      let* () = need (Mir_type.equal (ty a) src) "fmov source" in
      Ok [ fty fsz ]
  | Fmov_to_gpr (fsz, a) ->
      let* () = need (fpr fsz (ty a)) "fmov source" in
      Ok [ (match fsz with Fsz.D -> Mir_type.i64 | Fsz.S -> Mir_type.i32) ]
  | Funary (_, fsz, a) ->
      let* () = need (fpr fsz (ty a)) "fp operand" in
      Ok [ fty fsz ]
  | Ins_half (a, b) ->
      let* () =
        need (is_arr Arr.S4 (ty a) && is_arr Arr.S2 (ty b)) "ins half operands"
      in
      Ok [ ty a ]
  | Ins_lane (fsz, k, a, x) ->
      let* () =
        need
          (lane_ok fsz k && is_arr (full fsz) (ty a) && fpr fsz (ty x))
          "ins lane operands"
      in
      Ok [ ty a ]
  | Ld1_lane (fsz, k, a, base) ->
      let* () =
        need
          (lane_ok fsz k
          && is_arr (full fsz) (ty a)
          && Mir_type.equal (ty base) Mir_type.Ptr)
          "ld1 lane operands"
      in
      Ok [ ty a ]
  | Ld1r (arr, base) ->
      let* () = need (Mir_type.equal (ty base) Mir_type.Ptr) "ld1r base" in
      Ok [ Arr.ty arr ]
  | Ldr_vec (arr, base, k) ->
      let* () = need (Mir_type.equal (ty base) Mir_type.Ptr) "ldr base" in
      let* () = need (vec_offset arr k) "ldr offset" in
      Ok [ Arr.ty arr ]
  | Ldr (m, base, k) ->
      let* () = need (Mir_type.equal (ty base) Mir_type.Ptr) "ldr base" in
      let size = Msz.bytes m in
      let* () =
        need
          (Int64.compare k 0L >= 0
          && Int64.equal (Int64.rem k size) 0L
          && Int64.compare (Int64.div k size) 4095L <= 0)
          "ldr offset"
      in
      Ok
        [
          (match m with
          | Msz.B -> Mir_type.i8
          | Msz.H -> Mir_type.i16
          | Msz.W -> Mir_type.i32
          | Msz.X -> Mir_type.i64
          | Msz.S -> Mir_type.F32
          | Msz.D -> Mir_type.F64);
        ]
  | Logic (_, sz, a, b) ->
      let* () =
        need (gpr sz (ty a) && Mir_type.equal (ty a) (ty b)) "logical operands"
      in
      let* () =
        need (not (Mir_type.equal (ty a) Mir_type.Ptr)) "logic on a pointer"
      in
      Ok [ ty a ]
  | Logic_imm (o, sz, a, k) ->
      let* () =
        need
          (gpr sz (ty a) && not (Mir_type.equal (ty a) Mir_type.Ptr))
          "logical operand"
      in
      let* () =
        need (bitmask_immediate ~bits:(Sz.bits sz) k) "bitmask immediate"
      in
      (* the result keeps its operand's canonical range *)
      let* () =
        need
          (match (ty a, o) with
          | Mir_type.Pred, (Logic.And | Logic.Eor) -> Int64.equal k 1L
          | Mir_type.Pred, Logic.Orr -> false
          | Mir_type.Int w, (Logic.Eor | Logic.Orr) ->
              Int64.equal (Int64.logand k (Int64.lognot (Mir_width.mask w))) 0L
          | _ -> true)
          "immediate outside the operand's range"
      in
      Ok [ ty a ]
  | Mov (sz, a) ->
      let* () = need (gpr sz (ty a)) "mov operand" in
      Ok [ ty a ]
  | Mrs_fpcr a | Msr_fpcr a ->
      let* () =
        need (Mir_type.equal (ty a) Mir_type.i64) "fpcr transfer operand"
      in
      Ok [ Mir_type.i64 ]
  | Movk (sz, a, imm, shift) ->
      let* () = need (arith sz (ty a)) "movk operand" in
      let* () =
        need
          (imm >= 0 && imm <= 0xFFFF
          && shift mod 16 = 0
          && shift >= 0
          && shift < Sz.bits sz)
          "movk immediate"
      in
      Ok [ ty a ]
  | Movn (sz, imm, shift) ->
      let* () =
        need
          (imm >= 0 && imm <= 0xFFFF
          && shift mod 16 = 0
          && shift >= 0
          && shift < Sz.bits sz)
          "move immediate"
      in
      Ok [ (match sz with Sz.W -> Mir_type.i32 | Sz.X -> Mir_type.i64) ]
  | Movz (t, imm, shift) ->
      let limit, bits =
        match t with
        | Mir_type.Pred -> (1, 16)
        | Mir_type.Int Mir_width.W8 -> (0xFF, 16)
        | Mir_type.Int Mir_width.W16 -> (0xFFFF, 16)
        | Mir_type.Int Mir_width.W32 -> (0xFFFF, 32)
        | Mir_type.Int Mir_width.W64 -> (0xFFFF, 64)
        | _ -> (-1, 0)
      in
      let* () =
        need
          (imm >= 0 && imm <= limit
          && shift mod 16 = 0
          && shift >= 0 && shift < bits)
          "move immediate"
      in
      Ok [ t ]
  | Msub (sz, a, b, c) ->
      let* () =
        need
          (arith sz (ty a)
          && Mir_type.equal (ty a) (ty b)
          && Mir_type.equal (ty a) (ty c))
          "msub operands"
      in
      Ok [ ty a ]
  | Mul (sz, a, b) | Sdiv (sz, a, b) ->
      let* () =
        need
          (arith sz (ty a) && Mir_type.equal (ty a) (ty b))
          "multiply operands"
      in
      Ok [ ty a ]
  | Scvtf (fsz, a) ->
      let* () = need (Mir_type.equal (ty a) Mir_type.i64) "scvtf source" in
      Ok [ fty fsz ]
  | Shift_imm (_, sz, a, k) ->
      let* () = need (arith sz (ty a)) "shift operand" in
      let* () = need (k >= 0 && k < Sz.bits sz) "shift amount" in
      Ok [ ty a ]
  | St1_lane (fsz, k, a, base) ->
      let* () =
        need
          (lane_ok fsz k
          && is_arr (full fsz) (ty a)
          && Mir_type.equal (ty base) Mir_type.Ptr)
          "st1 lane operands"
      in
      Ok []
  | Str_vec (arr, base, k, x) ->
      let* () = need (Mir_type.equal (ty base) Mir_type.Ptr) "str base" in
      let* () = need (vec_offset arr k) "str offset" in
      let* () = need (is_arr arr (ty x)) "str value" in
      Ok []
  | Str (m, base, k, x) ->
      let* () = need (Mir_type.equal (ty base) Mir_type.Ptr) "str base" in
      let size = Msz.bytes m in
      let* () =
        need
          (Int64.compare k 0L >= 0
          && Int64.equal (Int64.rem k size) 0L
          && Int64.compare (Int64.div k size) 4095L <= 0)
          "str offset"
      in
      let* () =
        need
          (match (m, ty x) with
          | Msz.B, Mir_type.Int Mir_width.W8
          | Msz.H, Mir_type.Int Mir_width.W16
          | Msz.W, Mir_type.Int Mir_width.W32
          | Msz.X, Mir_type.Int Mir_width.W64
          | Msz.S, Mir_type.F32
          | Msz.D, Mir_type.F64 ->
              true
          | _ -> false)
          "str value"
      in
      Ok []
  | Sub (sz, a, b) -> (
      let* () =
        need
          ((arith sz (ty a) || (sz = Sz.X && Mir_type.equal (ty a) Mir_type.Ptr))
          && arith sz (ty b))
          "sub operands"
      in
      match (ty a, ty b) with
      | Mir_type.Ptr, Mir_type.Int Mir_width.W64 -> Ok [ Mir_type.Ptr ]
      | t, u ->
          if Mir_type.equal t u && not (Mir_type.equal t Mir_type.Ptr) then
            Ok [ t ]
          else Error "sub operand types")
  | Sxtw a | Uxtw a ->
      let* () = need (Mir_type.equal (ty a) Mir_type.i32) "extend source" in
      Ok [ Mir_type.i64 ]
  | Trunc (w, a) ->
      let* () =
        need
          ((w = Mir_width.W8 || w = Mir_width.W16)
          && Mir_type.equal (ty a) Mir_type.i32)
          "narrowing source"
      in
      Ok [ Mir_type.Int w ]
  | Vfbin (_, arr, a, b) ->
      let* () =
        need (is_arr arr (ty a) && is_arr arr (ty b)) "vector operands"
      in
      Ok [ Arr.ty arr ]
  | Vfmla (arr, a, b, c) ->
      let* () =
        need
          (is_arr arr (ty a) && is_arr arr (ty b) && is_arr arr (ty c))
          "fmla operands"
      in
      Ok [ Arr.ty arr ]
  | Vfunary (_, arr, a) | Vmov (arr, a) ->
      let* () = need (is_arr arr (ty a)) "vector operand" in
      Ok [ Arr.ty arr ]
  | Vwiden a ->
      let* () = need (is_arr Arr.S2 (ty a)) "widen source" in
      Ok [ Arr.ty Arr.S4 ]
  | Wtrunc a ->
      let* () = need (Mir_type.equal (ty a) Mir_type.i64) "truncate source" in
      Ok [ Mir_type.i32 ]

let test_typing = function
  | B_cond (_, f) ->
      need (Mir_type.equal (ty f) Mir_type.Flags) "branch on a non-condition"
  | Cbnz (sz, a) | Cbz (sz, a) ->
      need
        (gpr sz (ty a) && not (Mir_type.equal (ty a) Mir_type.Ptr))
        "compare-and-branch operand"

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
  | Dup_lane (fsz, k, a) ->
      Fmt.pf fmt "dup.%s %a.%s[%d]" (Fsz.name fsz) pv a (Fsz.name fsz) k
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
  | Sdiv (sz, a, b) -> Fmt.pf fmt "sdiv.%s %a" (Sz.name sz) vs [ a; b ]
  | Shift_imm (o, sz, a, k) ->
      Fmt.pf fmt "%s.%s %a, #%d" (Shift.name o) (Sz.name sz) pv a k
  | St1_lane (fsz, k, a, base) ->
      Fmt.pf fmt "st1 %a.%s[%d], [%a]" pv a (Fsz.name fsz) k pv base
  | Str (m, base, k, x) ->
      Fmt.pf fmt "str.%s %a, [%a, #%Ld]" (Msz.name m) pv x pv base k
  | Str_vec (arr, base, k, x) ->
      Fmt.pf fmt "str.%s %a, [%a, #%Ld]" (Arr.name arr) pv x pv base k
  | Sub (sz, a, b) -> Fmt.pf fmt "sub.%s %a" (Sz.name sz) vs [ a; b ]
  | Sxtw a -> Fmt.pf fmt "sxtw %a" pv a
  | Uxtw a -> Fmt.pf fmt "uxtw %a" pv a
  | Trunc (w, a) ->
      Fmt.pf fmt "uxt%s.trunc %a"
        (match w with Mir_width.W8 -> "b" | _ -> "h")
        pv a
  | Vfbin (o, arr, a, b) ->
      Fmt.pf fmt "%s.%s %a" (Fop.name o) (Arr.name arr) vs [ a; b ]
  | Vfmla (arr, a, b, c) ->
      Fmt.pf fmt "fmla.%s %a" (Arr.name arr) vs [ a; b; c ]
  | Vfunary (u, arr, a) ->
      Fmt.pf fmt "%s.%s %a" (Funary.name u) (Arr.name arr) pv a
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
