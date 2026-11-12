(* Typing of the admitted AArch64 forms: the operand types each form takes and
   the types it returns, over the closed variant of [A64_form]. *)

open Machine_ir
open A64_form

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

let is_mask arr (t : Mir_type.t) =
  match Arr.mask_ty arr with Some m -> Mir_type.equal t m | None -> false

let lane_ok fsz k = k >= 0 && k < Arr.lanes (full fsz)

(* The integer lane width of a register-wide integer vector: [i8x16] or
   [i16x8]. *)
let int_register (t : Mir_type.t) =
  match t with
  | Mir_type.Vec (Mir_type.Elem.Int Mir_width.W8, n)
    when Mir_type.Lanes.to_int n = 16 ->
      Some Mir_width.W8
  | Mir_type.Vec (Mir_type.Elem.Int Mir_width.W16, n)
    when Mir_type.Lanes.to_int n = 8 ->
      Some Mir_width.W16
  | _ -> None

let int32x4 =
  Mir_type.Vec (Mir_type.Elem.Int Mir_width.W32, Mir_type.Lanes.of_int 4)

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
  | Dup_mask (arr, a) -> (
      let want =
        match arr with
        | Arr.D2 -> Mir_type.i64
        | Arr.S2 | Arr.S4 -> Mir_type.i32
      in
      let* () = need (Mir_type.equal (ty a) want) "dup mask source" in
      match Arr.mask_ty arr with
      | Some m -> Ok [ m ]
      | None -> Error "a half-width compare has no mask")
  | Dup_w a ->
      let* () = need (Mir_type.equal (ty a) Mir_type.i32) "dup source" in
      Ok [ int32x4 ]
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
  | Ldr_ivec (t, base, k) ->
      let* () = need (Mir_type.equal (ty base) Mir_type.Ptr) "ldr base" in
      let* () = need (vec_offset Arr.S4 k) "ldr offset" in
      let* () = need (Option.is_some (int_register t)) "integer register" in
      Ok [ t ]
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
  | Ldr_idx (m, base, i) ->
      let* () = need (Mir_type.equal (ty base) Mir_type.Ptr) "ldr base" in
      let* () = need (Mir_type.equal (ty i) Mir_type.i32) "ldr index" in
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
  | Scvtf_vec a ->
      let* () = need (Mir_type.equal (ty a) int32x4) "scvtf source lanes" in
      Ok [ Arr.ty Arr.S4 ]
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
  | Str_ivec (t, base, k, x) ->
      let* () = need (Mir_type.equal (ty base) Mir_type.Ptr) "str base" in
      let* () = need (vec_offset Arr.S4 k) "str offset" in
      let* () =
        need
          (Option.is_some (int_register t) && Mir_type.equal (ty x) t)
          "integer register"
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
  | Str_idx (m, base, i, x) ->
      let* () = need (Mir_type.equal (ty base) Mir_type.Ptr) "str base" in
      let* () = need (Mir_type.equal (ty i) Mir_type.i32) "str index" in
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
  | Sub_imm (sz, a, k) ->
      let* () = need (arith sz (ty a)) "sub operand" in
      let* () = need (imm12 k) "sub immediate" in
      Ok [ ty a ]
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
  | Vbit (arr, other, a, m) ->
      let* () = need (is_mask arr (ty m)) "bit mask" in
      (* int32 lanes choose under a four-lane mask as binary32 lanes do *)
      let lanes t =
        is_arr arr t || (arr = Arr.S4 && Mir_type.equal t int32x4)
      in
      let* () =
        need (lanes (ty a) && Mir_type.equal (ty a) (ty other)) "bit operands"
      in
      Ok [ ty a ]
  | Uzp1_h (a, b) ->
      let* () =
        need
          (Mir_type.equal (ty a) int32x4 && Mir_type.equal (ty b) int32x4)
          "int32 lane operands"
      in
      Ok
        [
          Mir_type.Vec (Mir_type.Elem.Int Mir_width.W16, Mir_type.Lanes.of_int 8);
        ]
  | Vadd_w (a, b) ->
      let* () =
        need
          (Mir_type.equal (ty a) int32x4 && Mir_type.equal (ty b) int32x4)
          "int32 lane operands"
      in
      Ok [ int32x4 ]
  | Vfcmp (_, arr, a, b) -> (
      let* () =
        need (is_arr arr (ty a) && is_arr arr (ty b)) "vector compare operands"
      in
      match Arr.mask_ty arr with
      | Some m -> Ok [ m ]
      | None -> Error "a half-width compare has no mask")
  | Vlogic (_, a, b) ->
      let* () =
        need
          ((Option.is_some (Arr.of_mask (ty a)) || Mir_type.equal (ty a) int32x4)
          && Mir_type.equal (ty a) (ty b))
          "mask or int32 lane operands"
      in
      Ok [ ty a ]
  | Vcast (t, a) ->
      let q x = Mir_type.equal x int32x4 || is_arr Arr.S4 x in
      let* () = need (q t && q (ty a)) "a Q register's lanes" in
      Ok [ t ]
  | Vshift (o, k, a) ->
      let* () = need (Mir_type.equal (ty a) int32x4) "shifted lanes" in
      let* () =
        need
          (match o with
          | Shift.Lsl -> k >= 0 && k < 32
          | Shift.Asr | Shift.Lsr -> k >= 1 && k <= 32)
          "lane shift amount"
      in
      Ok [ int32x4 ]
  | Vnot a ->
      let* () = need (Option.is_some (Arr.of_mask (ty a))) "mask operand" in
      Ok [ ty a ]
  | Vfunary (_, arr, a) | Vmov (arr, a) ->
      let* () = need (is_arr arr (ty a)) "vector operand" in
      Ok [ Arr.ty arr ]
  | Vwiden a ->
      let* () = need (is_arr Arr.S2 (ty a)) "widen source" in
      Ok [ Arr.ty Arr.S4 ]
  | Wtrunc a ->
      let* () = need (Mir_type.equal (ty a) Mir_type.i64) "truncate source" in
      Ok [ Mir_type.i32 ]
  | Xtl_w { signed = _; from; first; src } ->
      let* () =
        need
          (match (int_register (ty src), from) with
          | Some Mir_width.W8, Mir_width.W8 -> List.mem first [ 0; 4; 8; 12 ]
          | Some Mir_width.W16, Mir_width.W16 -> List.mem first [ 0; 4 ]
          | _ -> false)
          "extended lanes"
      in
      Ok [ int32x4 ]

let test_typing = function
  | B_cond (_, f) ->
      need (Mir_type.equal (ty f) Mir_type.Flags) "branch on a non-condition"
  | Cbnz (sz, a) | Cbz (sz, a) ->
      need
        (gpr sz (ty a) && not (Mir_type.equal (ty a) Mir_type.Ptr))
        "compare-and-branch operand"
