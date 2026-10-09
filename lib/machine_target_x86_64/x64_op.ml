(* The admitted x86-64 selected forms over virtual values, each one Intel SDM
   form (Order Number 325462, revision 084): GPR forms at 32 or 64 bits,
   legacy-SSE scalar forms at single or double precision, SSE4.1 ROUNDSD/SS
   and FMA3 VFMADD231SD/SS only under their features. Destructive forms tie
   their result to the operand they overwrite; IDIV's dividend and results
   sit in fixed registers. Every consumer matches this type exhaustively and
   alphabetically; typing refuses an immediate the form cannot encode. *)

open Machine_ir

module Sz = struct
  type t = L | Q

  let bits = function L -> 32 | Q -> 64
  let name = function L -> "l" | Q -> "q"
end

module Fsz = struct
  type t = D | S

  let name = function D -> "sd" | S -> "ss"
end

(* A memory access size and the register file it moves. *)
module Msz = struct
  type t = B | D | H | L | Q | S

  let bytes = function B -> 1L | H -> 2L | L | S -> 4L | D | Q -> 8L

  let name = function
    | B -> "b"
    | D -> "sd"
    | H -> "w"
    | L -> "l"
    | Q -> "q"
    | S -> "ss"
end

(* Our compact RFLAGS: CF 1, PF 2, AF 4, ZF 8, SF 16, OF 32. *)
module Flag = struct
  let cf = 1L
  let pf = 2L
  let af = 4L
  let zf = 8L
  let sf = 16L
  let of_ = 32L
  let all = 63L
end

module Cond = struct
  type t = A | Ae | B | Be | E | G | Ge | L | Le | Ne | Np | P

  let reads = function
    | A | Be -> Int64.logor Flag.cf Flag.zf
    | Ae | B -> Flag.cf
    | E | Ne -> Flag.zf
    | G | Le -> Int64.logor Flag.zf (Int64.logor Flag.sf Flag.of_)
    | Ge | L -> Int64.logor Flag.sf Flag.of_
    | Np | P -> Flag.pf

  let holds t f =
    let bit b = not (Int64.equal (Int64.logand f b) 0L) in
    match t with
    | A -> (not (bit Flag.cf)) && not (bit Flag.zf)
    | Ae -> not (bit Flag.cf)
    | B -> bit Flag.cf
    | Be -> bit Flag.cf || bit Flag.zf
    | E -> bit Flag.zf
    | G -> (not (bit Flag.zf)) && bit Flag.sf = bit Flag.of_
    | Ge -> bit Flag.sf = bit Flag.of_
    | L -> bit Flag.sf <> bit Flag.of_
    | Le -> bit Flag.zf || bit Flag.sf <> bit Flag.of_
    | Ne -> not (bit Flag.zf)
    | Np -> not (bit Flag.pf)
    | P -> bit Flag.pf

  let name = function
    | A -> "a"
    | Ae -> "ae"
    | B -> "b"
    | Be -> "be"
    | E -> "e"
    | G -> "g"
    | Ge -> "ge"
    | L -> "l"
    | Le -> "le"
    | Ne -> "ne"
    | Np -> "np"
    | P -> "p"
end

module Alu = struct
  type t = Add | And | Or | Sub | Xor

  let name = function
    | Add -> "add"
    | And -> "and"
    | Or -> "or"
    | Sub -> "sub"
    | Xor -> "xor"
end

module Shift = struct
  type t = Sar | Shl | Shr

  let name = function Sar -> "sar" | Shl -> "shl" | Shr -> "shr"
end

module Fop = struct
  (* [Max] is MAXSD/MAXSS: the source operand when either is a NaN or both
     are zero, not IEEE maximum *)
  type t = Add | Div | Max | Mul | Sub

  let name = function
    | Add -> "add"
    | Div -> "div"
    | Max -> "max"
    | Mul -> "mul"
    | Sub -> "sub"
end

module Flogic = struct
  type t = And | Andn | Or | Xor

  let name = function
    | And -> "andp"
    | Andn -> "andnp"
    | Or -> "orp"
    | Xor -> "xorp"
end

(* CMPSD/CMPSS and CMPPS/CMPPD predicates admitted: 0 (equal, ordered), 1
   (less than, ordered), 2 (less or equal, ordered) and 3 (unordered). *)
module Cmp_pred = struct
  type t = Eq | Le | Lt | Unord

  let imm = function Eq -> 0L | Lt -> 1L | Le -> 2L | Unord -> 3L
  let name = function Eq -> "eq" | Le -> "le" | Lt -> "lt" | Unord -> "unord"
end

(* A packed format: four binary32 lanes (PS) or two binary64 lanes (PD) of an
   XMM register. *)
module Pk = struct
  type t = Pd | Ps

  let fsz = function Pd -> Fsz.D | Ps -> Fsz.S
  let lanes = function Pd -> 2 | Ps -> 4

  let ty t =
    Mir_type.Vec
      ( (match t with Pd -> Mir_type.Elem.F64 | Ps -> Mir_type.Elem.F32),
        Mir_type.Lanes.of_int (lanes t) )

  (* the mask a packed compare of these lanes fills: every bit of a lane set or
     clear *)
  let mask_ty t = Mir_type.Mask (Mir_type.Lanes.of_int (lanes t))
  let name = function Pd -> "pd" | Ps -> "ps"
end

(* Two binary32 lanes in an XMM register's low half. *)
let half_ty = Mir_type.Vec (Mir_type.Elem.F32, Mir_type.Lanes.of_int 2)

type v = Mir_value.t

(* [base + index * scale + disp]: scale 1, 2, 4 or 8; disp a signed 32-bit
   value *)
module Addr = struct
  type t = { base : v; index : (v * int64) option; disp : int64 }

  let uses a = a.base :: Option.to_list (Option.map fst a.index)
end

type t =
  | Alu of Alu.t * Sz.t * v * v
  | Alu_imm of Alu.t * Sz.t * v * int64
      (** an imm32, sign-extended at 64 bits; tied *)
  | Bt of Sz.t * v * int  (** CF := the bit; OF, SF, AF, PF undefined *)
  | Call of {
      callee : Mir_op.Callee.t;
      args : v list;
      results : Mir_type.t list;
    }
  | Cmov of Sz.t * Cond.t * v * v * v  (** flags, taken, else (tied) *)
  | Cmp of Sz.t * v * v
  | Cmp_imm of Sz.t * v * int64  (** CMP r, imm32 *)
  | Cmps of Cmp_pred.t * Fsz.t * v * v
  | Cqo_idiv of v * v  (** CQO then IDIV: rdx:rax / divisor *)
  | Cvt of Fsz.t * v  (** CVTSS2SD or CVTSD2SS, to the given precision *)
  | Cvtpd2ps of v  (** CVTPD2PS: two binary64 lanes to a low binary32 half *)
  | Cvtps2pd of v  (** CVTPS2PD: a low binary32 half to two binary64 lanes *)
  | Cvtsi2s of Fsz.t * v  (** from a 64-bit integer *)
  | Cvtts2si of Fsz.t * v  (** to a 64-bit integer, truncating *)
  | Ext of { signed : bool; from : Mir_width.t; src : v }
      (** MOVSX/MOVZX r32 from a byte or word *)
  | Fbin of Fop.t * Fsz.t * v * v
  | Flogic of Flogic.t * Fsz.t * v * v
  | Fmadd231 of Fsz.t * v * v * v  (** [a * b + c], one rounding, tied to [c] *)
  | Imul of Sz.t * v * v
  | Imul_imm of Sz.t * v * int64  (** IMUL r, r/m, imm32: not tied *)
  | Lea of v * int64
  | Lea_view of Mir_id.View.t  (** RIP-relative *)
  | Load of Msz.t * Addr.t
  | Mov of Sz.t * v
  | Mov_imm of Mir_type.t * int64  (** MOV r32, imm32 or MOVABS r64, imm64 *)
  | Movap of v  (** MOVAPS/MOVAPD register copy *)
  | Movap_to_mask of Pk.t * v  (** the same, the copy typed as a mask *)
  | Movlhps of v * v
      (** MOVLHPS: the half as the upper 64 bits, tied to the vector *)
  | Movq_from_gpr of Fsz.t * v
  | Movq_low of v  (** MOVQ xmm, xmm: a PS vector's low half, upper zero *)
  | Movq_to_gpr of Fsz.t * v
  | Movq_widen of v  (** MOVQ xmm, xmm: a half as a PS vector, upper zero *)
  | Movup_load of Pk.t * Addr.t  (** MOVUPS/MOVUPD from memory *)
  | Movup_store of Pk.t * Addr.t * v  (** MOVUPS/MOVUPD to memory *)
  | Movsxd of v
  | Movzx32 of v  (** MOV r32, r32: a 32-bit value zero-extended *)
  | Neg of Sz.t * v
  | Pbin of Fop.t * Pk.t * v * v
      (** ADDPS … SUBPD, tied; MAXPS is not IEEE maximum, as for {!Fbin} *)
  | Pcmp of Cmp_pred.t * Pk.t * v * v
      (** CMPPS/CMPPD: a mask, every bit of a lane set where the compare holds;
          tied to the first operand, which is a copy typed as the mask
          ({!Movap_to_mask}) because a tie needs equal types *)
  | Pfmadd231 of Pk.t * v * v * v
      (** VFMADD231PS/PD: [a * b + c] a lane, tied to [c] *)
  | Plogic of Flogic.t * Pk.t * v * v
      (** ANDPS … XORPD, tied; the operands are vectors or masks of the lanes,
          and the result has the first operand's type *)
  | Pones of Pk.t  (** PCMPEQD r, r: a mask with every bit set *)
  | Pshuf of Pk.t * int * v * v
      (** SHUFPS/SHUFPD: lanes from the first operand, then from the second,
          chosen by the immediate; tied to the first *)
  | Pshufd_half of v  (** PSHUFD 0xEE: a PS vector's high half, low *)
  | Pshufd_lane of Fsz.t * int * v  (** PSHUFD: lane [k] to lane 0, a scalar *)
  | Pshufd_splat of Pk.t * v  (** PSHUFD 0x00 or 0x44: a scalar to every lane *)
  | Pshufd_splat_mask of Pk.t * v
      (** the same, the scalar's bits taken as a lane of a mask *)
  | Psqrt of Pk.t * v  (** SQRTPS/SQRTPD *)
  | Round_trunc of Fsz.t * v  (** ROUNDSD/ROUNDSS with immediate 3 *)
  | Setcc_zx of Cond.t * v
      (** SETcc r8 then MOVZX r32, r8: probed natively as the pair *)
  | Shift_imm of Shift.t * Sz.t * v * int
  | Sqrt of Fsz.t * v
  | Store of Msz.t * Addr.t * v
  | Test of Sz.t * v * v
  | Trunc32 of v  (** MOV r32, r32 of a 64-bit value *)
  | Trunc_zx of Mir_width.t * v
      (** MOVZX r32 from the low byte or word of an r32 *)
  | Ucomis of Fsz.t * v * v

type test = Jcc of Cond.t * v

let uses = function
  | Alu (_, _, a, b)
  | Cmp (_, a, b)
  | Cmps (_, _, a, b)
  | Cqo_idiv (a, b)
  | Fbin (_, _, a, b)
  | Flogic (_, _, a, b)
  | Movlhps (a, b)
  | Pbin (_, _, a, b)
  | Pcmp (_, _, a, b)
  | Plogic (_, _, a, b)
  | Pshuf (_, _, a, b)
  | Imul (_, a, b)
  | Test (_, a, b)
  | Ucomis (_, a, b) ->
      [ a; b ]
  | Alu_imm (_, _, a, _)
  | Bt (_, a, _)
  | Cmp_imm (_, a, _)
  | Cvt (_, a)
  | Cvtpd2ps a
  | Cvtps2pd a
  | Cvtsi2s (_, a)
  | Cvtts2si (_, a)
  | Ext { src = a; _ }
  | Imul_imm (_, a, _)
  | Lea (a, _)
  | Mov (_, a)
  | Movap a
  | Movap_to_mask (_, a)
  | Movq_from_gpr (_, a)
  | Movq_low a
  | Movq_to_gpr (_, a)
  | Movq_widen a
  | Movsxd a
  | Movzx32 a
  | Neg (_, a)
  | Pshufd_half a
  | Pshufd_lane (_, _, a)
  | Pshufd_splat (_, a)
  | Pshufd_splat_mask (_, a)
  | Psqrt (_, a)
  | Round_trunc (_, a)
  | Setcc_zx (_, a)
  | Shift_imm (_, _, a, _)
  | Sqrt (_, a)
  | Trunc32 a
  | Trunc_zx (_, a) ->
      [ a ]
  | Call { args; _ } -> args
  | Cmov (_, _, f, a, b) -> [ f; a; b ]
  | Fmadd231 (_, a, b, c) | Pfmadd231 (_, a, b, c) -> [ a; b; c ]
  | Lea_view _ | Mov_imm _ | Pones _ -> []
  | Load (_, addr) | Movup_load (_, addr) -> Addr.uses addr
  | Movup_store (_, addr, x) | Store (_, addr, x) -> Addr.uses addr @ [ x ]

let test_uses = function Jcc (_, f) -> [ f ]
let test_flags_read = function Jcc (c, _) -> Cond.reads c

let ordered = function
  | Call _ | Load _ | Movup_load _ | Movup_store _ | Store _ -> true
  | _ -> false

let tied0 = [ Mir_target.Constraint.Tied { result = 0; use = 0 } ]

let constraints = function
  | Alu _ | Alu_imm _ | Cmps _ | Fbin _ | Flogic _ | Imul _ | Movlhps _ | Neg _
  | Pbin _ | Pcmp _ | Plogic _ | Pshuf _ | Shift_imm _ ->
      tied0
  | Cmov _ | Fmadd231 _ | Pfmadd231 _ ->
      [ Mir_target.Constraint.Tied { result = 0; use = 2 } ]
  | Cqo_idiv _ ->
      (* CQO writes rdx before IDIV reads the divisor *)
      [
        Mir_target.Constraint.Fixed_use
          { use = 0; view = X64_reg.q X64_reg.rax };
        Mir_target.Constraint.Fixed_result
          { result = 0; view = X64_reg.q X64_reg.rax };
        Mir_target.Constraint.Fixed_result
          { result = 1; view = X64_reg.q X64_reg.rdx };
        Mir_target.Constraint.Early_clobber 1;
      ]
  | Call { args; results; _ } ->
      List.mapi
        (fun use view -> Mir_target.Constraint.Fixed_use { use; view })
        (X64_regs.args (List.map (fun (a : v) -> a.Mir_value.ty) args))
      @ List.mapi
          (fun result view ->
            Mir_target.Constraint.Fixed_result { result; view })
          (X64_regs.results (results @ [ Mir_type.i32 ]))
  | _ -> []

(* A call leaves undefined every bit System V does not preserve: rax, rcx,
   rdx, rsi, rdi, r8-r11, every XMM register whole, and the flags. *)
let call_clobbers =
  List.map X64_reg.q
    [
      X64_reg.rax;
      X64_reg.rcx;
      X64_reg.rdx;
      X64_reg.rsi;
      X64_reg.rdi;
      8;
      9;
      10;
      11;
    ]
  @ List.map X64_reg.xmm (X64_reg.range 0 15)
  @ [ X64_reg.rflags ]

let clobbers = function Call _ -> call_clobbers | _ -> []

let references = function
  | Call { callee; _ } -> [ Mir_target.Reference.Call callee ]
  | Lea_view view ->
      [
        Mir_target.Reference.View (view, Mir_target.Reference.Form.Pc_relative);
      ]
  | _ -> []

let writes_flags = function
  | Alu _ | Alu_imm _ | Bt _ | Call _ | Cmp _ | Cmp_imm _ | Cqo_idiv _ | Imul _
  | Imul_imm _ | Neg _ | Shift_imm _ | Test _ | Ucomis _ ->
      true
  | _ -> false

let flags_defined = function
  | Bt _ -> Flag.cf
  | Cmp _ | Cmp_imm _ | Ucomis _ -> Flag.all
  | Test _ -> Int64.logand Flag.all (Int64.lognot Flag.af)
  | _ -> 0L

let flags_read = function
  | Cmov (_, c, _, _, _) | Setcc_zx (c, _) -> Cond.reads c
  | _ -> 0L

(* Legacy-SSE scalar results keep the destination's bits above them;
   whole-register bitwise forms and copies leave them untracked; GPR writes and
   scalar loads zero them. *)
let result_write = function
  | Cmps _ | Cvt _ | Cvtsi2s _ | Fbin _ | Fmadd231 _ | Round_trunc _ | Sqrt _ ->
      Mir_target.Write.Merge
  | Flogic _ | Movap _ | Pshufd_half _ | Pshufd_lane _ ->
      Mir_target.Write.Undefined_upper
  | _ -> Mir_target.Write.Zero_upper

let unit_bits = function
  | Mir_target.Bank.Control -> 32
  | Mir_target.Bank.Flags -> 6
  | Mir_target.Bank.Fpr -> 128
  | Mir_target.Bank.Gpr -> 64

let op_features = function
  | Fmadd231 _ | Pfmadd231 _ -> [ Mir_target.Feature.Fma ]
  | Round_trunc _ -> [ Mir_target.Feature.Sse41 ]
  | Cmps _ | Cvt _ | Cvtpd2ps _ | Cvtps2pd _ | Cvtsi2s _ | Cvtts2si _ | Fbin _
  | Flogic _ | Movap _ | Movap_to_mask _ | Movlhps _ | Movq_from_gpr _
  | Movq_low _ | Movq_to_gpr _ | Movq_widen _ | Movup_load _ | Movup_store _
  | Pbin _ | Pcmp _ | Plogic _ | Pones _ | Pshuf _ | Pshufd_half _
  | Pshufd_lane _ | Pshufd_splat _ | Pshufd_splat_mask _ | Psqrt _ | Sqrt _
  | Ucomis _ ->
      [ Mir_target.Feature.Sse2 ]
  | Load ((Msz.S | Msz.D), _) | Store ((Msz.S | Msz.D), _, _) ->
      [ Mir_target.Feature.Sse2 ]
  | Call { args; results; _ } ->
      if
        List.exists
          (fun (t : Mir_type.t) -> t = Mir_type.F32 || t = Mir_type.F64)
          (results @ List.map (fun (a : v) -> a.Mir_value.ty) args)
      then [ Mir_target.Feature.Sse2 ]
      else []
  | _ -> []

let ( let* ) = Result.bind
let ty (v : v) = v.Mir_value.ty
let need ok why = if ok then Ok () else Error why

let arith sz (t : Mir_type.t) =
  match (sz, t) with
  | Sz.Q, Mir_type.Int Mir_width.W64 | Sz.L, Mir_type.Int Mir_width.W32 -> true
  | _ -> false

(* a GPR operand of the size, predicates and pointers included *)
let gpr sz (t : Mir_type.t) =
  match (sz, t) with
  | Sz.Q, (Mir_type.Int Mir_width.W64 | Mir_type.Ptr) -> true
  | ( Sz.L,
      ( Mir_type.Int (Mir_width.W8 | Mir_width.W16 | Mir_width.W32)
      | Mir_type.Pred ) ) ->
      true
  | _ -> false

let fpr fsz (t : Mir_type.t) =
  match (fsz, t) with
  | Fsz.D, Mir_type.F64 | Fsz.S, Mir_type.F32 -> true
  | _ -> false

let fty = function Fsz.D -> Mir_type.F64 | Fsz.S -> Mir_type.F32
let is_pk pk t = Mir_type.equal t (Pk.ty pk)
let is_mask pk t = Mir_type.equal t (Pk.mask_ty pk)
let is_lanes pk t = is_pk pk t || is_mask pk t
let is_half t = Mir_type.equal t half_ty

let disp32 k =
  Int64.compare k (-0x8000_0000L) >= 0 && Int64.compare k 0x7FFF_FFFFL <= 0

(* An immediate operand at [sz]: the 32 bits at 32, a value the imm32's sign
   extension reaches at 64. *)
let imm32 sz k =
  match sz with Sz.L -> Mir_width.canonical Mir_width.W32 k | Sz.Q -> disp32 k

let addr_ok (a : Addr.t) =
  let* () =
    need (Mir_type.equal (ty a.Addr.base) Mir_type.Ptr) "address base"
  in
  let* () = need (disp32 a.Addr.disp) "displacement" in
  match a.Addr.index with
  | None -> Ok ()
  | Some (i, s) ->
      let* () = need (Mir_type.equal (ty i) Mir_type.i64) "address index" in
      need (List.mem s [ 1L; 2L; 4L; 8L ]) "address scale"

let typing op =
  match op with
  | Alu (o, sz, a, b) -> (
      match (o, sz, ty a, ty b) with
      | (Alu.Add | Alu.Sub), Sz.Q, Mir_type.Ptr, Mir_type.Int Mir_width.W64 ->
          Ok [ Mir_type.Ptr ]
      | (Alu.And | Alu.Or | Alu.Xor), Sz.L, Mir_type.Pred, Mir_type.Pred ->
          Ok [ Mir_type.Pred ]
      | _, _, t, u ->
          let* () = need (arith sz t && Mir_type.equal t u) "alu operands" in
          Ok [ t ])
  | Alu_imm (_, sz, a, k) ->
      let* () = need (arith sz (ty a)) "alu operand" in
      let* () = need (imm32 sz k) "alu immediate" in
      Ok [ ty a ]
  | Bt (sz, a, k) ->
      let* () = need (arith sz (ty a)) "bt operand" in
      let* () = need (k >= 0 && k < Sz.bits sz) "bit index" in
      Ok [ Mir_type.Flags ]
  | Call { args; results; _ } ->
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
      let fits f tys =
        match f tys with _ -> true | exception Invalid_argument _ -> false
      in
      let* () =
        need
          (fits X64_regs.args (List.map ty args)
          && fits X64_regs.results (results @ [ Mir_type.i32 ]))
          "more call operands than registers"
      in
      Ok (results @ [ Mir_type.i32 ])
  | Cmov (sz, _, f, a, b) ->
      let* () = need (Mir_type.equal (ty f) Mir_type.Flags) "cmov flags" in
      let* () =
        need (gpr sz (ty a) && Mir_type.equal (ty a) (ty b)) "cmov operands"
      in
      Ok [ ty a ]
  | Cmp (sz, a, b) | Test (sz, a, b) ->
      let* () =
        need (gpr sz (ty a) && Mir_type.equal (ty a) (ty b)) "compare operands"
      in
      Ok [ Mir_type.Flags ]
  | Cmp_imm (sz, a, k) ->
      let* () = need (arith sz (ty a)) "compare operand" in
      let* () = need (imm32 sz k) "compare immediate" in
      Ok [ Mir_type.Flags ]
  | Cmps (_, fsz, a, b) | Fbin (_, fsz, a, b) | Flogic (_, fsz, a, b) ->
      let* () = need (fpr fsz (ty a) && fpr fsz (ty b)) "sse operands" in
      Ok [ fty fsz ]
  | Cqo_idiv (a, b) ->
      let* () =
        need
          (Mir_type.equal (ty a) Mir_type.i64
          && Mir_type.equal (ty b) Mir_type.i64)
          "idiv operands"
      in
      Ok [ Mir_type.i64; Mir_type.i64 ]
  | Cvt (fsz, a) ->
      let src = match fsz with Fsz.D -> Fsz.S | Fsz.S -> Fsz.D in
      let* () = need (fpr src (ty a)) "cvt operand" in
      Ok [ fty fsz ]
  | Cvtpd2ps a ->
      let* () = need (is_pk Pk.Pd (ty a)) "cvtpd2ps operand" in
      Ok [ half_ty ]
  | Cvtps2pd a ->
      let* () = need (is_half (ty a)) "cvtps2pd operand" in
      Ok [ Pk.ty Pk.Pd ]
  | Cvtsi2s (fsz, a) ->
      let* () = need (Mir_type.equal (ty a) Mir_type.i64) "cvtsi2s operand" in
      Ok [ fty fsz ]
  | Cvtts2si (fsz, a) ->
      let* () = need (fpr fsz (ty a)) "cvtts2si operand" in
      Ok [ Mir_type.i64 ]
  | Ext { from; src; _ } ->
      let* () =
        need
          ((from = Mir_width.W8 || from = Mir_width.W16)
          && Mir_type.equal (ty src) (Mir_type.Int from))
          "extension source"
      in
      Ok [ Mir_type.i32 ]
  | Fmadd231 (fsz, a, b, c) ->
      let* () =
        need (fpr fsz (ty a) && fpr fsz (ty b) && fpr fsz (ty c)) "fma operands"
      in
      Ok [ fty fsz ]
  | Imul (sz, a, b) ->
      let* () =
        need (arith sz (ty a) && Mir_type.equal (ty a) (ty b)) "imul operands"
      in
      Ok [ ty a ]
  | Imul_imm (sz, a, k) ->
      let* () = need (arith sz (ty a)) "imul operand" in
      let* () = need (imm32 sz k) "imul immediate" in
      Ok [ ty a ]
  | Lea (a, k) ->
      let* () = need (Mir_type.equal (ty a) Mir_type.Ptr) "lea base" in
      let* () = need (disp32 k) "lea displacement" in
      Ok [ Mir_type.Ptr ]
  | Lea_view _ -> Ok [ Mir_type.Ptr ]
  | Load (m, a) ->
      let* () = addr_ok a in
      Ok
        [
          (match m with
          | Msz.B -> Mir_type.i8
          | Msz.H -> Mir_type.i16
          | Msz.L -> Mir_type.i32
          | Msz.Q -> Mir_type.i64
          | Msz.S -> Mir_type.F32
          | Msz.D -> Mir_type.F64);
        ]
  | Mov (sz, a) ->
      let* () = need (gpr sz (ty a)) "mov operand" in
      Ok [ ty a ]
  | Mov_imm (t, k) ->
      let* () =
        need
          (match t with
          | Mir_type.Int w -> Mir_width.canonical w k
          | Mir_type.Pred -> Int64.equal k 0L || Int64.equal k 1L
          | _ -> false)
          "move immediate"
      in
      Ok [ t ]
  | Movap a ->
      let* () =
        need
          (fpr Fsz.D (ty a)
          || fpr Fsz.S (ty a)
          || List.exists (fun pk -> is_lanes pk (ty a)) [ Pk.Pd; Pk.Ps ])
          "movap operand"
      in
      Ok [ ty a ]
  | Movlhps (a, b) ->
      let* () =
        need (is_pk Pk.Ps (ty a) && is_half (ty b)) "movlhps operands"
      in
      Ok [ ty a ]
  | Movq_low a ->
      let* () = need (is_pk Pk.Ps (ty a)) "movq operand" in
      Ok [ half_ty ]
  | Movq_widen a ->
      let* () = need (is_half (ty a)) "movq operand" in
      Ok [ Pk.ty Pk.Ps ]
  | Movup_load (pk, a) ->
      let* () = addr_ok a in
      Ok [ Pk.ty pk ]
  | Movup_store (pk, a, x) ->
      let* () = addr_ok a in
      let* () = need (is_pk pk (ty x)) "movup value" in
      Ok []
  | Pbin (_, pk, a, b) ->
      let* () = need (is_pk pk (ty a) && is_pk pk (ty b)) "packed operands" in
      Ok [ Pk.ty pk ]
  | Pfmadd231 (pk, a, b, c) ->
      let* () =
        need
          (is_pk pk (ty a) && is_pk pk (ty b) && is_pk pk (ty c))
          "packed fma operands"
      in
      Ok [ Pk.ty pk ]
  | Movap_to_mask (pk, a) ->
      let* () = need (is_pk pk (ty a)) "movap operand" in
      Ok [ Pk.mask_ty pk ]
  | Pcmp (_, pk, a, b) ->
      let* () = need (is_mask pk (ty a) && is_pk pk (ty b)) "packed operands" in
      Ok [ Pk.mask_ty pk ]
  | Plogic (_, pk, a, b) ->
      let* () =
        need (is_lanes pk (ty a) && is_lanes pk (ty b)) "packed operands"
      in
      Ok [ ty a ]
  | Pones pk -> Ok [ Pk.mask_ty pk ]
  | Pshuf (pk, k, a, b) ->
      let* () = need (is_pk pk (ty a) && is_pk pk (ty b)) "packed operands" in
      let* () =
        need
          (match pk with
          | Pk.Ps -> k >= 0 && k < 256
          | Pk.Pd -> k >= 0 && k < 4)
          "shuffle immediate"
      in
      Ok [ Pk.ty pk ]
  | Pshufd_half a ->
      let* () = need (is_pk Pk.Ps (ty a)) "pshufd operand" in
      Ok [ half_ty ]
  | Pshufd_lane (fsz, k, a) ->
      let pk = match fsz with Fsz.D -> Pk.Pd | Fsz.S -> Pk.Ps in
      let* () =
        need (k >= 0 && k < Pk.lanes pk && is_pk pk (ty a)) "pshufd lane"
      in
      Ok [ fty fsz ]
  | Pshufd_splat (pk, a) ->
      let* () = need (fpr (Pk.fsz pk) (ty a)) "pshufd scalar" in
      Ok [ Pk.ty pk ]
  | Pshufd_splat_mask (pk, a) ->
      let* () = need (fpr (Pk.fsz pk) (ty a)) "pshufd scalar" in
      Ok [ Pk.mask_ty pk ]
  | Psqrt (pk, a) ->
      let* () = need (is_pk pk (ty a)) "sqrtp operand" in
      Ok [ Pk.ty pk ]
  | Movq_from_gpr (fsz, a) ->
      let src =
        match fsz with Fsz.D -> Mir_type.i64 | Fsz.S -> Mir_type.i32
      in
      let* () = need (Mir_type.equal (ty a) src) "movq source" in
      Ok [ fty fsz ]
  | Movq_to_gpr (fsz, a) ->
      let* () = need (fpr fsz (ty a)) "movq source" in
      Ok [ (match fsz with Fsz.D -> Mir_type.i64 | Fsz.S -> Mir_type.i32) ]
  | Movsxd a | Movzx32 a ->
      let* () = need (Mir_type.equal (ty a) Mir_type.i32) "extend source" in
      Ok [ Mir_type.i64 ]
  | Neg (sz, a) ->
      let* () = need (arith sz (ty a)) "neg operand" in
      Ok [ ty a ]
  | Round_trunc (fsz, a) | Sqrt (fsz, a) ->
      let* () = need (fpr fsz (ty a)) "sse operand" in
      Ok [ fty fsz ]
  | Setcc_zx (_, f) ->
      let* () = need (Mir_type.equal (ty f) Mir_type.Flags) "setcc flags" in
      Ok [ Mir_type.Pred ]
  | Shift_imm (_, sz, a, k) ->
      let* () = need (arith sz (ty a)) "shift operand" in
      let* () = need (k >= 0 && k < Sz.bits sz) "shift count" in
      Ok [ ty a ]
  | Store (m, a, x) ->
      let* () = addr_ok a in
      let* () =
        need
          (match (m, ty x) with
          | Msz.B, Mir_type.Int Mir_width.W8
          | Msz.H, Mir_type.Int Mir_width.W16
          | Msz.L, Mir_type.Int Mir_width.W32
          | Msz.Q, Mir_type.Int Mir_width.W64
          | Msz.S, Mir_type.F32
          | Msz.D, Mir_type.F64 ->
              true
          | _ -> false)
          "store value"
      in
      Ok []
  | Trunc32 a ->
      let* () = need (Mir_type.equal (ty a) Mir_type.i64) "truncate source" in
      Ok [ Mir_type.i32 ]
  | Trunc_zx (w, a) ->
      let* () =
        need
          ((w = Mir_width.W8 || w = Mir_width.W16)
          && Mir_type.equal (ty a) Mir_type.i32)
          "narrowing source"
      in
      Ok [ Mir_type.Int w ]
  | Ucomis (fsz, a, b) ->
      let* () = need (fpr fsz (ty a) && fpr fsz (ty b)) "ucomis operands" in
      Ok [ Mir_type.Flags ]

let test_typing = function
  | Jcc (_, f) ->
      need (Mir_type.equal (ty f) Mir_type.Flags) "branch on a non-condition"

let pp_addr pv fmt (a : Addr.t) =
  Fmt.pf fmt "[%a%a%+Ld]" pv a.Addr.base
    Fmt.(option (fun fmt (i, s) -> Fmt.pf fmt " + %a*%Ld" pv i s))
    a.Addr.index a.Addr.disp

let pp_op pv fmt op =
  let vs = Fmt.(list ~sep:(any ", ") pv) in
  match op with
  | Alu (o, sz, a, b) ->
      Fmt.pf fmt "%s%s %a" (Alu.name o) (Sz.name sz) vs [ a; b ]
  | Alu_imm (o, sz, a, k) ->
      Fmt.pf fmt "%s%s %a, $%Ld" (Alu.name o) (Sz.name sz) pv a k
  | Bt (sz, a, k) -> Fmt.pf fmt "bt%s %a, $%d" (Sz.name sz) pv a k
  | Call { callee; args; _ } ->
      Fmt.pf fmt "call %a(%a)" Mir_op.Callee.pp callee vs args
  | Cmov (sz, c, f, a, b) ->
      Fmt.pf fmt "cmov%s%s %a" (Cond.name c) (Sz.name sz) vs [ f; a; b ]
  | Cmp (sz, a, b) -> Fmt.pf fmt "cmp%s %a" (Sz.name sz) vs [ a; b ]
  | Cmp_imm (sz, a, k) -> Fmt.pf fmt "cmp%s %a, $%Ld" (Sz.name sz) pv a k
  | Cmps (p, fsz, a, b) ->
      Fmt.pf fmt "cmp%s%s %a" (Cmp_pred.name p) (Fsz.name fsz) vs [ a; b ]
  | Cqo_idiv (a, b) -> Fmt.pf fmt "cqo; idivq %a" vs [ a; b ]
  | Cvt (fsz, a) -> Fmt.pf fmt "cvt.%s %a" (Fsz.name fsz) pv a
  | Cvtpd2ps a -> Fmt.pf fmt "cvtpd2ps %a" pv a
  | Cvtps2pd a -> Fmt.pf fmt "cvtps2pd %a" pv a
  | Cvtsi2s (fsz, a) -> Fmt.pf fmt "cvtsi2%sq %a" (Fsz.name fsz) pv a
  | Cvtts2si (fsz, a) -> Fmt.pf fmt "cvtt%s2siq %a" (Fsz.name fsz) pv a
  | Fbin (o, fsz, a, b) ->
      Fmt.pf fmt "%s%s %a" (Fop.name o) (Fsz.name fsz) vs [ a; b ]
  | Flogic (o, fsz, a, b) ->
      Fmt.pf fmt "%s%s %a" (Flogic.name o)
        (match fsz with Fsz.D -> "d" | Fsz.S -> "s")
        vs [ a; b ]
  | Ext { signed; from; src } ->
      Fmt.pf fmt "mov%s%sl %a"
        (if signed then "s" else "z")
        (match from with Mir_width.W8 -> "b" | _ -> "w")
        pv src
  | Fmadd231 (fsz, a, b, c) ->
      Fmt.pf fmt "vfmadd231%s %a" (Fsz.name fsz) vs [ a; b; c ]
  | Imul (sz, a, b) -> Fmt.pf fmt "imul%s %a" (Sz.name sz) vs [ a; b ]
  | Imul_imm (sz, a, k) -> Fmt.pf fmt "imul%s %a, $%Ld" (Sz.name sz) pv a k
  | Lea (a, k) -> Fmt.pf fmt "leaq %Ld(%a)" k pv a
  | Lea_view view -> Fmt.pf fmt "leaq %a(%%rip)" Mir_id.View.pp view
  | Load (m, a) -> Fmt.pf fmt "load.%s %a" (Msz.name m) (pp_addr pv) a
  | Mov (sz, a) -> Fmt.pf fmt "mov%s %a" (Sz.name sz) pv a
  | Mov_imm (t, k) -> Fmt.pf fmt "mov.%a $%Ld" Mir_type.pp t k
  | Movap a -> Fmt.pf fmt "movap %a" pv a
  | Movap_to_mask (_, a) -> Fmt.pf fmt "movap.mask %a" pv a
  | Movlhps (a, b) -> Fmt.pf fmt "movlhps %a" vs [ a; b ]
  | Movq_low a -> Fmt.pf fmt "movq.low %a" pv a
  | Movq_widen a -> Fmt.pf fmt "movq.widen %a" pv a
  | Movup_load (pk, a) -> Fmt.pf fmt "movu%s %a" (Pk.name pk) (pp_addr pv) a
  | Movup_store (pk, a, x) ->
      Fmt.pf fmt "movu%s %a, %a" (Pk.name pk) pv x (pp_addr pv) a
  | Movq_from_gpr (fsz, a) ->
      Fmt.pf fmt "mov%s.from_gpr %a"
        (match fsz with Fsz.D -> "q" | Fsz.S -> "d")
        pv a
  | Movq_to_gpr (fsz, a) ->
      Fmt.pf fmt "mov%s.to_gpr %a"
        (match fsz with Fsz.D -> "q" | Fsz.S -> "d")
        pv a
  | Movsxd a -> Fmt.pf fmt "movslq %a" pv a
  | Movzx32 a -> Fmt.pf fmt "movl.zx %a" pv a
  | Neg (sz, a) -> Fmt.pf fmt "neg%s %a" (Sz.name sz) pv a
  | Pbin (o, pk, a, b) ->
      Fmt.pf fmt "%s%s %a" (Fop.name o) (Pk.name pk) vs [ a; b ]
  | Pfmadd231 (pk, a, b, c) ->
      Fmt.pf fmt "vfmadd231%s %a" (Pk.name pk) vs [ a; b; c ]
  | Pcmp (p, pk, a, b) ->
      Fmt.pf fmt "cmp%s%s %a" (Cmp_pred.name p) (Pk.name pk) vs [ a; b ]
  | Plogic (o, pk, a, b) ->
      Fmt.pf fmt "%s%s %a" (Flogic.name o)
        (match pk with Pk.Pd -> "d" | Pk.Ps -> "s")
        vs [ a; b ]
  | Pones pk -> Fmt.pf fmt "pcmpeqd.ones%s" (Pk.name pk)
  | Pshuf (pk, k, a, b) ->
      Fmt.pf fmt "shuf%s $0x%x, %a" (Pk.name pk) k vs [ a; b ]
  | Pshufd_half a -> Fmt.pf fmt "pshufd $0xee, %a" pv a
  | Pshufd_lane (fsz, k, a) ->
      Fmt.pf fmt "pshufd.%s[%d] %a" (Fsz.name fsz) k pv a
  | Pshufd_splat (pk, a) -> Fmt.pf fmt "pshufd.splat%s %a" (Pk.name pk) pv a
  | Pshufd_splat_mask (pk, a) ->
      Fmt.pf fmt "pshufd.splat_mask%s %a" (Pk.name pk) pv a
  | Psqrt (pk, a) -> Fmt.pf fmt "sqrt%s %a" (Pk.name pk) pv a
  | Round_trunc (fsz, a) -> Fmt.pf fmt "round%s $3, %a" (Fsz.name fsz) pv a
  | Setcc_zx (c, f) -> Fmt.pf fmt "set%s+movzbl %a" (Cond.name c) pv f
  | Shift_imm (o, sz, a, k) ->
      Fmt.pf fmt "%s%s $%d, %a" (Shift.name o) (Sz.name sz) k pv a
  | Sqrt (fsz, a) -> Fmt.pf fmt "sqrt%s %a" (Fsz.name fsz) pv a
  | Store (m, a, x) ->
      Fmt.pf fmt "store.%s %a, %a" (Msz.name m) pv x (pp_addr pv) a
  | Test (sz, a, b) -> Fmt.pf fmt "test%s %a" (Sz.name sz) vs [ a; b ]
  | Trunc32 a -> Fmt.pf fmt "movl.trunc %a" pv a
  | Trunc_zx (w, a) ->
      Fmt.pf fmt "movz%sl.trunc %a"
        (match w with Mir_width.W8 -> "b" | _ -> "w")
        pv a
  | Ucomis (fsz, a, b) -> Fmt.pf fmt "ucomi%s %a" (Fsz.name fsz) vs [ a; b ]

let pp_test pv fmt = function
  | Jcc (c, f) -> Fmt.pf fmt "j%s %a" (Cond.name c) pv f

let name = "x86_64"

let source =
  { Mir_target.Source.document = "Intel SDM 325462"; revision = "084" }
