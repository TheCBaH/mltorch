(* The integer, address and binary32 sequences selection builds from forms:
   constants, compares and arithmetic of words and of i64 pairs, folded
   displacements, IEEE maximum and truncation. *)

open Machine_ir
open Rv32_op
open Rv32_select_st

(* {1 Constants} *)

let s32 x = Int64.of_int32 (Int64.to_int32 x)

(* A 32-bit word into a register of [t]: [li] for a 12-bit signed value, else
   [lui] and an [addi] of the low twelve bits, sign-adjusted. *)
let materialize st ?result (t : Mir_type.t) bits =
  let s = s32 bits in
  if imm12 s then emit st ?result t (Li (t, s))
  else
    let low = Int64.sub (Int64.logxor (Int64.logand s 0xFFFL) 0x800L) 0x800L in
    let high = Int64.logand (Int64.shift_right (Int64.sub s low) 12) 0xFFFFFL in
    let word () =
      let l = emit st Mir_type.i32 (Lui high) in
      if Int64.equal low 0L then l else emit st Mir_type.i32 (Addi (l, low))
    in
    match t with
    | Mir_type.Int Mir_width.W32 -> (
        match (result, Int64.equal low 0L) with
        | Some _, true -> emit st ?result t (Lui high)
        | _, true -> emit st t (Lui high)
        | _ ->
            let l = emit st Mir_type.i32 (Lui high) in
            emit st ?result t (Addi (l, low)))
    | Mir_type.Int ((Mir_width.W8 | Mir_width.W16) as w) ->
        emit st ?result t (Trunc (w, word ()))
    | _ -> refuse st (Refusal.Width t)

let const st ?result (c : Mir_const.t) =
  match c.Mir_const.ty with
  | Mir_type.Int (Mir_width.W8 | Mir_width.W16 | Mir_width.W32) ->
      ignore (materialize st ?result c.Mir_const.ty c.Mir_const.bits)
  | Mir_type.Pred ->
      ignore
        (emit st ?result Mir_type.Pred (Li (Mir_type.Pred, c.Mir_const.bits)))
  | Mir_type.F32 ->
      let g = materialize st Mir_type.i32 c.Mir_const.bits in
      ignore (emit st ?result Mir_type.F32 (Fmv_w_x g))
  | t -> refuse st (Refusal.Width t)

(* A constant i64's halves into [p]. *)
let const_pair st (p : pair) bits =
  ignore
    (materialize st ~result:p.lo Mir_type.i32 (Int64.logand bits 0xFFFF_FFFFL));
  ignore
    (materialize st ~result:p.hi Mir_type.i32
       (Int64.shift_right_logical bits 32))

(* {1 Integer helpers} *)

let alu st ?result o a b =
  emit st ?result
    (if Alu.compares o then Mir_type.Pred else ty a)
    (Alu (o, a, b))

let alui st ?result o a k =
  let t = if Alui.compares o then Mir_type.Pred else ty a in
  emit st ?result t (Alui (o, a, k))

let pnot st ?result p = emit st ?result Mir_type.Pred (Alui (Alui.Xori, p, 1L))

(* [a op c] with a constant operand the form encodes ([ok]): the other
   operand and the constant, which is [c] unless the operation [commutes]. *)
let immediate st ~commutes ~ok a c =
  let imm v =
    match const_of st v with
    | Some k when ok (s32 k) -> Some (s32 k)
    | _ -> None
  in
  match imm c with
  | Some k -> Some (a, k)
  | None when commutes -> Option.map (fun k -> (c, k)) (imm a)
  | None -> None

let log2 k =
  List.find_opt
    (fun n -> Int64.equal k (Int64.shift_left 1L n))
    (List.init 31 Fun.id)

(* An i32 operation, immediates folded. *)
let iarith32 st ?result (o : Mir_op.Iarith.t) a b =
  let shift k =
    match const_of st b with
    | Some c -> alui st ?result k a (Int64.logand c 31L)
    | None -> refuse st (Refusal.Operation "shift by a variable amount")
  in
  let logic r i =
    match immediate st ~commutes:true ~ok:imm12 a b with
    | Some (x, k) -> alui st ?result i x k
    | None -> alu st ?result r a b
  in
  match o with
  | Mir_op.Iarith.Add -> (
      match immediate st ~commutes:true ~ok:imm12 a b with
      | Some (x, k) -> emit st ?result (ty a) (Addi (x, k))
      | None -> alu st ?result Alu.Add a b)
  | Mir_op.Iarith.And -> logic Alu.And Alui.Andi
  | Mir_op.Iarith.Mul -> (
      match
        immediate st ~commutes:true ~ok:(fun k -> Option.is_some (log2 k)) a b
      with
      | Some (x, k) ->
          alui st ?result Alui.Slli x (Int64.of_int (Option.get (log2 k)))
      | None -> alu st ?result Alu.Mul a b)
  | Mir_op.Iarith.Or -> logic Alu.Or Alui.Ori
  | Mir_op.Iarith.Shl -> shift Alui.Slli
  | Mir_op.Iarith.Shr_s -> shift Alui.Srai
  | Mir_op.Iarith.Shr_u -> shift Alui.Srli
  | Mir_op.Iarith.Sub -> (
      let neg_imm k = imm12 (Int64.neg k) in
      match immediate st ~commutes:false ~ok:neg_imm a b with
      | Some (x, k) -> emit st ?result (ty a) (Addi (x, Int64.neg k))
      | None -> alu st ?result Alu.Sub a b)
  | Mir_op.Iarith.Xor -> logic Alu.Xor Alui.Xori

(* An i8 or i16 value as the i32 it is, sign- or zero-extended. *)
let widen st ~signed (v : Mir_value.t) =
  match ty v with
  | Mir_type.Int Mir_width.W32 -> v
  | Mir_type.Int ((Mir_width.W8 | Mir_width.W16) as w) ->
      emit st Mir_type.i32 (if signed then Sext (w, v) else Zext v)
  | Mir_type.Pred -> emit st Mir_type.i32 (Zext v)
  | t -> refuse st (Refusal.Width t)

(* [a == b] and the other compares of two i32 words, as a predicate. *)
let compare32 st ?result (c : Mir_op.Icmp.t) a b =
  let eq ?result () =
    match const_of st b with
    | Some k when Int64.equal (s32 k) 0L -> alui st ?result Alui.Sltiu a 1L
    | _ ->
        let x =
          match immediate st ~commutes:true ~ok:imm12 a b with
          | Some (x, k) -> alui st Alui.Xori x k
          | None -> alu st Alu.Xor a b
        in
        alui st ?result Alui.Sltiu x 1L
  in
  let lt o oi ?result x y =
    match immediate st ~commutes:false ~ok:imm12 x y with
    | Some (x, k) -> alui st ?result oi x k
    | None -> alu st ?result o x y
  in
  match c with
  | Mir_op.Icmp.Eq -> eq ?result ()
  | Mir_op.Icmp.Ne -> pnot st ?result (eq ())
  | Mir_op.Icmp.Slt -> lt Alu.Slt Alui.Slti ?result a b
  | Mir_op.Icmp.Ult -> lt Alu.Sltu Alui.Sltiu ?result a b
  | Mir_op.Icmp.Sle -> pnot st ?result (alu st Alu.Slt b a)
  | Mir_op.Icmp.Ule -> pnot st ?result (alu st Alu.Sltu b a)

(* The same compare of two i64 pairs: the high words decide unless equal. *)
let compare64 st ?result (c : Mir_op.Icmp.t) (a : pair) (b : pair) =
  if a.sext && b.sext then compare32 st ?result c a.lo b.lo
  else
    let eq ?result () =
      let o =
        alu st Alu.Or (alu st Alu.Xor a.lo b.lo) (alu st Alu.Xor a.hi b.hi)
      in
      alui st ?result Alui.Sltiu o 1L
    in
    let lt ~signed ?result (x : pair) (y : pair) =
      let high = alu st (if signed then Alu.Slt else Alu.Sltu) x.hi y.hi in
      let same = alui st Alui.Sltiu (alu st Alu.Xor x.hi y.hi) 1L in
      let low =
        alu st
          (if mutated st Mutation.Signed_compare then Alu.Slt else Alu.Sltu)
          x.lo y.lo
      in
      alu st ?result Alu.Or high (alu st Alu.And same low)
    in
    match c with
    | Mir_op.Icmp.Eq -> eq ?result ()
    | Mir_op.Icmp.Ne -> pnot st ?result (eq ())
    | Mir_op.Icmp.Slt -> lt ~signed:true ?result a b
    | Mir_op.Icmp.Ult -> lt ~signed:false ?result a b
    | Mir_op.Icmp.Sle -> pnot st ?result (lt ~signed:true b a)
    | Mir_op.Icmp.Ule -> pnot st ?result (lt ~signed:false b a)

(* An i64 operation on pairs, into the result's halves [r]. *)
let iarith64 st (o : Mir_op.Iarith.t) (a : pair) (b_v : Mir_value.t) (r : pair)
    =
  let b () = pair st b_v in
  let set_lo op = ignore (emit st ~result:r.lo Mir_type.i32 op)
  and set_hi op = ignore (emit st ~result:r.hi Mir_type.i32 op) in
  let mv_lo v = set_lo (Mv v) and mv_hi v = set_hi (Mv v) in
  let shift_count () =
    match const_of st b_v with
    | Some c -> Int64.to_int (Int64.logand c 63L)
    | None -> refuse st (Refusal.Operation "shift by a variable amount")
  in
  match o with
  | Mir_op.Iarith.Add ->
      let b = b () in
      set_lo (Alu (Alu.Add, a.lo, b.lo));
      let h = alu st Alu.Add a.hi b.hi in
      if mutated st Mutation.Dropped_carry then mv_hi h
      else set_hi (Alu (Alu.Add, h, alu st Alu.Sltu r.lo a.lo))
  | Mir_op.Iarith.Sub ->
      let b = b () in
      set_lo (Alu (Alu.Sub, a.lo, b.lo));
      let h = alu st Alu.Sub a.hi b.hi in
      set_hi (Alu (Alu.Sub, h, alu st Alu.Sltu a.lo b.lo))
  | Mir_op.Iarith.Mul ->
      let b = b () in
      set_lo (Alu (Alu.Mul, a.lo, b.lo));
      let h = alu st Alu.Mulhu a.lo b.lo in
      let c1 = alu st Alu.Mul a.lo b.hi and c2 = alu st Alu.Mul a.hi b.lo in
      set_hi (Alu (Alu.Add, alu st Alu.Add h c1, c2))
  | Mir_op.Iarith.And | Mir_op.Iarith.Or | Mir_op.Iarith.Xor ->
      let b = b () in
      let o =
        match o with
        | Mir_op.Iarith.And -> Alu.And
        | Mir_op.Iarith.Or -> Alu.Or
        | _ -> Alu.Xor
      in
      set_lo (Alu (o, a.lo, b.lo));
      set_hi (Alu (o, a.hi, b.hi))
  | Mir_op.Iarith.Shl ->
      let k = Int64.of_int (shift_count ()) in
      if Int64.equal k 0L then (
        mv_lo a.lo;
        mv_hi a.hi)
      else if Int64.compare k 32L < 0 then (
        set_lo (Alui (Alui.Slli, a.lo, k));
        set_hi
          (Alu
             ( Alu.Or,
               alui st Alui.Slli a.hi k,
               alui st Alui.Srli a.lo (Int64.sub 32L k) )))
      else (
        set_hi (Alui (Alui.Slli, a.lo, Int64.sub k 32L));
        ignore (materialize st ~result:r.lo Mir_type.i32 0L))
  | Mir_op.Iarith.Shr_u | Mir_op.Iarith.Shr_s ->
      let k = Int64.of_int (shift_count ()) in
      let right = if o = Mir_op.Iarith.Shr_s then Alui.Srai else Alui.Srli in
      if Int64.equal k 0L then (
        mv_lo a.lo;
        mv_hi a.hi)
      else if Int64.compare k 32L < 0 then (
        set_lo
          (Alu
             ( Alu.Or,
               alui st Alui.Srli a.lo k,
               alui st Alui.Slli a.hi (Int64.sub 32L k) ));
        set_hi (Alui (right, a.hi, k)))
      else (
        set_lo (Alui (right, a.hi, Int64.sub k 32L));
        if right = Alui.Srai then set_hi (Alui (Alui.Srai, a.hi, 31L))
        else ignore (materialize st ~result:r.hi Mir_type.i32 0L))

(* {1 Addresses} *)

(* An access address as a base and a 12-bit displacement: a pointer plus a
   constant that fits (and [extra] more bytes, for the second word of a pair)
   folds; anything else is its own base. *)
let displaced st ?(extra = 0L) (addr : Mir_value.t) =
  match Hashtbl.find_opt st.b.Mir_select.defs (key addr) with
  | Some (Mir_op.Ptr_add (base, k)) -> (
      match const_of st k with
      | Some c when imm12 (s32 c) && imm12 (Int64.add (s32 c) extra) ->
          (u st base, s32 c)
      | _ -> (u st addr, 0L))
  | _ -> (u st addr, 0L)

(* {1 Floating point} *)

let fsel st ?result p a b = emit st ?result Mir_type.F32 (Fsel (p, a, b))

(* IEEE 754-2019 maximum from maximumNumber: where either operand is a NaN,
   their sum, which is one. *)
let fmax st ?result a b =
  if mutated st Mutation.Number_max then
    emit st ?result Mir_type.F32 (Fbin (Fop.Max, a, b))
  else
    let m = emit st Mir_type.F32 (Fbin (Fop.Max, a, b)) in
    let ordered =
      alu st Alu.And
        (emit st Mir_type.Pred (Fcmp (Fcmp.Eq, a, a)))
        (emit st Mir_type.Pred (Fcmp (Fcmp.Eq, b, b)))
    in
    let nan = emit st Mir_type.F32 (Fbin (Fop.Add, a, b)) in
    fsel st ?result ordered m nan

(* Toward zero: below 2^23 in magnitude through a 32-bit integer and back,
   the operand's sign restored (so -0.5 gives -0); from 2^23 up, and for a
   NaN or an infinity, the operand itself. *)
let ftrunc st ?result x =
  let t =
    emit st Mir_type.F32
      (Fsgnj
         ( Sgnj.J,
           emit st Mir_type.F32 (Fcvt_s_w (emit st Mir_type.i32 (Fcvt_w_s x))),
           x ))
  in
  if mutated st Mutation.Unguarded_trunc then
    emit st ?result Mir_type.F32 (Fsgnj (Sgnj.J, t, x))
  else
    let mag = emit st Mir_type.F32 (Fsgnj (Sgnj.Jx, x, x)) in
    let limit =
      emit st Mir_type.F32 (Fmv_w_x (materialize st Mir_type.i32 0x4B00_0000L))
    in
    let small = emit st Mir_type.Pred (Fcmp (Fcmp.Lt, mag, limit)) in
    fsel st ?result small t x

let f32_only st (v : Mir_value.t) =
  if not (Mir_type.equal (ty v) Mir_type.F32) then
    refuse st (Refusal.Width (ty v))

(* [n] ordered forms in a row, chained on the order [o] threads. *)
let is_vector (v : Mir_value.t) =
  match ty v with Mir_type.Vec _ | Mir_type.Mask _ -> true | _ -> false

(* {1 i64 to binary32}

   One rounding from a full i64, in word arithmetic: the magnitude's top 32
   bits (its high word shifted up by its leading zeros, filled from the low
   word) with every bit below folded into the lowest one, converted unsigned —
   the sticky bit lies seven places under binary32's rounding position, so the
   one rounding is the correct one — then scaled exactly by the 2^s the shift
   took off, and given the sign. A magnitude below 2^32 converts directly. *)
let i64_to_f32 st ?result (p : pair) =
  let k bits = materialize st Mir_type.i32 bits in
  let a o x y = alu st o x y and ai o x c = alui st o x c in
  let add x c = emit st Mir_type.i32 (Addi (x, c)) in
  let word pr = emit st Mir_type.i32 (Zext pr) in
  (* the magnitude, [x xor s - s] with [s] all ones for a negative *)
  let s = ai Alui.Srai p.hi 31L in
  let one = ai Alui.Andi s 1L in
  let ml = a Alu.Add (a Alu.Xor p.lo s) one in
  let mh = a Alu.Add (a Alu.Xor p.hi s) (word (a Alu.Sltu ml one)) in
  (* the high word's leading zeros: a binary search, each step shifting it up
     by 16, 8, 4, 2 or 1 where its top bits are clear *)
  let x = ref mh and zeros = ref (k 0L) in
  List.iter
    (fun (bound, log) ->
      let step = ai Alui.Slli (word (a Alu.Sltu !x (k bound))) log in
      x := a Alu.Sll !x step;
      zeros := a Alu.Add !zeros step)
    [
      (0x0001_0000L, 4L);
      (0x0100_0000L, 3L);
      (0x1000_0000L, 2L);
      (0x4000_0000L, 1L);
      (0x8000_0000L, 0L);
    ];
  (* shifting right by [shift] = 32 - zeros (1..32): the top word is the high
     word up by [zeros] or'd with the low word down by [shift], done as one and
     then [shift] - 1 so that 32 is reachable; the bits shifted out of the low
     word are it up by [zeros] *)
  let shift = a Alu.Sub (k 32L) !zeros in
  let top = a Alu.Or !x (a Alu.Srl (ai Alui.Srli ml 1L) (add shift (-1L))) in
  let lost = a Alu.Sll ml !zeros in
  let sticky = word (pnot st (ai Alui.Sltiu lost 1L)) in
  let large = emit st Mir_type.F32 (Fcvt_s_wu (a Alu.Or top sticky)) in
  (* times 2^shift: the exponent field 127 + shift *)
  let scale =
    emit st Mir_type.F32 (Fmv_w_x (ai Alui.Slli (add shift 127L) 23L))
  in
  let large = emit st Mir_type.F32 (Fbin (Fop.Mul, large, scale)) in
  let small = emit st Mir_type.F32 (Fcvt_s_wu ml) in
  let mag = fsel st (ai Alui.Sltiu mh 1L) small large in
  (* the sign of [s]: its top bit *)
  emit st ?result Mir_type.F32
    (Fsgnj (Sgnj.Jx, mag, emit st Mir_type.F32 (Fmv_w_x s)))
