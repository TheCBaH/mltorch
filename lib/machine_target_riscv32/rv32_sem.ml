(* The semantics of each admitted form, from the RISC-V unprivileged ISA
   (20240411): integer operations wrap at 32 bits; DIV by zero gives -1 and
   REM the dividend, the most negative number divided by -1 gives itself and
   remainder 0 (M, "Division Operations"). binary32 results round to nearest
   even (frm is 0); a NaN result is the canonical NaN 0x7fc00000; FMAX and FMIN
   are maximumNumber and minimumNumber, -0 below +0; FEQ, FLT and FLE are 0 on
   a NaN; FCVT.W.S toward zero saturates and gives 2^31 - 1 for a NaN
   (F, "Single-Precision Floating-Point Conversion and Move Instructions"). *)

open Machine_ir
open Machine_interp
open Rv32_op
module E = Mir_sel_env
module N = Mir_numeric

let m32 x = Int64.logand x 0xFFFF_FFFFL
let s32 x = Int64.of_int32 (Int64.to_int32 x)
let bits env v = E.bits env v
let b x = Mir_datum.Bits x
let canonical_nan = 0x7FC0_0000L

(* A binary32 result: rounded once, a NaN made canonical. *)
let fres x = if Float.is_nan x then canonical_nan else N.round32 x
let fval env v = N.f32 (bits env v)
let fbits x = if Float.is_nan (N.f32 x) then canonical_nan else x

let ptr_add env (p : Mir_memory.Pointer.t) delta =
  match Mir_memory.offset_by p delta with
  | Some q -> Mir_datum.Ptr q
  | None -> env.E.defect Mir_observation.Defect.Bad_access

let addr env base k =
  match Mir_memory.offset_by (E.ptr env base) k with
  | Some p -> p
  | None -> env.E.defect Mir_observation.Defect.Bad_access

(* An operand of an addition: a pointer stays one, an integer is its signed
   32-bit reading (a pointer plus a word wraps modulo 2^32, which equals the
   signed offset for every address inside an object). *)
let add env x y =
  match (env.E.get x, env.E.get y) with
  | Mir_datum.Ptr p, Mir_datum.Bits d | Mir_datum.Bits d, Mir_datum.Ptr p ->
      ptr_add env p (s32 d)
  | Mir_datum.Bits a, Mir_datum.Bits c -> b (m32 (Int64.add a c))
  | _ -> env.E.defect Mir_observation.Defect.Invalid_program

let div n d =
  let n = s32 n and d = s32 d in
  if Int64.equal d 0L then -1L
  else if Int64.equal n (-0x8000_0000L) && Int64.equal d (-1L) then n
  else Int64.div n d

let rem n d =
  let n' = s32 n and d' = s32 d in
  if Int64.equal d' 0L then n'
  else if Int64.equal n' (-0x8000_0000L) && Int64.equal d' (-1L) then 0L
  else Int64.rem n' d'

let alu (o : Alu.t) x y =
  m32
    (match o with
    | Alu.Add -> Int64.add x y
    | Alu.And -> Int64.logand x y
    | Alu.Div -> div x y
    | Alu.Mul -> Int64.mul x y
    | Alu.Mulhu -> Int64.shift_right_logical (Int64.mul (m32 x) (m32 y)) 32
    | Alu.Or -> Int64.logor x y
    | Alu.Rem -> rem x y
    | Alu.Sll -> Int64.shift_left x (Int64.to_int (Int64.logand y 31L))
    | Alu.Slt -> if Int64.compare (s32 x) (s32 y) < 0 then 1L else 0L
    | Alu.Sltu -> if Int64.unsigned_compare (m32 x) (m32 y) < 0 then 1L else 0L
    | Alu.Srl ->
        Int64.shift_right_logical (m32 x) (Int64.to_int (Int64.logand y 31L))
    | Alu.Sub -> Int64.sub x y
    | Alu.Xor -> Int64.logxor x y)

(* [mulhu]'s 64-bit product of two 32-bit words fits [Int64]'s 64 bits as an
   unsigned number, and the logical shift reads its high word. *)

let alui (o : Alui.t) x k =
  m32
    (match o with
    | Alui.Andi -> Int64.logand x k
    | Alui.Ori -> Int64.logor x k
    | Alui.Slli -> Int64.shift_left x (Int64.to_int k)
    | Alui.Slti -> if Int64.compare (s32 x) k < 0 then 1L else 0L
    | Alui.Sltiu ->
        (* the immediate is sign-extended, then compared unsigned *)
        if Int64.unsigned_compare (m32 x) (m32 k) < 0 then 1L else 0L
    | Alui.Srai -> Int64.shift_right (s32 x) (Int64.to_int k)
    | Alui.Srli -> Int64.shift_right_logical (m32 x) (Int64.to_int k)
    | Alui.Xori -> Int64.logxor x k)

let fop (o : Fop.t) x y =
  match o with
  | Fop.Add -> fres (x +. y)
  | Fop.Div -> fres (x /. y)
  | Fop.Mul -> fres (x *. y)
  | Fop.Sub -> fres (x -. y)
  | Fop.Max | Fop.Min -> (
      match (Float.is_nan x, Float.is_nan y) with
      | true, true -> canonical_nan
      | true, false -> N.round32 y
      | false, true -> N.round32 x
      | false, false ->
          let max = o = Fop.Max in
          if x = 0. && y = 0. then
            (* -0 is below +0 *)
            let neg = Float.sign_bit x in
            let pick_x = if max then not neg else neg in
            N.round32 (if pick_x then x else y)
          else if max then N.round32 (if x > y then x else y)
          else N.round32 (if x < y then x else y))

let sgnj (o : Sgnj.t) x y =
  let sign = 0x8000_0000L in
  let mag = Int64.logand x 0x7FFF_FFFFL in
  let s =
    match o with
    | Sgnj.J -> Int64.logand y sign
    | Sgnj.Jn -> Int64.logand (Int64.lognot y) sign
    | Sgnj.Jx -> Int64.logand (Int64.logxor x y) sign
  in
  Int64.logor mag s

let fcvt_w_s x =
  if Float.is_nan x then 0x7FFF_FFFFL
  else if x >= 2147483648. then 0x7FFF_FFFFL
  else if x < -2147483648. then 0x8000_0000L
  else m32 (Int64.of_float x)

let pick env p x y =
  match bits env p with
  | 0L -> env.E.get y
  | 1L -> env.E.get x
  | _ -> env.E.defect Mir_observation.Defect.Invalid_program

let load env base k m =
  E.load env (addr env base k) ~bytes:(Msz.bytes m) ~align:1L

(* {1 Vectors}

   A vector's lanes are the 32-bit words of its first 128 bits; a mask lane
   is all ones or zero. Each lane is the scalar operation on the lanes. *)

let lanes env v =
  match env.E.get v with
  | Mir_datum.Lanes l -> l
  | Mir_datum.Bits _ | Mir_datum.Flags _ | Mir_datum.Order | Mir_datum.Ptr _ ->
      env.E.defect Mir_observation.Defect.Invalid_program

let vec l = Mir_datum.Lanes l
let lanewise env f a = vec (Array.map f (lanes env a))
let lanewise2 env f a b = vec (Array.map2 f (lanes env a) (lanes env b))
let ones = 0xFFFF_FFFFL

(* The lane count of a vector type a form names. *)
let count_of t =
  match Rv32_vop.lanes_of t with
  | Some n -> Mir_type.Lanes.to_int n
  | None -> invalid_arg "Rv32_sem: a form's lane type"

let vexec env (op : Rv32_vop.t) =
  let module V = Rv32_vop in
  match op with
  | V.Vcast (_, a) | V.Vmv a -> [ vec (lanes env a) ]
  | V.Vext { signed; from; first; src } ->
      let l = lanes env src in
      [
        vec
          (Array.init 4 (fun j ->
               let x = Mir_width.normalize from l.(first + j) in
               if signed then m32 (Mir_width.signed from x) else x));
      ]
  | V.Vle_int (w, base) ->
      let size = Mir_width.bytes w in
      [
        vec
          (Array.init
             (16 / Int64.to_int size)
             (fun j ->
               E.load env
                 (addr env base (Int64.mul (Int64.of_int j) size))
                 ~bytes:size ~align:1L));
      ]
  | V.Vshift (o, k, a) ->
      [
        lanewise env
          (fun x ->
            match o with
            | Rv32_kind.Shift.Sll -> m32 (Int64.shift_left x k)
            | Rv32_kind.Shift.Srl -> Int64.shift_right_logical (m32 x) k
            | Rv32_kind.Shift.Sra -> m32 (Int64.shift_right (s32 x) k))
          a;
      ]
  | V.Vfbin (o, a, b) ->
      [ lanewise2 env (fun x y -> fop o (N.f32 x) (N.f32 y)) a b ]
  | V.Vfcmp (o, a, b) ->
      [
        lanewise2 env
          (fun x y ->
            let x = N.f32 x and y = N.f32 y in
            let holds =
              match o with
              | Fcmp.Eq -> x = y
              | Fcmp.Le -> x <= y
              | Fcmp.Lt -> x < y
            in
            if holds then ones else 0L)
          a b;
      ]
  | V.Vfcvt_f_x a -> [ lanewise env (fun x -> N.s64_to_f32 (s32 x)) a ]
  | V.Vfcvt_x_f a -> [ lanewise env (fun x -> fcvt_w_s (N.f32 x)) a ]
  | V.Vfmacc (acc, a, b) ->
      let c = lanes env acc and x = lanes env a and y = lanes env b in
      [
        vec
          (Array.init (Array.length c) (fun k ->
               fbits (N.fma32 x.(k) y.(k) c.(k))));
      ]
  | V.Vfmv_f_s (k, a) -> [ b (lanes env a).(k) ]
  | V.Vfmv_v_f (t, a) -> [ vec (Array.make (count_of t) (m32 (bits env a))) ]
  | V.Vfsqrt a -> [ lanewise env (fun x -> fres (Float.sqrt (N.f32 x))) a ]
  | V.Vinsert (k, a, x) ->
      let l = Array.copy (lanes env a) in
      l.(k) <- m32 (bits env x);
      [ vec l ]
  | V.Vle32 (t, base) ->
      [
        vec
          (Array.init (count_of t) (fun k ->
               load env base (Int64.of_int (4 * k)) Msz.W));
      ]
  | V.Vlogic (o, a, c) ->
      [
        lanewise2 env
          (fun x y ->
            match o with
            | Rv32_kind.Logic.And -> Int64.logand x y
            | Rv32_kind.Logic.Or -> Int64.logor x y
            | Rv32_kind.Logic.Xor -> Int64.logxor x y)
          a c;
      ]
  | V.Vlse32 (t, base, stride) ->
      let s = s32 (bits env stride) in
      [
        vec
          (Array.init (count_of t) (fun k ->
               load env base (Int64.mul (Int64.of_int k) s) Msz.W));
      ]
  | V.Vmerge (other, a, m) ->
      let e = lanes env other and t = lanes env a and mk = lanes env m in
      [
        vec
          (Array.init (Array.length t) (fun k ->
               if Int64.equal mk.(k) 0L then e.(k) else t.(k)));
      ]
  | V.Vmv_v_x (t, a) -> [ vec (Array.make (count_of t) (m32 (bits env a))) ]
  | V.Vnot a -> [ lanewise env (fun x -> Int64.logxor x ones) a ]
  | V.Vse32 (base, x) ->
      Array.iteri
        (fun k l ->
          E.store env
            (addr env base (Int64.of_int (4 * k)))
            ~bytes:4L ~align:1L l)
        (lanes env x);
      []
  | V.Vsgnj (o, a, c) -> [ lanewise2 env (sgnj o) a c ]
  | V.Vsse32 (base, stride, x) ->
      let s = s32 (bits env stride) in
      Array.iteri
        (fun k l ->
          E.store env
            (addr env base (Int64.mul (Int64.of_int k) s))
            ~bytes:4L ~align:1L l)
        (lanes env x);
      []

let exec env op =
  match op with
  | Add_wide (x, k) | Addi (x, k) -> (
      match env.E.get x with
      | Mir_datum.Ptr p -> [ ptr_add env p k ]
      | Mir_datum.Bits a -> [ b (m32 (Int64.add a k)) ]
      | _ -> env.E.defect Mir_observation.Defect.Invalid_program)
  | Alu (Alu.Add, x, y) -> [ add env x y ]
  | Alu (o, x, y) -> [ b (alu o (bits env x) (bits env y)) ]
  | Alui (o, x, k) -> [ b (alui o (bits env x) k) ]
  | Call { callee; args; results } ->
      let rs = env.E.call callee (List.map env.E.get args) in
      if List.length rs <> List.length results + 1 then
        env.E.defect Mir_observation.Defect.Invalid_program;
      rs
  | Fbin (o, x, y) -> [ b (fop o (fval env x) (fval env y)) ]
  | Fcmp (o, x, y) ->
      let x = fval env x and y = fval env y in
      let holds =
        match o with Fcmp.Eq -> x = y | Fcmp.Le -> x <= y | Fcmp.Lt -> x < y
      in
      [ b (if holds then 1L else 0L) ]
  | Fcvt_s_w x -> [ b (N.s64_to_f32 (s32 (bits env x))) ]
  | Fcvt_s_wu x -> [ b (N.s64_to_f32 (m32 (bits env x))) ]
  | Fcvt_w_s x -> [ b (fcvt_w_s (fval env x)) ]
  | Flw (base, k) -> [ b (load env base k Msz.W) ]
  | Fmadd (x, y, z) ->
      [ b (fbits (N.fma32 (bits env x) (bits env y) (bits env z))) ]
  | Fmv_w_x x | Fmv_x_w x -> [ b (m32 (bits env x)) ]
  | Fsel (p, x, y) | Sel (p, x, y) -> [ pick env p x y ]
  | Fsgnj (o, x, y) -> [ b (sgnj o (bits env x) (bits env y)) ]
  | Fsqrt x -> [ b (fres (Float.sqrt (fval env x))) ]
  | Fsw (base, k, x) ->
      E.store env (addr env base k) ~bytes:4L ~align:1L (bits env x);
      []
  | La view -> (
      match env.E.view view with
      | Some p -> [ Mir_datum.Ptr p ]
      | None -> env.E.defect Mir_observation.Defect.Invalid_program)
  | Li (_, k) -> [ b (m32 k) ]
  | Load (m, base, k) -> [ b (load env base k m) ]
  | Lui k -> [ b (m32 (Int64.shift_left k 12)) ]
  | Mv x -> [ env.E.get x ]
  | Sext (w, x) ->
      [ b (m32 (Mir_width.signed w (Mir_width.normalize w (bits env x)))) ]
  | Store (m, base, k, x) ->
      E.store env (addr env base k) ~bytes:(Msz.bytes m) ~align:1L
        (Int64.logand (bits env x) (Mir_width.mask (Msz.width m)));
      []
  | Sw_pair (base, k, lo, hi) ->
      E.store env (addr env base k) ~bytes:4L ~align:1L (bits env lo);
      E.store env
        (addr env base (Int64.add k 4L))
        ~bytes:4L ~align:1L (bits env hi);
      []
  | Trunc (w, x) -> [ b (Mir_width.normalize w (bits env x)) ]
  | Vec x -> vexec env x
  | Zext x -> [ b (bits env x) ]

let test env = function
  | Bcmp (c, x, y) -> (
      let x = bits env x and y = bits env y in
      match c with
      | Bcond.Eq -> Int64.equal x y
      | Bcond.Ne -> not (Int64.equal x y)
      | Bcond.Lt -> Int64.compare (s32 x) (s32 y) < 0
      | Bcond.Ge -> Int64.compare (s32 x) (s32 y) >= 0
      | Bcond.Ltu -> Int64.unsigned_compare (m32 x) (m32 y) < 0
      | Bcond.Geu -> Int64.unsigned_compare (m32 x) (m32 y) >= 0)
  | Beqz x -> Int64.equal (bits env x) 0L
  | Bnez x -> not (Int64.equal (bits env x) 0L)
