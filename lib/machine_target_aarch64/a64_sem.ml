(* The semantics of each admitted form, from DDI 0487 K.a pseudocode: W forms
   compute on bits 31:0 and zero-extend; SUBS sets N from the result's top bit,
   Z on zero, C when no borrow, V on signed overflow; FCMP sets NZCV to 0011
   unordered, 0110 equal, 1000 less, 0010 greater; FCVTZS saturates (NaN to 0);
   SDIV by zero gives 0 and min / -1 gives min; FMAX returns a NaN when either
   operand is one and orders -0 below +0. Floating point follows FPCR with
   round-to-nearest-even, no flush-to-zero and default-NaN off. *)

open Machine_ir
open Machine_interp
open A64_op
module E = Mir_sel_env
module N = Mir_numeric

let mask = function Sz.W -> 0xFFFF_FFFFL | Sz.X -> -1L
let norm sz x = Int64.logand x (mask sz)

let signed sz x =
  match sz with Sz.W -> Int64.of_int32 (Int64.to_int32 x) | Sz.X -> x

let bits env v = E.bits env v
let b x = Mir_datum.Bits x

(* NZCV of [a - b] at width [sz]. *)
let subs_flags sz a b =
  let n = match sz with Sz.W -> 32 | Sz.X -> 64 in
  let r = norm sz (Int64.sub a b) in
  let top x =
    not (Int64.equal (Int64.logand (Int64.shift_right_logical x (n - 1)) 1L) 0L)
  in
  let nf = top r and zf = Int64.equal r 0L in
  let cf = Int64.unsigned_compare (norm sz a) (norm sz b) >= 0 in
  let sa = signed sz a and sb = signed sz b in
  let vf =
    top (Int64.logand (Int64.logxor sa sb) (Int64.logxor sa (signed sz r)))
  in
  let bit c v = if c then v else 0L in
  Int64.logor (bit nf 8L)
    (Int64.logor (bit zf 4L) (Int64.logor (bit cf 2L) (bit vf 1L)))

let fcmp_flags x y =
  if Float.is_nan x || Float.is_nan y then 3L
  else if x = y then 6L
  else if x < y then 8L
  else 2L

let flags v = Mir_datum.Flags { bits = v; defined = 15L }

let fval env fsz v =
  let x = bits env v in
  match fsz with Fsz.D -> N.f64 x | Fsz.S -> N.f32 x

let fres fsz x = b (match fsz with Fsz.D -> N.of_f64 x | Fsz.S -> N.round32 x)

let ptr_add env (p : Mir_memory.Pointer.t) delta =
  match Mir_memory.offset_by p delta with
  | Some q -> Mir_datum.Ptr q
  | None -> env.E.defect Mir_observation.Defect.Bad_access

(* An X operand as a number: a pointer is its synthetic address. *)
let number env v =
  match env.E.get v with
  | Mir_datum.Bits x -> x
  | Mir_datum.Ptr p -> Mir_memory.address env.E.memory p
  | Mir_datum.Flags _ | Mir_datum.Lanes _ | Mir_datum.Order ->
      env.E.defect Mir_observation.Defect.Invalid_program

let addr env base k =
  match Mir_memory.offset_by (E.ptr env base) k with
  | Some p -> p
  | None -> env.E.defect Mir_observation.Defect.Bad_access

let shift_imm sz o x k =
  match (o : Shift.t) with
  | Shift.Lsl -> norm sz (Int64.shift_left x k)
  | Shift.Lsr -> Int64.shift_right_logical (norm sz x) k
  | Shift.Asr -> norm sz (Int64.shift_right (signed sz x) k)

let fcvtzs x =
  if Float.is_nan x then 0L
  else if x >= 9223372036854775808. then Int64.max_int
  else if x < -9223372036854775808. then Int64.min_int
  else Int64.of_float x

let logic (o : Logic.t) sz x y =
  norm sz
    (match o with
    | Logic.And -> Int64.logand x y
    | Logic.Eor -> Int64.logxor x y
    | Logic.Orr -> Int64.logor x y)

(* A vector operand's lanes. *)
let lanes env v =
  match env.E.get v with
  | Mir_datum.Lanes l -> l
  | Mir_datum.Bits _ | Mir_datum.Flags _ | Mir_datum.Order | Mir_datum.Ptr _ ->
      env.E.defect Mir_observation.Defect.Invalid_program

let vec l = Mir_datum.Lanes l
let lane_bytes = function Fsz.D -> 8L | Fsz.S -> 4L

(* A binary32 or binary64 lane's value, and a result rounded to its lane. *)
let lval fsz x = match fsz with Fsz.D -> N.f64 x | Fsz.S -> N.f32 x
let lres fsz x = match fsz with Fsz.D -> N.of_f64 x | Fsz.S -> N.round32 x

let fop (o : Fop.t) x y =
  match o with
  | Fop.Add -> x +. y
  | Fop.Div -> x /. y
  | Fop.Max -> N.fmax x y
  | Fop.Mul -> x *. y
  | Fop.Sub -> x -. y

let funary fsz (u : Funary.t) x =
  match u with
  | Funary.Fneg -> (
      match fsz with
      | Fsz.D -> Int64.logxor x Int64.min_int
      | Fsz.S -> Int64.logxor x 0x8000_0000L)
  | Funary.Frintz -> lres fsz (Float.trunc (lval fsz x))
  | Funary.Fsqrt -> lres fsz (Float.sqrt (lval fsz x))

let exec env op =
  let get = env.E.get in
  match op with
  | Add (sz, x, y) -> (
      match (get x, get y) with
      | Mir_datum.Ptr p, Mir_datum.Bits d | Mir_datum.Bits d, Mir_datum.Ptr p ->
          [ ptr_add env p d ]
      | Mir_datum.Bits a, Mir_datum.Bits c -> [ b (norm sz (Int64.add a c)) ]
      | _ -> env.E.defect Mir_observation.Defect.Invalid_program)
  | Add_imm (sz, x, k) -> (
      match get x with
      | Mir_datum.Ptr p -> [ ptr_add env p k ]
      | Mir_datum.Bits a -> [ b (norm sz (Int64.add a k)) ]
      | _ -> env.E.defect Mir_observation.Defect.Invalid_program)
  | Add_lo12 (x, view) -> (
      match env.E.view view with
      | Some v ->
          [
            ptr_add env (E.ptr env x)
              (Int64.logand v.Mir_memory.Pointer.offset 0xFFFL);
          ]
      | None -> env.E.defect Mir_observation.Defect.Invalid_program)
  | Adrp view -> (
      (* regions sit at page-aligned synthetic bases, so the page of the view's
         first byte is its region offset with the low twelve bits cleared *)
      match env.E.view view with
      | Some p ->
          [
            Mir_datum.Ptr
              {
                p with
                Mir_memory.Pointer.offset =
                  Int64.logand p.Mir_memory.Pointer.offset (Int64.lognot 0xFFFL);
              };
          ]
      | None -> env.E.defect Mir_observation.Defect.Invalid_program)
  | Bl { callee; args; results } ->
      let rs = env.E.call callee (List.map get args) in
      if List.length rs <> List.length results + 1 then
        env.E.defect Mir_observation.Defect.Invalid_program;
      rs
  | Cmp (sz, x, y) -> [ flags (subs_flags sz (number env x) (number env y)) ]
  | Cmp_imm (sz, x, k) -> [ flags (subs_flags sz (number env x) k) ]
  | Csel (_, c, f, x, y) | Fcsel (_, c, f, x, y) ->
      [
        (if Cond.holds c (E.flags env f ~mask:(Cond.reads c)) then get x
         else get y);
      ]
  | Dup_elem (arr, x) -> [ vec (Array.make (Arr.lanes arr) (bits env x)) ]
  | Dup_half (k, x) -> [ vec (Array.sub (lanes env x) (2 * k) 2) ]
  | Dup_mask (arr, x) -> [ vec (Array.make (Arr.lanes arr) (bits env x)) ]
  | Dup_lane (_, k, x) -> [ b (lanes env x).(k) ]
  | Ext { signed; from; src } ->
      let k = Mir_width.bits from in
      let x = Int64.logand (bits env src) (Mir_width.mask from) in
      let x =
        if signed then Int64.shift_right (Int64.shift_left x (64 - k)) (64 - k)
        else x
      in
      [ b (norm Sz.W x) ]
  | Cset (c, f) ->
      [
        b (if Cond.holds c (E.flags env f ~mask:(Cond.reads c)) then 1L else 0L);
      ]
  | Fbin (o, fsz, x, y) ->
      let x = fval env fsz x and y = fval env fsz y in
      [
        fres fsz
          (match o with
          | Fop.Add -> x +. y
          | Fop.Div -> x /. y
          | Fop.Max -> N.fmax x y
          | Fop.Mul -> x *. y
          | Fop.Sub -> x -. y);
      ]
  | Fcmp (fsz, x, y) -> [ flags (fcmp_flags (fval env fsz x) (fval env fsz y)) ]
  | Fcvt (Fsz.D, x) -> [ b (N.of_f64 (N.f32 (bits env x))) ]
  | Fcvt (Fsz.S, x) -> [ b (N.round32 (N.f64 (bits env x))) ]
  | Fcvtl x -> [ vec (Array.map (fun l -> N.of_f64 (N.f32 l)) (lanes env x)) ]
  | Fcvtn x -> [ vec (Array.map (fun l -> N.round32 (N.f64 l)) (lanes env x)) ]
  | Fcvtzs (fsz, x) -> [ b (fcvtzs (fval env fsz x)) ]
  | Fmadd (Fsz.D, x, y, z) ->
      [
        b
          (N.of_f64
             (Float.fma
                (N.f64 (bits env x))
                (N.f64 (bits env y))
                (N.f64 (bits env z))));
      ]
  | Fmadd (Fsz.S, x, y, z) ->
      [ b (N.fma32 (bits env x) (bits env y) (bits env z)) ]
  | Fmov (_, x) | Fmov_from_gpr (_, x) | Fmov_to_gpr (_, x) ->
      [ b (bits env x) ]
  | Funary (u, fsz, x) ->
      let v = fval env fsz x in
      [
        (match u with
        | Funary.Fneg -> (
            (* a sign flip on the bits, NaN included *)
            match fsz with
            | Fsz.D -> b (Int64.logxor (bits env x) Int64.min_int)
            | Fsz.S -> b (Int64.logxor (bits env x) 0x8000_0000L))
        | Funary.Frintz -> fres fsz (Float.trunc v)
        | Funary.Fsqrt -> fres fsz (Float.sqrt v));
      ]
  | Ins_half (x, y) ->
      let l = Array.copy (lanes env x) and h = lanes env y in
      l.(2) <- h.(0);
      l.(3) <- h.(1);
      [ vec l ]
  | Ins_lane (_, k, x, y) ->
      let l = Array.copy (lanes env x) in
      l.(k) <- bits env y;
      [ vec l ]
  | Ld1_lane (fsz, k, x, base) ->
      let l = Array.copy (lanes env x) in
      l.(k) <- E.load env (E.ptr env base) ~bytes:(lane_bytes fsz) ~align:1L;
      [ vec l ]
  | Ld1r (arr, base) ->
      let x =
        E.load env (E.ptr env base) ~bytes:(lane_bytes (Arr.fsz arr)) ~align:1L
      in
      [ vec (Array.make (Arr.lanes arr) x) ]
  | Ldr_vec (arr, base, k) ->
      let size = lane_bytes (Arr.fsz arr) in
      [
        vec
          (Array.init (Arr.lanes arr) (fun j ->
               E.load env
                 (addr env base (Int64.add k (Int64.mul (Int64.of_int j) size)))
                 ~bytes:size ~align:1L));
      ]
  | Ldr (m, base, k) ->
      let size = Msz.bytes m in
      (* normal memory permits an unaligned access (DDI 0487 B2.5) *)
      [ b (E.load env (addr env base k) ~bytes:size ~align:1L) ]
  | Logic (o, sz, x, y) -> [ b (logic o sz (bits env x) (bits env y)) ]
  | Logic_imm (o, sz, x, k) -> [ b (logic o sz (bits env x) k) ]
  | Mov (_, x) -> [ get x ]
  | Mrs_fpcr x | Msr_fpcr x -> [ b (bits env x) ]
  | Movk (sz, x, imm, sh) ->
      let keep = Int64.lognot (Int64.shift_left 0xFFFFL sh) in
      [
        b
          (norm sz
             (Int64.logor
                (Int64.logand (bits env x) keep)
                (Int64.shift_left (Int64.of_int imm) sh)));
      ]
  | Movn (sz, imm, sh) ->
      [ b (norm sz (Int64.lognot (Int64.shift_left (Int64.of_int imm) sh))) ]
  | Movz (_, imm, sh) -> [ b (Int64.shift_left (Int64.of_int imm) sh) ]
  | Msub (sz, x, y, z) ->
      [
        b
          (norm sz
             (Int64.sub (bits env z) (Int64.mul (bits env x) (bits env y))));
      ]
  | Mul (sz, x, y) -> [ b (norm sz (Int64.mul (bits env x) (bits env y))) ]
  | Scvtf (Fsz.D, x) -> [ b (N.s64_to_f64 (bits env x)) ]
  | Scvtf (Fsz.S, x) -> [ b (N.s64_to_f32 (bits env x)) ]
  | Sdiv (sz, x, y) ->
      let n = signed sz (bits env x) and d = signed sz (bits env y) in
      let q =
        if Int64.equal d 0L then 0L
        else if Int64.equal d (-1L) then Int64.neg n (* min / -1 wraps to min *)
        else Int64.div n d
      in
      [ b (norm sz q) ]
  | Shift_imm (o, sz, x, k) -> [ b (shift_imm sz o (bits env x) k) ]
  | St1_lane (fsz, k, x, base) ->
      E.store env (E.ptr env base) ~bytes:(lane_bytes fsz) ~align:1L
        (lanes env x).(k);
      []
  | Str (m, base, k, x) ->
      let size = Msz.bytes m in
      E.store env (addr env base k) ~bytes:size ~align:1L (bits env x);
      []
  | Str_vec (arr, base, k, x) ->
      let size = lane_bytes (Arr.fsz arr) in
      Array.iteri
        (fun j l ->
          E.store env
            (addr env base (Int64.add k (Int64.mul (Int64.of_int j) size)))
            ~bytes:size ~align:1L l)
        (lanes env x);
      []
  | Sub (sz, x, y) -> (
      match get x with
      | Mir_datum.Ptr p -> [ ptr_add env p (Int64.neg (bits env y)) ]
      | _ -> [ b (norm sz (Int64.sub (bits env x) (bits env y))) ])
  | Sub_imm (sz, x, k) -> [ b (norm sz (Int64.sub (bits env x) k)) ]
  | Sxtw x -> [ b (Int64.of_int32 (Int64.to_int32 (bits env x))) ]
  | Trunc (w, x) -> [ b (Int64.logand (bits env x) (Mir_width.mask w)) ]
  | Uxtw x | Wtrunc x -> [ b (Int64.logand (bits env x) 0xFFFF_FFFFL) ]
  | Vfbin (o, arr, x, y) ->
      let fsz = Arr.fsz arr in
      [
        vec
          (Array.map2
             (fun p q -> lres fsz (fop o (lval fsz p) (lval fsz q)))
             (lanes env x) (lanes env y));
      ]
  | Vfmla (arr, acc, x, y) ->
      let fsz = Arr.fsz arr in
      let a = lanes env acc and p = lanes env x and q = lanes env y in
      [
        vec
          (Array.init (Array.length a) (fun j ->
               match fsz with
               | Fsz.S -> N.fma32 p.(j) q.(j) a.(j)
               | Fsz.D ->
                   N.of_f64
                     (Float.fma (N.f64 p.(j)) (N.f64 q.(j)) (N.f64 a.(j)))));
      ]
  | Vfunary (u, arr, x) ->
      [ vec (Array.map (funary (Arr.fsz arr) u) (lanes env x)) ]
  | Vmov (_, x) -> [ vec (lanes env x) ]
  | Vfcmp (c, arr, x, y) ->
      (* a lane all ones where the ordered compare holds, zero where it does
         not or where either operand is a NaN *)
      let fsz = Arr.fsz arr in
      let ones = match fsz with Fsz.D -> -1L | Fsz.S -> 0xFFFF_FFFFL in
      [
        vec
          (Array.map2
             (fun p q ->
               let a = lval fsz p and b = lval fsz q in
               let holds =
                 match c with
                 | Vcmp.Eq -> a = b
                 | Vcmp.Ge -> a >= b
                 | Vcmp.Gt -> a > b
               in
               if holds then ones else 0L)
             (lanes env x) (lanes env y));
      ]
  | Vlogic (o, x, y) ->
      [
        vec
          (Array.map2 (fun p q -> logic o Sz.X p q) (lanes env x) (lanes env y));
      ]
  | Vnot x ->
      let l = lanes env x in
      let w = if Array.length l = 4 then 0xFFFF_FFFFL else -1L in
      [ vec (Array.map (fun p -> Int64.logand (Int64.lognot p) w) l) ]
  | Vbit (_, other, x, m) ->
      let lo = lanes env other and lx = lanes env x and lm = lanes env m in
      [
        vec
          (Array.init (Array.length lm) (fun k ->
               Int64.logor
                 (Int64.logand lm.(k) lx.(k))
                 (Int64.logand (Int64.lognot lm.(k)) lo.(k))));
      ]
  | Vwiden x ->
      let l = lanes env x in
      [ vec [| l.(0); l.(1); 0L; 0L |] ]

let test env = function
  | B_cond (c, f) -> Cond.holds c (E.flags env f ~mask:(Cond.reads c))
  | Cbnz (sz, x) -> not (Int64.equal (norm sz (bits env x)) 0L)
  | Cbz (sz, x) -> Int64.equal (norm sz (bits env x)) 0L
