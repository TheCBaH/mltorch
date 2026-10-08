(* The semantics of each admitted form, from the Intel SDM's operation
   descriptions: 32-bit GPR writes zero-extend; CMP sets CF on unsigned borrow,
   ZF, SF from the result's top bit, OF on signed overflow, AF on a borrow out
   of bit 3, PF on an even number of ones in the low byte; TEST clears CF and
   OF and leaves AF undefined; UCOMISD/SS set ZF, PF and CF to 111 unordered,
   100 equal, 001 less, 000 greater and clear OF, SF and AF; BT sets CF only;
   MAXSD/SS return the source operand when either operand is a NaN or both are
   zero; CVTTSD2SI returns the integer indefinite 0x8000000000000000 when out of
   range; IDIV faults (#DE) on a zero divisor or an unrepresentable quotient,
   which the interpreter reports as a domain defect. MXCSR is fixed to round to
   nearest even with no flush-to-zero and all exceptions masked. *)

open Machine_ir
open Machine_interp
open X64_op
module E = Mir_sel_env
module N = Mir_numeric

let mask = function Sz.L -> 0xFFFF_FFFFL | Sz.Q -> -1L
let norm sz x = Int64.logand x (mask sz)

let signed sz x =
  match sz with Sz.L -> Int64.of_int32 (Int64.to_int32 x) | Sz.Q -> x

let bits env v = E.bits env v
let b x = Mir_datum.Bits x
let bit c f = if c then f else 0L

let parity r =
  let x = ref (Int64.to_int (Int64.logand r 0xFFL)) and n = ref 0 in
  while !x <> 0 do
    n := !n + (!x land 1);
    x := !x lsr 1
  done;
  !n land 1 = 0

let top sz r =
  not
    (Int64.equal
       (Int64.logand (Int64.shift_right_logical r (Sz.bits sz - 1)) 1L)
       0L)

(* RFLAGS of [a - b] at [sz]. *)
let sub_flags sz a b =
  let r = norm sz (Int64.sub a b) in
  let sa = signed sz a and sb = signed sz b in
  let ovf =
    top Sz.Q (Int64.logand (Int64.logxor sa sb) (Int64.logxor sa (signed sz r)))
  in
  List.fold_left Int64.logor 0L
    [
      bit (Int64.unsigned_compare (norm sz a) (norm sz b) < 0) Flag.cf;
      bit (parity r) Flag.pf;
      bit
        (not
           (Int64.equal
              (Int64.logand (Int64.logxor (Int64.logxor a b) r) 0x10L)
              0L))
        Flag.af;
      bit (Int64.equal r 0L) Flag.zf;
      bit (top sz r) Flag.sf;
      bit ovf Flag.of_;
    ]

let flags v defined = Mir_datum.Flags { bits = Int64.logand v defined; defined }
let fval fsz x = match fsz with Fsz.D -> N.f64 x | Fsz.S -> N.f32 x
let lres fsz x = match fsz with Fsz.D -> N.of_f64 x | Fsz.S -> N.round32 x
let fres fsz x = b (lres fsz x)
let ones = function Fsz.D -> -1L | Fsz.S -> 0xFFFF_FFFFL
let fmask = function Fsz.D -> -1L | Fsz.S -> 0xFFFF_FFFFL

let ptr_add env (p : Mir_memory.Pointer.t) delta =
  match Mir_memory.offset_by p delta with
  | Some q -> q
  | None -> env.E.defect Mir_observation.Defect.Bad_access

let address env (a : Addr.t) =
  let base = E.ptr env a.Addr.base in
  let delta =
    match a.Addr.index with
    | Some (i, s) -> Int64.add (Int64.mul (bits env i) s) a.Addr.disp
    | None -> a.Addr.disp
  in
  ptr_add env base delta

let maxs x y =
  if Float.is_nan x || Float.is_nan y then y
  else if x = 0. && y = 0. then y
  else if x > y then x
  else y

let cvtts2si x =
  if Float.is_nan x || x >= 9223372036854775808. || x < -9223372036854775808.
  then Int64.min_int
  else Int64.of_float x

(* A vector operand's lanes. *)
let lanes env v =
  match env.E.get v with
  | Mir_datum.Lanes l -> l
  | Mir_datum.Bits _ | Mir_datum.Flags _ | Mir_datum.Order | Mir_datum.Ptr _ ->
      env.E.defect Mir_observation.Defect.Invalid_program

let vec l = Mir_datum.Lanes l
let lane_bytes = function Fsz.D -> 8L | Fsz.S -> 4L

(* Lane [j]'s address in a packed access at [a]. *)
let lane_at env (a : Addr.t) pk j =
  match
    Mir_memory.offset_by (address env a)
      (Int64.mul (Int64.of_int j) (lane_bytes (Pk.fsz pk)))
  with
  | Some p -> p
  | None -> env.E.defect Mir_observation.Defect.Bad_access

let exec env op =
  let get = env.E.get in
  match op with
  | Alu (o, sz, x, y) -> (
      match (get x, o) with
      | Mir_datum.Ptr p, Alu.Add ->
          [ Mir_datum.Ptr (ptr_add env p (bits env y)) ]
      | Mir_datum.Ptr p, Alu.Sub ->
          [ Mir_datum.Ptr (ptr_add env p (Int64.neg (bits env y))) ]
      | _ ->
          let a = bits env x and c = bits env y in
          [
            b
              (norm sz
                 (match o with
                 | Alu.Add -> Int64.add a c
                 | Alu.And -> Int64.logand a c
                 | Alu.Or -> Int64.logor a c
                 | Alu.Sub -> Int64.sub a c
                 | Alu.Xor -> Int64.logxor a c));
          ])
  | Alu_imm (o, sz, x, k) ->
      let a = bits env x in
      [
        b
          (norm sz
             (match o with
             | Alu.Add -> Int64.add a k
             | Alu.And -> Int64.logand a k
             | Alu.Or -> Int64.logor a k
             | Alu.Sub -> Int64.sub a k
             | Alu.Xor -> Int64.logxor a k));
      ]
  | Bt (_, x, k) ->
      [
        flags
          (bit
             (Int64.equal
                (Int64.logand (Int64.shift_right_logical (bits env x) k) 1L)
                1L)
             Flag.cf)
          Flag.cf;
      ]
  | Call { callee; args; results } ->
      let rs = env.E.call callee (List.map get args) in
      if List.length rs <> List.length results + 1 then
        env.E.defect Mir_observation.Defect.Invalid_program;
      rs
  | Cmov (_, c, f, x, y) ->
      [
        (if Cond.holds c (E.flags env f ~mask:(Cond.reads c)) then get x
         else get y);
      ]
  | Cmp (sz, x, y) ->
      let n v =
        match get v with
        | Mir_datum.Ptr p -> Mir_memory.address env.E.memory p
        | _ -> bits env v
      in
      [ flags (sub_flags sz (n x) (n y)) Flag.all ]
  | Cmp_imm (sz, x, k) -> [ flags (sub_flags sz (bits env x) k) Flag.all ]
  | Cmps (p, fsz, x, y) ->
      let a = fval fsz (bits env x) and c = fval fsz (bits env y) in
      let t =
        match p with
        | Cmp_pred.Eq -> a = c
        | Cmp_pred.Unord -> Float.is_nan a || Float.is_nan c
      in
      [ b (if t then ones fsz else 0L) ]
  | Cqo_idiv (x, y) ->
      let n = bits env x and d = bits env y in
      if Int64.equal d 0L || (Int64.equal n Int64.min_int && Int64.equal d (-1L))
      then env.E.defect Mir_observation.Defect.Domain
      else [ b (Int64.div n d); b (Int64.rem n d) ]
  | Cvt (Fsz.D, x) -> [ b (N.of_f64 (N.f32 (bits env x))) ]
  | Cvt (Fsz.S, x) -> [ b (N.round32 (N.f64 (bits env x))) ]
  | Cvtsi2s (Fsz.D, x) -> [ b (N.s64_to_f64 (bits env x)) ]
  | Cvtsi2s (Fsz.S, x) -> [ b (N.s64_to_f32 (bits env x)) ]
  | Cvtts2si (fsz, x) -> [ b (cvtts2si (fval fsz (bits env x))) ]
  | Ext { signed; from; src } ->
      let k = Mir_width.bits from in
      let x = Int64.logand (bits env src) (Mir_width.mask from) in
      let x =
        if signed then Int64.shift_right (Int64.shift_left x (64 - k)) (64 - k)
        else x
      in
      [ b (norm Sz.L x) ]
  | Fbin (o, fsz, x, y) ->
      let a = fval fsz (bits env x) and c = fval fsz (bits env y) in
      [
        fres fsz
          (match o with
          | Fop.Add -> a +. c
          | Fop.Div -> a /. c
          | Fop.Max -> maxs a c
          | Fop.Mul -> a *. c
          | Fop.Sub -> a -. c);
      ]
  | Cvtpd2ps x ->
      [ vec (Array.map (fun l -> N.round32 (N.f64 l)) (lanes env x)) ]
  | Cvtps2pd x ->
      [ vec (Array.map (fun l -> N.of_f64 (N.f32 l)) (lanes env x)) ]
  | Flogic (o, fsz, x, y) ->
      let a = bits env x and c = bits env y in
      [
        b
          (Int64.logand (fmask fsz)
             (match o with
             | Flogic.And -> Int64.logand a c
             | Flogic.Andn -> Int64.logand (Int64.lognot a) c
             | Flogic.Or -> Int64.logor a c
             | Flogic.Xor -> Int64.logxor a c));
      ]
  | Fmadd231 (Fsz.D, x, y, z) ->
      [
        b
          (N.of_f64
             (Float.fma
                (N.f64 (bits env x))
                (N.f64 (bits env y))
                (N.f64 (bits env z))));
      ]
  | Fmadd231 (Fsz.S, x, y, z) ->
      [ b (N.fma32 (bits env x) (bits env y) (bits env z)) ]
  | Imul (sz, x, y) -> [ b (norm sz (Int64.mul (bits env x) (bits env y))) ]
  | Imul_imm (sz, x, k) -> [ b (norm sz (Int64.mul (bits env x) k)) ]
  | Lea (x, k) -> [ Mir_datum.Ptr (ptr_add env (E.ptr env x) k) ]
  | Lea_view view -> (
      match env.E.view view with
      | Some p -> [ Mir_datum.Ptr p ]
      | None -> env.E.defect Mir_observation.Defect.Invalid_program)
  | Load (m, a) ->
      [ b (E.load env (address env a) ~bytes:(Msz.bytes m) ~align:1L) ]
  | Movlhps (x, y) ->
      let l = Array.copy (lanes env x) and h = lanes env y in
      l.(2) <- h.(0);
      l.(3) <- h.(1);
      [ vec l ]
  | Movq_low x -> [ vec (Array.sub (lanes env x) 0 2) ]
  | Movq_widen x ->
      let l = lanes env x in
      [ vec [| l.(0); l.(1); 0L; 0L |] ]
  | Movup_load (pk, a) ->
      [
        vec
          (Array.init (Pk.lanes pk) (fun j ->
               E.load env (lane_at env a pk j)
                 ~bytes:(lane_bytes (Pk.fsz pk))
                 ~align:1L));
      ]
  | Movup_store (pk, a, x) ->
      Array.iteri
        (fun j l ->
          E.store env (lane_at env a pk j)
            ~bytes:(lane_bytes (Pk.fsz pk))
            ~align:1L l)
        (lanes env x);
      []
  | Mov (_, x) | Movap x -> [ get x ]
  | Mov_imm (_, k) -> [ b k ]
  | Movq_from_gpr (_, x) | Movq_to_gpr (_, x) -> [ b (bits env x) ]
  | Movsxd x -> [ b (Int64.of_int32 (Int64.to_int32 (bits env x))) ]
  | Movzx32 x | Trunc32 x -> [ b (Int64.logand (bits env x) 0xFFFF_FFFFL) ]
  | Trunc_zx (w, x) -> [ b (Int64.logand (bits env x) (Mir_width.mask w)) ]
  | Neg (sz, x) -> [ b (norm sz (Int64.neg (bits env x))) ]
  | Pbin (o, pk, x, y) ->
      let fsz = Pk.fsz pk in
      [
        vec
          (Array.map2
             (fun p q ->
               let p = fval fsz p and q = fval fsz q in
               lres fsz
                 (match o with
                 | Fop.Add -> p +. q
                 | Fop.Div -> p /. q
                 | Fop.Max -> maxs p q
                 | Fop.Mul -> p *. q
                 | Fop.Sub -> p -. q))
             (lanes env x) (lanes env y));
      ]
  | Pfmadd231 (pk, x, y, z) ->
      let p = lanes env x and q = lanes env y and c = lanes env z in
      [
        vec
          (Array.init (Array.length c) (fun j ->
               match pk with
               | Pk.Ps -> N.fma32 p.(j) q.(j) c.(j)
               | Pk.Pd ->
                   N.of_f64
                     (Float.fma (N.f64 p.(j)) (N.f64 q.(j)) (N.f64 c.(j)))));
      ]
  | Plogic (o, pk, x, y) ->
      let m = fmask (Pk.fsz pk) in
      [
        vec
          (Array.map2
             (fun a c ->
               Int64.logand m
                 (match o with
                 | Flogic.And -> Int64.logand a c
                 | Flogic.Andn -> Int64.logand (Int64.lognot a) c
                 | Flogic.Or -> Int64.logor a c
                 | Flogic.Xor -> Int64.logxor a c))
             (lanes env x) (lanes env y));
      ]
  | Pshufd_half x -> [ vec (Array.sub (lanes env x) 2 2) ]
  | Pshufd_lane (_, k, x) -> [ b (lanes env x).(k) ]
  | Pshufd_splat (pk, x) -> [ vec (Array.make (Pk.lanes pk) (bits env x)) ]
  | Psqrt (pk, x) ->
      let fsz = Pk.fsz pk in
      [
        vec
          (Array.map
             (fun l -> lres fsz (Float.sqrt (fval fsz l)))
             (lanes env x));
      ]
  | Round_trunc (fsz, x) -> [ fres fsz (Float.trunc (fval fsz (bits env x))) ]
  | Setcc_zx (c, f) ->
      [
        b (if Cond.holds c (E.flags env f ~mask:(Cond.reads c)) then 1L else 0L);
      ]
  | Shift_imm (o, sz, x, k) ->
      let a = bits env x in
      [
        b
          (match o with
          | Shift.Shl -> norm sz (Int64.shift_left a k)
          | Shift.Shr -> Int64.shift_right_logical (norm sz a) k
          | Shift.Sar -> norm sz (Int64.shift_right (signed sz a) k));
      ]
  | Sqrt (fsz, x) -> [ fres fsz (Float.sqrt (fval fsz (bits env x))) ]
  | Store (m, a, x) ->
      E.store env (address env a) ~bytes:(Msz.bytes m) ~align:1L (bits env x);
      []
  | Test (sz, x, y) ->
      let r = norm sz (Int64.logand (bits env x) (bits env y)) in
      [
        flags
          (List.fold_left Int64.logor 0L
             [
               bit (parity r) Flag.pf;
               bit (Int64.equal r 0L) Flag.zf;
               bit (top sz r) Flag.sf;
             ])
          (Int64.logand Flag.all (Int64.lognot Flag.af));
      ]
  | Ucomis (fsz, x, y) ->
      let a = fval fsz (bits env x) and c = fval fsz (bits env y) in
      let f =
        if Float.is_nan a || Float.is_nan c then
          Int64.logor Flag.zf (Int64.logor Flag.pf Flag.cf)
        else if a = c then Flag.zf
        else if a < c then Flag.cf
        else 0L
      in
      [ flags f Flag.all ]

let test env = function
  | Jcc (c, f) -> Cond.holds c (E.flags env f ~mask:(Cond.reads c))
