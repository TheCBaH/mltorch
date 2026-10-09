(* The model side of the x86-64 per-form conformance: boundary and random
   operands, the interpreter's semantics (or one deliberately wrong entry under
   a mutation), and the prediction of a form's results, flags and buffer from
   its inputs. *)

open Machine_ir
open Machine_interp
open Machine_target_x86_64
open X64_op
open X64_forms

(* ---- operands ---------------------------------------------------------------- *)

let gpr_boundary =
  [
    0L;
    1L;
    -1L;
    2L;
    0x7FFF_FFFFL;
    0x8000_0000L;
    0xFFFF_FFFFL;
    Int64.min_int;
    Int64.max_int;
    0x1234_5678_9ABC_DEF0L;
    -7L;
    3L;
  ]

let f64_boundary =
  List.map Int64.bits_of_float
    [
      0.;
      -0.;
      1.;
      -1.;
      1.5;
      -2.5;
      Float.infinity;
      Float.neg_infinity;
      4.9e-324;
      -2.2250738585072009e-308;
      1.7976931348623157e308;
      0.1;
      9223372036854775808.;
      -9223372036854775808.;
      9007199254740993.;
      1e-310;
      3.4028235677973366e38;
    ]
  @ [ 0x7FF8_0000_0000_0001L; 0xFFF0_0000_0000_0001L ]

let f32_boundary =
  List.map
    (fun x ->
      Int64.logand (Int64.of_int32 (Int32.bits_of_float x)) 0xFFFF_FFFFL)
    [
      0.;
      -0.;
      1.;
      -1.;
      1.5;
      Float.infinity;
      Float.neg_infinity;
      1.4e-45;
      1.1754942e-38;
      3.4028235e38;
      0.1;
      16777216.;
      -2147483648.;
    ]
  @ [ 0x7FC0_0001L; 0xFF80_0001L; 0x0000_0001L ]

let width_mask = function
  | Mir_type.Int w -> Mir_width.mask w
  | Mir_type.Pred -> 1L
  | _ -> -1L

let rand64 st =
  Int64.logxor
    (Int64.shift_left (Int64.of_int (Random.State.bits st)) 34)
    (Int64.logxor
       (Int64.shift_left (Int64.of_int (Random.State.bits st)) 4)
       (Int64.of_int (Random.State.bits st)))

let rand_f64 st =
  if Random.State.bool st then
    Int64.bits_of_float
      (Float.ldexp
         (Random.State.float st 2. -. 1.)
         (Random.State.int st 200 - 100))
  else rand64 st

let rand_f32 st =
  Int64.logand
    (if Random.State.bool st then
       Int64.of_int32
         (Int32.bits_of_float
            (Float.ldexp
               (Random.State.float st 2. -. 1.)
               (Random.State.int st 60 - 30)))
     else rand64 st)
    0xFFFF_FFFFL

let pick l k = List.nth l (k mod List.length l)

(* The lane element a vector type's lanes hold. *)
let lane_ty (t : Mir_type.t) =
  match t with
  | Mir_type.Vec (e, n) -> Some (e, Mir_type.Lanes.to_int n)
  | _ -> None

(* One operand: the datum the model reads and the 128-bit register image the
   CPU starts from (bits the datum does not define are noise, so a form that
   disturbs them is seen). [k] selects a boundary value, or [None] a random
   one. *)
type operand = { datum : Mir_datum.t; lo : int64; hi : int64 }

let operand st (ty : Mir_type.t) k =
  let scalar f32 f64 =
    match k with
    | Some k -> (pick f32 k, pick f64 k)
    | None -> (rand_f32 st, rand_f64 st)
  in
  match ty with
  | Mir_type.Int _ | Mir_type.Pred ->
      let x =
        match k with Some k -> pick gpr_boundary k | None -> rand64 st
      in
      let x = Int64.logand x (width_mask ty) in
      { datum = Mir_datum.Bits x; lo = x; hi = 0L }
  | Mir_type.F64 ->
      let _, x = scalar f32_boundary f64_boundary in
      { datum = Mir_datum.Bits x; lo = x; hi = rand64 st }
  | Mir_type.F32 ->
      let x, _ = scalar f32_boundary f64_boundary in
      {
        datum = Mir_datum.Bits x;
        lo = Int64.logor x (Int64.shift_left (rand64 st) 32);
        hi = rand64 st;
      }
  | Mir_type.Vec (Mir_type.Elem.F64, n) ->
      let n = Mir_type.Lanes.to_int n in
      let lanes =
        Array.init n (fun j ->
            match k with
            | Some k -> pick f64_boundary (k + (j * 5))
            | None -> rand_f64 st)
      in
      { datum = Mir_datum.Lanes lanes; lo = lanes.(0); hi = lanes.(1) }
  | Mir_type.Vec (Mir_type.Elem.F32, n) ->
      let n = Mir_type.Lanes.to_int n in
      let lanes =
        Array.init n (fun j ->
            match k with
            | Some k -> pick f32_boundary (k + (j * 5))
            | None -> rand_f32 st)
      in
      let pack a b = Int64.logor a (Int64.shift_left b 32) in
      {
        datum = Mir_datum.Lanes lanes;
        lo = pack lanes.(0) lanes.(1);
        hi = (if n = 4 then pack lanes.(2) lanes.(3) else rand64 st);
      }
  | _ -> invalid_arg "X64_model.operand"

(* ---- the interpreter side ----------------------------------------------------- *)

module Mutation = struct
  type t = Cmp_carry | Fma_unfused | Max_zero | Ucomi_nan

  let all =
    [
      ("cmp-carry", Cmp_carry);
      ("fma-unfused", Fma_unfused);
      ("max-zero", Max_zero);
      ("ucomi-nan", Ucomi_nan);
    ]
end

exception Model_defect of Mir_observation.Defect.t

let semantics mutation env op =
  let real () = X64_sem.exec env op in
  let bits v = Mir_sel_env.bits env v in
  match (mutation, op) with
  | Some Mutation.Cmp_carry, Cmp (_, _, _) ->
      List.map
        (function
          | Mir_datum.Flags { bits = f; defined } ->
              Mir_datum.Flags { bits = Int64.logxor f Flag.cf; defined }
          | d -> d)
        (real ())
  | Some Mutation.Fma_unfused, Fmadd231 (Fsz.D, a, b, c) ->
      let f v = Int64.float_of_bits (bits v) in
      [
        Mir_datum.Bits
          (Int64.bits_of_float (Sys.opaque_identity (f a *. f b) +. f c));
      ]
  | Some Mutation.Max_zero, Fbin (Fop.Max, Fsz.D, a, b) ->
      let x = Int64.float_of_bits (bits a)
      and y = Int64.float_of_bits (bits b) in
      if x = 0. && y = 0. then [ Mir_datum.Bits (bits a) ] else real ()
  | Some Mutation.Ucomi_nan, Ucomis (fsz, a, b) ->
      let f v =
        match fsz with
        | Fsz.D -> Int64.float_of_bits (bits v)
        | Fsz.S -> Int32.float_of_bits (Int64.to_int32 (bits v))
      in
      if Float.is_nan (f a) || Float.is_nan (f b) then
        [ Mir_datum.Flags { bits = Flag.zf; defined = Flag.all } ]
      else real ()
  | _ -> real ()

(* ---- the prediction ----------------------------------------------------------- *)

let buf_bytes = 32
let buf_base = 8L

(* RFLAGS in the model's compact form: CF 1, PF 2, AF 4, ZF 8, SF 16, OF 32. *)
let compact_flags rflags =
  let b n = Int64.logand (Int64.shift_right_logical rflags n) 1L in
  List.fold_left Int64.logor 0L
    [
      b 0;
      Int64.shift_left (b 2) 1;
      Int64.shift_left (b 4) 2;
      Int64.shift_left (b 6) 3;
      Int64.shift_left (b 7) 4;
      Int64.shift_left (b 11) 5;
    ]

let rflags_image compact =
  let b n = Int64.logand (Int64.shift_right_logical compact n) 1L in
  List.fold_left Int64.logor 2L
    [
      b 0;
      Int64.shift_left (b 1) 2;
      Int64.shift_left (b 2) 4;
      Int64.shift_left (b 3) 6;
      Int64.shift_left (b 4) 7;
      Int64.shift_left (b 5) 11;
    ]

(* What one result is expected to be, under a mask of the bits it defines. *)
type expected =
  | Gpr of int64
  | Ptr_delta of int64  (** the result minus the base it was made from *)
  | Xmm of {
      ty : Mir_type.t;
      lo : int64;
      hi : int64;
      mask_lo : int64;
      mask_hi : int64;
    }
  | Flags_out of { bits : int64; defined : int64 }
  | Flags_kept  (** the flags are as seeded *)
  | Flags_unknown

type prediction = { results : expected list; buffer : int array }

(* One vector: the inputs (one per role, a [Flags] role carrying the compact
   seed), the destination seed and the buffer. *)
type vector = {
  inputs : operand option list;  (** [None]: the flags, base or index role *)
  index : int64;
  flags : int64;  (** compact *)
  seed_lo : int64;
  seed_hi : int64;
  buf : Bytes.t;
}

let roles_values (f : X64_forms.t) =
  List.mapi
    (fun i r ->
      {
        Mir_value.id = Mir_id.Value.of_int i;
        ty =
          (match r with
          | Flags -> Mir_type.Flags
          | Val t -> t
          | Base -> Mir_type.Ptr
          | Index -> Mir_type.i64);
      })
    f.roles

let half_mask_of_ty (t : Mir_type.t) =
  match t with
  | Mir_type.F32 -> (0xFFFF_FFFFL, 0L)
  | Mir_type.F64 -> (-1L, 0L)
  | Mir_type.Vec (Mir_type.Elem.F32, n) when Mir_type.Lanes.to_int n = 2 ->
      (-1L, 0L)
  | Mir_type.Vec _ -> (-1L, -1L)
  | _ -> (-1L, -1L)

let pack_lanes (t : Mir_type.t) (l : int64 array) =
  match t with
  | Mir_type.Vec (Mir_type.Elem.F64, _) ->
      (l.(0), if Array.length l > 1 then l.(1) else 0L)
  | _ ->
      let g j = if j < Array.length l then l.(j) else 0L in
      let pack a b =
        Int64.logor (Int64.logand a 0xFFFF_FFFFL) (Int64.shift_left b 32)
      in
      (pack (g 0) (g 1), pack (g 2) (g 3))

let predict mutation (f : X64_forms.t) (v : vector) : prediction =
  let memory = Mir_memory.create () in
  let key =
    Option.get
      (Mir_memory.alloc memory ~size:(Int64.of_int buf_bytes) ~align:16L ())
  in
  Mir_memory.write_string memory key ~offset:0L (Bytes.to_string v.buf);
  let base =
    Option.get
      (Mir_memory.offset_by
         (Mir_memory.pointer memory key ~lo:0L ~hi:(Int64.of_int buf_bytes))
         buf_base)
  in
  let values = roles_values f in
  let datum_of i =
    match List.nth f.roles i with
    | Flags -> Mir_datum.Flags { bits = v.flags; defined = Flag.all }
    | Base -> Mir_datum.Ptr base
    | Index -> Mir_datum.Bits v.index
    | Val _ -> (Option.get (List.nth v.inputs i)).datum
  in
  let env =
    {
      Mir_sel_env.get =
        (fun (x : Mir_value.t) -> datum_of (Mir_id.Value.to_int x.Mir_value.id));
      memory;
      view = (fun _ -> None);
      defect = (fun d -> raise (Model_defect d));
      call =
        (fun _ _ -> raise (Model_defect Mir_observation.Defect.Invalid_program));
    }
  in
  let op = f.make values in
  let rs = semantics mutation env op in
  let tys =
    match X64_op.typing op with
    | Ok t -> t
    | Error e -> invalid_arg ("X64_model.predict: " ^ e)
  in
  let tied_use =
    List.find_map
      (function
        | Mir_target.Constraint.Tied { result = 0; use } -> Some use | _ -> None)
      (X64_op.constraints op)
  in
  let old_of_dest ty =
    match tied_use with
    | Some u -> (
        match X64_op.uses op with
        | _ -> (
            let id =
              Mir_id.Value.to_int (List.nth (X64_op.uses op) u).Mir_value.id
            in
            match List.nth v.inputs id with
            | Some o -> (o.lo, o.hi)
            | None -> (v.seed_lo, v.seed_hi)))
    | None ->
        ignore ty;
        (v.seed_lo, v.seed_hi)
  in
  let write = X64_op.result_write op in
  let base_off = base.Mir_memory.Pointer.offset in
  let results =
    List.map2
      (fun (ty : Mir_type.t) (d : Mir_datum.t) ->
        match (ty, d) with
        | Mir_type.Flags, Mir_datum.Flags { bits; defined } ->
            Flags_out { bits; defined }
        | (Mir_type.Int _ | Mir_type.Pred), Mir_datum.Bits b -> Gpr b
        | Mir_type.Ptr, Mir_datum.Ptr p ->
            Ptr_delta (Int64.sub p.Mir_memory.Pointer.offset base_off)
        | (Mir_type.F32 | Mir_type.F64), Mir_datum.Bits b -> (
            let olo, ohi = old_of_dest ty in
            let keep =
              match ty with Mir_type.F32 -> 0xFFFF_FFFFL | _ -> -1L
            in
            let lo_keep =
              match ty with
              | Mir_type.F32 -> Int64.logand olo (Int64.lognot keep)
              | _ -> 0L
            in
            match write with
            | Mir_target.Write.Merge ->
                Xmm
                  {
                    ty;
                    lo = Int64.logor (Int64.logand b keep) lo_keep;
                    hi = ohi;
                    mask_lo = -1L;
                    mask_hi = -1L;
                  }
            | Mir_target.Write.Zero_upper ->
                Xmm
                  {
                    ty;
                    lo = Int64.logand b keep;
                    hi = 0L;
                    mask_lo = -1L;
                    mask_hi = -1L;
                  }
            | Mir_target.Write.Undefined_upper ->
                Xmm
                  {
                    ty;
                    lo = Int64.logand b keep;
                    hi = 0L;
                    mask_lo = keep;
                    mask_hi = 0L;
                  })
        | Mir_type.Vec _, Mir_datum.Lanes l ->
            let lo, hi = pack_lanes ty l in
            let mlo, mhi = half_mask_of_ty ty in
            let mlo, mhi =
              match write with
              | Mir_target.Write.Undefined_upper -> (mlo, mhi)
              | _ -> (-1L, -1L)
            in
            Xmm { ty; lo; hi; mask_lo = mlo; mask_hi = mhi }
        | _ -> invalid_arg "X64_model.predict: result shape")
      tys rs
  in
  let results =
    if
      List.exists (function Flags_out _ -> true | _ -> false) results
      || X64_op.writes_flags op
    then
      if List.exists (function Flags_out _ -> true | _ -> false) results then
        results
      else results @ [ Flags_unknown ]
    else results @ [ Flags_kept ]
  in
  let buffer =
    Array.init buf_bytes (fun i ->
        Option.value ~default:0
          (Mir_memory.read_bytes memory key ~offset:0L ~n:buf_bytes).(i))
  in
  { results; buffer }

(* ---- the vectors -------------------------------------------------------------- *)

let seed = 20261009
let random_per_form = 150

let make_vector st (f : X64_forms.t) ~k =
  let inputs =
    List.map
      (function
        | Val ty -> Some (operand st ty k) | Flags | Base | Index -> None)
      f.roles
  in
  let n = match k with Some k -> k | None -> Random.State.bits st in
  {
    inputs;
    index = pick [ 0L; 1L; 2L; -1L ] (n / 3);
    flags = Int64.of_int (Random.State.int st 64);
    seed_lo = rand64 st;
    seed_hi = rand64 st;
    buf =
      Bytes.init buf_bytes (fun j -> Char.chr (((n * 31) + (j * 17)) land 0xFF));
  }

(* A form's boundary vectors, then random ones. Boundary values rotate
   per operand at coprime strides so pairs of them meet. *)
let vectors (f : X64_forms.t) =
  let st = Random.State.make [| seed; Hashtbl.hash f.name |] in
  let edge_count = 17 * 17 in
  let edges =
    List.init edge_count (fun k ->
        let v = make_vector st f ~k:(Some 0) in
        let inputs =
          List.mapi
            (fun i -> function
              | None -> None
              | Some _ -> (
                  match List.nth f.roles i with
                  | Val ty ->
                      let stride = if i = 0 then 1 else 17 in
                      Some (operand st ty (Some ((k / stride) + i)))
                  | _ -> None))
            v.inputs
        in
        { v with inputs; flags = Int64.of_int (k land 63) })
  in
  let randoms = List.init random_per_form (fun _ -> make_vector st f ~k:None) in
  edges @ randoms

(* ---- comparison classes ------------------------------------------------------- *)

(* Every NaN is one class: payload and sign are not modelled, and the CPU's
   default NaN differs from the host interpreter's by architecture. *)
let canon32 x =
  let e = Int64.logand (Int64.shift_right_logical x 23) 0xFFL
  and m = Int64.logand x 0x7F_FFFFL in
  if Int64.equal e 0xFFL && not (Int64.equal m 0L) then 0x7FC0_0000L
  else Int64.logand x 0xFFFF_FFFFL

let canon64 x =
  let e = Int64.logand (Int64.shift_right_logical x 52) 0x7FFL
  and m = Int64.logand x 0xF_FFFF_FFFF_FFFFL in
  if Int64.equal e 0x7FFL && not (Int64.equal m 0L) then 0x7FF8_0000_0000_0000L
  else x

let pair32 w =
  Int64.logor (canon32 w)
    (Int64.shift_left (canon32 (Int64.shift_right_logical w 32)) 32)

(* A register image with the float lanes of a result of type [ty] canonical. *)
let canonical_xmm (ty : Mir_type.t) (lo, hi) =
  match ty with
  | Mir_type.F32 ->
      ( Int64.logor (canon32 lo) (Int64.logand lo (Int64.shift_left (-1L) 32)),
        hi )
  | Mir_type.F64 -> (canon64 lo, hi)
  | Mir_type.Vec (Mir_type.Elem.F64, _) -> (canon64 lo, canon64 hi)
  | Mir_type.Vec (Mir_type.Elem.F32, n) when Mir_type.Lanes.to_int n = 2 ->
      (pair32 lo, hi)
  | Mir_type.Vec (Mir_type.Elem.F32, _) -> (pair32 lo, pair32 hi)
  | _ -> (lo, hi)
