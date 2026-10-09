(* The model side of the AArch64 per-form conformance: boundary and random
   operands, the interpreter's semantics (or one deliberately wrong entry under
   a mutation), and the prediction of a form's destination, flags and buffer
   from its inputs. Shared by the executables that run the forms on the CPU,
   whichever way the instructions are made. *)

open Machine_ir
open Machine_interp
open Machine_target_aarch64
open A64_op
open A64_forms

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

let canonical = function
  | G Sz.W | F Fsz.S -> fun x -> Int64.logand x 0xFFFF_FFFFL
  | G Sz.X | F Fsz.D -> Fun.id

let boundary = function
  | G _ -> gpr_boundary
  | F Fsz.D -> f64_boundary
  | F Fsz.S -> f32_boundary

let random st = function
  | G _ ->
      if Random.State.bool st then Random.State.int64 st Int64.max_int
      else Int64.neg (Random.State.int64 st Int64.max_int)
  | F Fsz.D ->
      if Random.State.bool st then
        Int64.bits_of_float
          (Float.ldexp
             (Random.State.float st 2. -. 1.)
             (Random.State.int st 200 - 100))
      else Random.State.int64 st Int64.max_int
  | F Fsz.S ->
      Int64.logand
        (if Random.State.bool st then
           Int64.of_int32
             (Int32.bits_of_float
                (Float.ldexp
                   (Random.State.float st 2. -. 1.)
                   (Random.State.int st 60 - 30)))
         else Random.State.int64 st 0x1_0000_0000L)
        0xFFFF_FFFFL

(* ---- the interpreter side ----------------------------------------------------- *)

module Mutation = struct
  type t = Cmp_carry | Fmadd_unfused | Fmax_zero | Nan_flags | W_merge

  let all =
    [
      ("cmp-carry", Cmp_carry);
      ("fmadd-unfused", Fmadd_unfused);
      ("fmax-zero", Fmax_zero);
      ("nan-flags", Nan_flags);
      ("w-merge", W_merge);
    ]
end

exception Model_defect of Mir_observation.Defect.t

(* [A64_sem.exec], or one deliberately wrong entry under a mutation. *)
let semantics mutation env op =
  let real () = A64_sem.exec env op in
  let bits v = Mir_sel_env.bits env v in
  match (mutation, op) with
  | Some Mutation.Cmp_carry, Cmp (sz, a, b) ->
      (* C as a signed comparison *)
      let r = A64_sem.exec env op in
      let lt =
        match sz with
        | Sz.X -> Int64.compare (bits a) (bits b) >= 0
        | Sz.W ->
            Int32.compare (Int64.to_int32 (bits a)) (Int64.to_int32 (bits b))
            >= 0
      in
      List.map
        (function
          | Mir_datum.Flags { bits = f; defined } ->
              Mir_datum.Flags
                {
                  bits =
                    Int64.logor
                      (Int64.logand f (Int64.lognot 2L))
                      (if lt then 2L else 0L);
                  defined;
                }
          | d -> d)
        r
  | Some Mutation.Fmadd_unfused, Fmadd (Fsz.D, a, b, c) ->
      let f v = Int64.float_of_bits (bits v) in
      [
        Mir_datum.Bits
          (Int64.bits_of_float (Sys.opaque_identity (f a *. f b) +. f c));
      ]
  | Some Mutation.Fmax_zero, Fbin (Fop.Max, Fsz.D, a, b) ->
      let x = Int64.float_of_bits (bits a)
      and y = Int64.float_of_bits (bits b) in
      if x = 0. && y = 0. then [ Mir_datum.Bits (Int64.bits_of_float (-0.)) ]
      else real ()
  | Some Mutation.Nan_flags, Fcmp (fsz, a, b) ->
      let f v =
        match fsz with
        | Fsz.D -> Int64.float_of_bits (bits v)
        | Fsz.S -> Int32.float_of_bits (Int64.to_int32 (bits v))
      in
      if Float.is_nan (f a) || Float.is_nan (f b) then
        [ Mir_datum.Flags { bits = 2L; defined = 15L } ]
      else real ()
  | _ -> real ()

let float_bits_nan r x =
  match r with
  | F Fsz.D -> Float.is_nan (Int64.float_of_bits x)
  | F Fsz.S -> Float.is_nan (Int32.float_of_bits (Int64.to_int32 x))
  | G _ -> false

(* What the model predicts: the full destination (low, high), NZCV and the
   buffer, from the inputs, the NZCV and destination seeds and the buffer. *)
let predict mutation (f : Form.t) (ins : int64 list) nz seed (buf : Bytes.t) =
  let memory = Mir_memory.create () in
  let key = Option.get (Mir_memory.alloc memory ~size:32L ~align:16L ()) in
  Mir_memory.write_string memory key ~offset:0L (Bytes.to_string buf);
  let base = Mir_memory.pointer memory key ~lo:0L ~hi:32L in
  let base = Option.get (Mir_memory.offset_by base 8L) in
  let vs =
    List.mapi
      (fun i r -> Mir_value.{ id = Mir_id.Value.of_int i; ty = ty_of r })
      f.Form.inputs
  in
  let flags_v = Mir_value.{ id = Mir_id.Value.of_int 9; ty = Mir_type.Flags } in
  let nzcv = Int64.logand (Int64.shift_right_logical nz 28) 15L in
  let get (v : Mir_value.t) =
    match Mir_id.Value.to_int v.Mir_value.id with
    | 9 -> Mir_datum.Flags { bits = nzcv; defined = 15L }
    | 8 -> Mir_datum.Ptr base
    | i -> Mir_datum.Bits (List.nth ins i)
  in
  let env =
    {
      Mir_sel_env.get;
      memory;
      view = (fun _ -> None);
      defect = (fun d -> raise (Model_defect d));
      call =
        (fun _ _ -> raise (Model_defect Mir_observation.Defect.Invalid_program));
    }
  in
  let mem () =
    Array.init 32 (fun i ->
        Option.value ~default:0
          (Mir_memory.read_bytes memory key ~offset:0L ~n:32).(i))
  in
  match f.Form.make vs flags_v with
  | `Test t -> ((if A64_sem.test env t then 1L else 0L), 0L, nz, mem ())
  | `Op op ->
      let rs = semantics mutation env op in
      let out_flags =
        List.find_map
          (function
            | Mir_datum.Flags { bits; _ } -> Some (Int64.shift_left bits 28)
            | _ -> None)
          rs
      in
      let value =
        List.find_map (function Mir_datum.Bits b -> Some b | _ -> None) rs
      in
      let lo, hi =
        match (f.Form.result, value) with
        | Some (G Sz.W), Some b ->
            (* W writes zero bits 63:32 — or, under the mutation, keep them *)
            if mutation = Some Mutation.W_merge then
              (Int64.logor (Int64.logand seed 0xFFFF_FFFF_0000_0000L) b, 0L)
            else (b, 0L)
        | Some (G Sz.X), Some b -> (b, 0L)
        | Some (F _), Some b -> (b, 0L)
        | _ -> (0L, 0L)
      in
      (lo, hi, Option.value out_flags ~default:nz, mem ())

(* ---- the vectors -------------------------------------------------------------- *)

let seed = 20261007
let random_per_form = 200

(* Every pair of boundary values over a form's first two inputs (a third input
   cycles through its own), then random ones; with the NZCV, destination seed
   and buffer each is run under. *)
let vectors () =
  let st = Random.State.make [| seed |] in
  (* vectors: every pair of boundary values over the first two inputs (a
     third input cycles through its own), then random ones *)
  let vectors =
    List.concat
      (List.mapi
         (fun k (f : Form.t) ->
           let rec product = function
             | [] -> [ [] ]
             | [ r ] -> List.map (fun x -> [ canonical r x ]) (boundary r)
             | r :: q :: rest ->
                 let tail = product (q :: rest) in
                 List.concat_map
                   (fun x -> List.map (fun t -> canonical r x :: t) tail)
                   (boundary r)
           in
           let edges =
             match f.Form.inputs with
             | [ a; b; c ] ->
                 List.mapi
                   (fun i xs ->
                     xs
                     @ [
                         canonical c
                           (List.nth (boundary c)
                              (i mod List.length (boundary c)));
                       ])
                   (product [ a; b ])
             | rs -> product rs
           in
           let randoms =
             List.init random_per_form (fun _ ->
                 List.map (fun r -> canonical r (random st r)) f.Form.inputs)
           in
           List.mapi
             (fun i ins ->
               let ins = ins @ List.init (3 - List.length ins) (fun _ -> 0L) in
               let nz = Int64.shift_left (Int64.of_int (i land 15)) 28 in
               let seed =
                 if i land 1 = 0 then -1L else 0x5A5A_5A5A_A5A5_A5A5L
               in
               let buf =
                 Bytes.init 32 (fun j ->
                     Char.chr (((i * 31) + (j * 17) + k) land 0xFF))
               in
               (k, f, ins, nz, seed, buf))
             (edges @ randoms))
         forms)
  in
  vectors
