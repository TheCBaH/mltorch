(* Native per-form conformance for AArch64 (C3n). Every admitted form
   instance runs on this CPU inside a generated inline-assembly harness: the
   destination is seeded with a pattern so the write mask is visible, NZCV is
   seeded before and read right after the instruction, memory forms use a
   canaried buffer, and FPCR is fixed to round-to-nearest-even, no flush to
   zero, no default NaN and no traps for the run, then restored. Each result is
   compared with [A64_sem.exec] under the form's masks: exact integer bits,
   exact zeroed upper bits, exact NZCV, and, for a float result, any NaN for a
   NaN (the model does not claim payloads).

   [--mutate NAME] replaces one semantic entry with a deliberately wrong one;
   the run then succeeds only if the comparison catches it. A host that is not
   AArch64, or lacks FP/ASIMD, is reported as unavailable (exit 2), never as a
   pass. *)

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

(* ---- driving ----------------------------------------------------------------- *)

let run_cmd cmd = Sys.command cmd

let () =
  let mutation =
    match Array.to_list Sys.argv with
    | [ _; "--mutate"; m ] -> (
        match List.assoc_opt m Mutation.all with
        | Some m -> Some m
        | None ->
            prerr_endline ("unknown mutation " ^ m);
            exit 64)
    | [ _ ] -> None
    | _ ->
        prerr_endline "usage: a64_conformance [--mutate NAME]";
        exit 64
  in
  let arch =
    String.trim (In_channel.input_all (Unix.open_process_in "uname -m"))
  in
  if arch <> "aarch64" then (
    Printf.printf "unavailable: host %s is not aarch64\n" arch;
    exit 2);
  let dir =
    let f = Filename.temp_file "a64_conformance" "" in
    Sys.remove f;
    Sys.mkdir f 0o700;
    f
  in
  let c = Filename.concat dir "harness.c"
  and exe = Filename.concat dir "harness" in
  Out_channel.with_open_text c (fun oc -> output_string oc (A64_c.c_program ()));
  if
    run_cmd
      (Printf.sprintf "gcc -O1 -o %s %s" (Filename.quote exe) (Filename.quote c))
    <> 0
  then (
    prerr_endline "the harness does not compile";
    exit 1);
  let seed = 20261007 in
  let st = Random.State.make [| seed |] in
  let random_per_form = 200 in
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
  let input = Filename.concat dir "in.txt"
  and output = Filename.concat dir "out.txt" in
  Out_channel.with_open_text input (fun oc ->
      List.iter
        (fun (k, _, ins, nz, seed, buf) ->
          Printf.fprintf oc "%d %Lx %Lx %Lx %Lx %Lx %Lx %Lx %Lx %Lx\n" k
            (List.nth ins 0) (List.nth ins 1) (List.nth ins 2) nz seed
            (Bytes.get_int64_le buf 0) (Bytes.get_int64_le buf 8)
            (Bytes.get_int64_le buf 16)
            (Bytes.get_int64_le buf 24))
        vectors);
  let status =
    run_cmd
      (Printf.sprintf "%s < %s > %s" (Filename.quote exe) (Filename.quote input)
         (Filename.quote output))
  in
  if status = 2 then (
    print_endline "unavailable: the host lacks FP/ASIMD";
    exit 2);
  if status <> 0 then (
    prerr_endline "the harness failed";
    exit 1);
  let lines =
    In_channel.with_open_text output (fun ic ->
        String.split_on_char '\n' (In_channel.input_all ic)
        |> List.filter (( <> ) ""))
  in
  let fpcr, results = match lines with f :: r -> (f, r) | [] -> ("?", []) in
  let addresses, results =
    List.partition
      (fun l -> String.length l > 5 && String.sub l 0 5 = "addr ")
      results
  in
  (* the model: a region at a page-aligned base, the view at the target's
     offset; ADRP gives the region offset with its low twelve bits cleared and
     ADD :lo12: adds them back *)
  let address_failures =
    List.filter_map
      (fun l ->
        match String.split_on_char ' ' l with
        | [ _; k; target; page; full ] ->
            let h x = Int64.of_string ("0x" ^ x) in
            let target = h target and page = h page and full = h full in
            let base = Int64.logand target (Int64.lognot 0xFFFL) in
            let offset = Int64.sub target base in
            let model_page =
              Int64.add base (Int64.logand offset (Int64.lognot 0xFFFL))
            in
            let model_full =
              Int64.add model_page (Int64.logand offset 0xFFFL)
            in
            if Int64.equal page model_page && Int64.equal full model_full then
              None
            else
              Some
                (Printf.sprintf
                   "adrp/add_lo12 +%s: native %Lx %Lx, model %Lx %Lx" k page
                   full model_page model_full)
        | _ -> Some ("unreadable " ^ l))
      addresses
  in
  let gcc =
    String.trim
      (In_channel.input_all (Unix.open_process_in "gcc --version | head -1"))
  in
  Printf.printf
    "aarch64 per-form conformance: %d forms, %d vectors (all boundary pairs, \
     %d random each), seed %d\n"
    (List.length forms) (List.length vectors) random_per_form seed;
  Printf.printf "host: %s; %s; %s\n" arch gcc fpcr;
  let failures = Hashtbl.create 16 in
  List.iter2
    (fun (_, (f : Form.t), ins, nz, seed, buf) line ->
      match
        String.split_on_char ' ' line
        |> List.map (fun s -> Int64.of_string ("0x" ^ s))
      with
      | [ lo; hi; flags; m0; m1; m2; m3 ] -> (
          let native_mem = Bytes.create 32 in
          List.iteri
            (fun i w -> Bytes.set_int64_le native_mem (8 * i) w)
            [ m0; m1; m2; m3 ];
          match predict mutation f ins nz seed buf with
          | exception Model_defect _ ->
              Hashtbl.replace failures f.Form.name
                (Printf.sprintf "model defect at inputs %s"
                   (String.concat "," (List.map (Printf.sprintf "%Lx") ins)))
          | plo, phi, pflags, pmem ->
              let result_ok =
                match f.Form.result with
                | Some (F _ as r)
                  when float_bits_nan r lo && float_bits_nan r plo ->
                    Int64.equal hi phi
                | _ -> Int64.equal lo plo && Int64.equal hi phi
              in
              let mem_ok =
                Array.for_all Fun.id
                  (Array.init 32 (fun i ->
                       Char.code (Bytes.get native_mem i) = pmem.(i)))
              in
              let flags_ok =
                Int64.equal
                  (Int64.logand flags 0xF000_0000L)
                  (Int64.logand pflags 0xF000_0000L)
              in
              if
                (not (result_ok && mem_ok && flags_ok))
                && not (Hashtbl.mem failures f.Form.name)
              then
                Hashtbl.replace failures f.Form.name
                  (Printf.sprintf
                     "inputs %s nzcv %Lx seed %Lx: native %Lx:%Lx flags %Lx, \
                      model %Lx:%Lx flags %Lx%s"
                     (String.concat "," (List.map (Printf.sprintf "%Lx") ins))
                     (Int64.shift_right_logical nz 28)
                     seed hi lo
                     (Int64.shift_right_logical flags 28)
                     phi plo
                     (Int64.shift_right_logical pflags 28)
                     (if mem_ok then "" else " (memory differs)")))
      | _ -> Hashtbl.replace failures f.Form.name "unreadable harness output")
    vectors results;
  let failed = Hashtbl.length failures in
  List.iter
    (fun (f : Form.t) ->
      match Hashtbl.find_opt failures f.Form.name with
      | Some why -> Printf.printf "FAIL %s: %s\n" f.Form.name why
      | None -> ())
    forms;
  List.iter (Printf.printf "FAIL %s\n") address_failures;
  Printf.printf
    "%d of %d forms agree; address forms (adrp, add :lo12:) at %d targets: %s\n"
    (List.length forms - failed)
    (List.length forms) (List.length addresses)
    (if address_failures = [] && addresses <> [] then "agree" else "DISAGREE");
  let failed =
    failed + List.length address_failures + if addresses = [] then 1 else 0
  in
  match mutation with
  | None -> exit (if failed = 0 then 0 else 1)
  | Some _ ->
      (* a mutation must be caught *)
      if failed > 0 then (
        print_endline "mutation detected";
        exit 0)
      else (
        print_endline "MUTATION SURVIVED";
        exit 1)
