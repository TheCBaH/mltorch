(* The Advanced SIMD forms the selector emits, as instances of one record: the
   kinds of their inputs and result, the selected form over virtual values, and
   which use (if any) the result is tied to. Run on the CPU through typed Rivet
   instructions against [A64_sem], over lane patterns drawn from the same
   boundary values as the scalar forms. *)

open Machine_ir
open Machine_target_aarch64
open A64_op

type kind =
  | V of Arr.t  (** a vector register of this arrangement *)
  | M of Arr.t  (** the mask a compare of this arrangement makes *)
  | Sc of Fsz.t  (** a scalar FP register *)
  | Gp of Sz.t  (** a general register *)

type t = {
  name : string;
  inputs : kind list;
  result : kind option;
  mem : bool;  (** the form takes a base pointer, to a canaried buffer *)
  tied : int option;  (** the use the result shares a register with *)
  make : Mir_value.t list -> Mir_value.t -> A64_op.t;
      (** over the inputs, then the base pointer *)
}

let ty_of = function
  | V a -> Arr.ty a
  | M a -> Option.get (Arr.mask_ty a)
  | Sc Fsz.D -> Mir_type.F64
  | Sc Fsz.S -> Mir_type.F32
  | Gp Sz.X -> Mir_type.i64
  | Gp Sz.W -> Mir_type.i32

let form ?(mem = false) ?tied name inputs result make =
  { name; inputs; result; mem; tied; make }

let arrs = Arr.[ S2; S4; D2 ]
let fszs = Fsz.[ S; D ]
let lanes_of_fsz = function Fsz.D -> 2 | Fsz.S -> 4

let forms =
  let u i vs = List.nth vs i in
  List.concat
    [
      List.concat_map
        (fun (o, on) ->
          List.map
            (fun arr ->
              form
                (Printf.sprintf "v%s.%s" on (Arr.name arr))
                [ V arr; V arr ] (Some (V arr))
                (fun vs _ -> Vfbin (o, arr, u 0 vs, u 1 vs)))
            arrs)
        Fop.
          [
            (Add, "fadd");
            (Sub, "fsub");
            (Mul, "fmul");
            (Div, "fdiv");
            (Max, "fmax");
          ];
      List.map
        (fun arr ->
          form ~tied:0
            (Printf.sprintf "vfmla.%s" (Arr.name arr))
            [ V arr; V arr; V arr ] (Some (V arr))
            (fun vs _ -> Vfmla (arr, u 0 vs, u 1 vs, u 2 vs)))
        arrs;
      List.concat_map
        (fun (o, on) ->
          List.map
            (fun arr ->
              form
                (Printf.sprintf "v%s.%s" on (Arr.name arr))
                [ V arr ] (Some (V arr))
                (fun vs _ -> Vfunary (o, arr, u 0 vs)))
            arrs)
        Funary.[ (Fneg, "fneg"); (Fsqrt, "fsqrt"); (Frintz, "frintz") ];
      List.map
        (fun arr ->
          form
            (Printf.sprintf "vmov.%s" (Arr.name arr))
            [ V arr ] (Some (V arr))
            (fun vs _ -> Vmov (arr, u 0 vs)))
        arrs;
      List.map
        (fun arr ->
          form
            (Printf.sprintf "dup.%s" (Arr.name arr))
            [ Sc (Arr.fsz arr) ]
            (Some (V arr))
            (fun vs _ -> Dup_elem (arr, u 0 vs)))
        arrs;
      List.map
        (fun k ->
          form (Printf.sprintf "dup.half[%d]" k) [ V Arr.S4 ] (Some (V Arr.S2))
            (fun vs _ -> Dup_half (k, u 0 vs)))
        [ 0; 1 ];
      List.concat_map
        (fun fsz ->
          List.init (lanes_of_fsz fsz) (fun k ->
              form
                (Printf.sprintf "dup.lane.%s[%d]" (Fsz.name fsz) k)
                [ V (if fsz = Fsz.D then Arr.D2 else Arr.S4) ]
                (Some (Sc fsz))
                (fun vs _ -> Dup_lane (fsz, k, u 0 vs))))
        fszs;
      [
        form "fcvtl" [ V Arr.S2 ] (Some (V Arr.D2)) (fun vs _ -> Fcvtl (u 0 vs));
        form "fcvtn" [ V Arr.D2 ] (Some (V Arr.S2)) (fun vs _ -> Fcvtn (u 0 vs));
        form ~tied:0 "ins.half" [ V Arr.S4; V Arr.S2 ] (Some (V Arr.S4))
          (fun vs _ -> Ins_half (u 0 vs, u 1 vs));
        form "vwiden" [ V Arr.S2 ] (Some (V Arr.S4)) (fun vs _ ->
            Vwiden (u 0 vs));
      ];
      List.concat_map
        (fun fsz ->
          let full = if fsz = Fsz.D then Arr.D2 else Arr.S4 in
          List.concat
            (List.init (lanes_of_fsz fsz) (fun k ->
                 [
                   form ~tied:0
                     (Printf.sprintf "ins.lane.%s[%d]" (Fsz.name fsz) k)
                     [ V full; Sc fsz ] (Some (V full))
                     (fun vs _ -> Ins_lane (fsz, k, u 0 vs, u 1 vs));
                   form ~mem:true ~tied:0
                     (Printf.sprintf "ld1.lane.%s[%d]" (Fsz.name fsz) k)
                     [ V full ] (Some (V full))
                     (fun vs base -> Ld1_lane (fsz, k, u 0 vs, base));
                   form ~mem:true
                     (Printf.sprintf "st1.lane.%s[%d]" (Fsz.name fsz) k)
                     [ V full ] None
                     (fun vs base -> St1_lane (fsz, k, u 0 vs, base));
                 ])))
        fszs;
      List.map
        (fun arr ->
          form ~mem:true
            (Printf.sprintf "ld1r.%s" (Arr.name arr))
            [] (Some (V arr))
            (fun _ base -> Ld1r (arr, base)))
        arrs;
      List.concat_map
        (fun arr ->
          let offsets =
            match arr with Arr.S2 -> [ 0L; 8L ] | _ -> [ 0L; 16L ]
          in
          List.concat_map
            (fun k ->
              [
                form ~mem:true
                  (Printf.sprintf "ldr.%s#%Ld" (Arr.name arr) k)
                  [] (Some (V arr))
                  (fun _ base -> Ldr_vec (arr, base, k));
                form ~mem:true
                  (Printf.sprintf "str.%s#%Ld" (Arr.name arr) k)
                  [ V arr ] None
                  (fun vs base -> Str_vec (arr, base, k, u 0 vs));
              ])
            offsets)
        arrs;
      List.concat_map
        (fun arr ->
          List.map
            (fun (c, cn) ->
              form
                (Printf.sprintf "%s.%s" cn (Arr.name arr))
                [ V arr; V arr ] (Some (M arr))
                (fun vs _ -> Vfcmp (c, arr, u 0 vs, u 1 vs)))
            Vcmp.[ (Eq, "fcmeq"); (Ge, "fcmge"); (Gt, "fcmgt") ])
        Arr.[ S4; D2 ];
      List.map
        (fun (o, on) ->
          form (Printf.sprintf "v%s.16b" on)
            [ M Arr.S4; M Arr.S4 ] (Some (M Arr.S4)) (fun vs _ ->
              Vlogic (o, u 0 vs, u 1 vs)))
        Logic.[ (And, "and"); (Orr, "orr"); (Eor, "eor") ];
      List.map
        (fun arr ->
          form
            (Printf.sprintf "vnot.%s" (Arr.name arr))
            [ M arr ] (Some (M arr))
            (fun vs _ -> Vnot (u 0 vs)))
        Arr.[ S4; D2 ];
      List.map
        (fun arr ->
          form ~tied:0
            (Printf.sprintf "vbit.%s" (Arr.name arr))
            [ V arr; V arr; M arr ] (Some (V arr))
            (fun vs _ -> Vbit (arr, u 0 vs, u 1 vs, u 2 vs)))
        Arr.[ S4; D2 ];
      [
        form "dup.mask.4s" [ Gp Sz.W ] (Some (M Arr.S4)) (fun vs _ ->
            Dup_mask (Arr.S4, u 0 vs));
        form "dup.mask.2d" [ Gp Sz.X ] (Some (M Arr.D2)) (fun vs _ ->
            Dup_mask (Arr.D2, u 0 vs));
      ];
    ]
