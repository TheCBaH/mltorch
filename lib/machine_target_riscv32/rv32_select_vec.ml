(* Selection of register-wide vector operations as Zve32f forms. By now every
   vector is a slice of the register width ({!Mir_vsplit}): four binary32 or
   int32 lanes, or a mask of four, at 16 bytes; sixteen at 64 (a group of
   four registers), where a lane insert or extract, an extension and a narrow
   integer load are refused. IEEE maximum, truncation and the unordered
   compare are the scalar sequences on lanes; a broadcast is a scalar load and
   a splat; an access at any other stride is [vlse32.v] or [vsse32.v] with the
   stride in a register. Integer lane arithmetic, other lane widths, and a
   vector through a slice or a join are refused. *)

open Machine_ir
open Rv32_select_st
open Rv32_select_seq
module V = Rv32_vop

let vec st ?result ty op = emit st ?result ty (Rv32_op.Vec op)
let f32x4 = V.f32x4
let mask4 = V.mask4

(* The lanes of a vector operand. *)
let lanes_of (v : Mir_value.t) =
  match V.lanes_of v.Mir_value.ty with
  | Some n -> n
  | None -> invalid_arg "Rv32_select_vec: a vector operand's lanes"

let cmp st ?result c (a : Mir_value.t) b =
  vec st ?result (Mir_type.Mask (lanes_of a)) (V.Vfcmp (c, a, b))

(* where neither lane operand is a NaN *)
let ordered st a b =
  vec st
    (Mir_type.Mask (lanes_of a))
    (V.Vlogic
       ( Rv32_kind.Logic.And,
         cmp st Rv32_kind.Fcmp.Eq a a,
         cmp st Rv32_kind.Fcmp.Eq b b ))

let splat_f32 st n bits =
  let f =
    emit st Mir_type.F32 (Rv32_op.Fmv_w_x (materialize st Mir_type.i32 bits))
  in
  vec st (V.f32v n) (V.Vfmv_v_f (V.f32v n, f))

let vector st (i : Mir_op.t Mir_instr.t) =
  let r () =
    match i.Mir_instr.results with
    | [ r ] -> r
    | _ -> invalid_arg "Rv32_select_vec: a vector result"
  in
  let def op = ignore (vec st ~result:(r ()) (ty (r ())) op) in
  let unsupported () =
    refuse st (Refusal.Operation (Mir_op.name i.Mir_instr.op))
  in
  let order = i.Mir_instr.order in
  let u = u st in
  let int_slice (v : Mir_value.t) = Hashtbl.find_opt st.int_slices (key v) in
  let shift_count (v : Mir_value.t) =
    Hashtbl.find_opt st.shift_counts (key v)
  in
  (* a fused slice or count lives in no register: its consumer reads it *)
  let lane_type (v : Mir_value.t) =
    match Rv32.shape (ty v) with
    | Some (Mir_target.Bank.Vector, _) -> ()
    | _ when Option.is_some (int_slice v) || Option.is_some (shift_count v) ->
        ()
    | _ -> refuse st (Refusal.Width (ty v))
  in
  List.iter lane_type
    (List.filter
       (fun v ->
         match ty v with Mir_type.Vec _ | Mir_type.Mask _ -> true | _ -> false)
       (Mir_op.operands i.Mir_instr.op @ i.Mir_instr.results));
  match i.Mir_instr.op with
  | Mir_op.Vslice _ when Option.is_some (int_slice (r ())) ->
      (* made by the extensions that read it *)
      ()
  | Mir_op.Iext (k, Mir_width.W32, s) when Option.is_some (int_slice s) ->
      let first, src = Option.get (int_slice s) in
      let from =
        match ty src with
        | Mir_type.Vec (Mir_type.Elem.Int w, _) -> w
        | t -> refuse st (Refusal.Width t)
      in
      def (V.Vext { signed = k = Mir_op.Iext.Sext; from; first; src = u src })
  | Mir_op.Vsplat _ when Option.is_some (shift_count (r ())) ->
      (* the count of the shifts that read it *)
      ()
  | Mir_op.Iarith (o, a, c) when Mir_op.Iarith.is_shift o && V.is_i32v (ty a)
    -> (
      match shift_count c with
      | None -> unsupported ()
      | Some k -> (
          let k = Int64.to_int k in
          match o with
          | Mir_op.Iarith.Shl -> def (V.Vshift (Rv32_kind.Shift.Sll, k, u a))
          | _ when k = 0 -> def (V.Vmv (u a))
          | Mir_op.Iarith.Shr_s -> def (V.Vshift (Rv32_kind.Shift.Sra, k, u a))
          | _ -> def (V.Vshift (Rv32_kind.Shift.Srl, k, u a))))
  | Mir_op.Iarith
      (((Mir_op.Iarith.And | Mir_op.Iarith.Or | Mir_op.Iarith.Xor) as o), a, b)
    when V.is_i32v (ty a) ->
      def
        (V.Vlogic
           ( (match o with
             | Mir_op.Iarith.And -> Rv32_kind.Logic.And
             | Mir_op.Iarith.Or -> Rv32_kind.Logic.Or
             | _ -> Rv32_kind.Logic.Xor),
             u a,
             u b ))
  | Mir_op.Bitcast (t, a) -> def (V.Vcast (t, u a))
  | Mir_op.Copy a -> def (V.Vmv (u a))
  | Mir_op.Fbinary (o, a, b) -> (
      let a = u a and b = u b in
      match o with
      | Mir_op.Fbinary.Add -> def (V.Vfbin (Rv32_kind.Fop.Add, a, b))
      | Mir_op.Fbinary.Div -> def (V.Vfbin (Rv32_kind.Fop.Div, a, b))
      | Mir_op.Fbinary.Max when mutated st Mutation.Number_max ->
          def (V.Vfbin (Rv32_kind.Fop.Max, a, b))
      | Mir_op.Fbinary.Max ->
          (* maximumNumber, with the operands' sum where either is a NaN *)
          let m = vec st (ty a) (V.Vfbin (Rv32_kind.Fop.Max, a, b)) in
          let nan = vec st (ty a) (V.Vfbin (Rv32_kind.Fop.Add, a, b)) in
          def (V.Vmerge (nan, m, ordered st a b))
      | Mir_op.Fbinary.Mul -> def (V.Vfbin (Rv32_kind.Fop.Mul, a, b))
      | Mir_op.Fbinary.Sub -> def (V.Vfbin (Rv32_kind.Fop.Sub, a, b)))
  | Mir_op.Fcmp (c, a, b) -> (
      let a = u a and b = u b in
      match c with
      | Mir_op.Fcmp.Eq -> def (V.Vfcmp (Rv32_kind.Fcmp.Eq, a, b))
      | Mir_op.Fcmp.Le -> def (V.Vfcmp (Rv32_kind.Fcmp.Le, a, b))
      | Mir_op.Fcmp.Lt -> def (V.Vfcmp (Rv32_kind.Fcmp.Lt, a, b))
      | Mir_op.Fcmp.Unordered -> def (V.Vnot (ordered st a b)))
  | Mir_op.Fconvert (Mir_op.Fconvert.S32_to_f32, a) -> def (V.Vfcvt_f_x (u a))
  | Mir_op.Ffma (a, b, c) -> def (V.Vfmacc (u c, u a, u b))
  | Mir_op.Funary (o, a) -> (
      let a = u a in
      match o with
      | Mir_op.Funary.Neg -> def (V.Vsgnj (Rv32_kind.Sgnj.Jn, a, a))
      | Mir_op.Funary.Sqrt -> def (V.Vfsqrt a)
      | Mir_op.Funary.Trunc ->
          (* below 2^23 in magnitude through int32 lanes and back, the sign
             restored; else the lane itself *)
          let n = lanes_of a in
          let t =
            vec st (ty a)
              (V.Vsgnj
                 ( Rv32_kind.Sgnj.J,
                   vec st (ty a)
                     (V.Vfcvt_f_x (vec st (V.i32v n) (V.Vfcvt_x_f a))),
                   a ))
          in
          if mutated st Mutation.Unguarded_trunc then def (V.Vmv t)
          else
            let mag = vec st (ty a) (V.Vsgnj (Rv32_kind.Sgnj.Jx, a, a)) in
            let small =
              cmp st Rv32_kind.Fcmp.Lt mag (splat_f32 st n 0x4B00_0000L)
            in
            def (V.Vmerge (a, t, small)))
  | Mir_op.Pbinary (o, a, b) ->
      def
        (V.Vlogic
           ( (match o with
             | Mir_op.Pbinary.And -> Rv32_kind.Logic.And
             | Mir_op.Pbinary.Or -> Rv32_kind.Logic.Or
             | Mir_op.Pbinary.Xor -> Rv32_kind.Logic.Xor),
             u a,
             u b ))
  | Mir_op.Pnot a -> def (V.Vnot (u a))
  | Mir_op.Select (m, a, b) -> def (V.Vmerge (u b, u a, u m))
  | Mir_op.Vextract (lane, a) ->
      if not (Mir_type.equal (ty a) f32x4) then refuse st (Refusal.Width (ty a));
      def (V.Vfmv_f_s (Mir_type.Lane.to_int lane, u a))
  | Mir_op.Vinsert (lane, a, x) ->
      if not (Mir_type.equal (ty a) f32x4) then refuse st (Refusal.Width (ty a));
      def (V.Vinsert (Mir_type.Lane.to_int lane, u a, u x))
  | Mir_op.Vsplat (_, x) -> (
      match ty x with
      | Mir_type.F32 -> def (V.Vfmv_v_f (ty (r ()), u x))
      | Mir_type.Int Mir_width.W32 -> def (V.Vmv_v_x (ty (r ()), u x))
      | Mir_type.Pred ->
          (* 0 or 1 as 0 or all ones, then to every lane *)
          let w = emit st Mir_type.i32 (Rv32_op.Zext (u x)) in
          let ones =
            emit st Mir_type.i32
              (Rv32_op.Alu (Rv32_op.Alu.Sub, materialize st Mir_type.i32 0L, w))
          in
          def (V.Vmv_v_x (ty (r ()), ones))
      | t -> refuse st (Refusal.Width t))
  | Mir_op.Vload { Mir_op.Vaccess.addr; stride; elem; _ } -> (
      let t = ty (r ()) in
      let addr = u addr in
      match (stride, elem) with
      | 1L, Mir_type.Elem.Int Mir_width.W8 | 2L, Mir_type.Elem.Int Mir_width.W16
        ->
          let w =
            match elem with Mir_type.Elem.Int w -> w | _ -> Mir_width.W8
          in
          if not (Mir_type.equal t (V.int_register w)) then
            refuse st (Refusal.Width t);
          push st ?order
            [ r () ]
            (Mir_sel.Op.Machine (Rv32_op.Vec (V.Vle_int (w, addr))))
      | 4L, (Mir_type.Elem.F32 | Mir_type.Elem.Int Mir_width.W32) ->
          push st ?order
            [ r () ]
            (Mir_sel.Op.Machine (Rv32_op.Vec (V.Vle32 (t, addr))))
      | 0L, Mir_type.Elem.F32 ->
          let f = fresh st Mir_type.F32 in
          push st ?order [ f ] (Mir_sel.Op.Machine (Rv32_op.Flw (addr, 0L)));
          def (V.Vfmv_v_f (t, f))
      | 0L, Mir_type.Elem.Int Mir_width.W32 ->
          let w = fresh st Mir_type.i32 in
          push st ?order [ w ]
            (Mir_sel.Op.Machine (Rv32_op.Load (Rv32_op.Msz.W, addr, 0L)));
          def (V.Vmv_v_x (t, w))
      | s, (Mir_type.Elem.F32 | Mir_type.Elem.Int Mir_width.W32) ->
          let stride = materialize st Mir_type.i32 s in
          push st ?order
            [ r () ]
            (Mir_sel.Op.Machine (Rv32_op.Vec (V.Vlse32 (t, addr, stride))))
      | _ -> refuse st (Refusal.Width t))
  | Mir_op.Vstore ({ Mir_op.Vaccess.addr; stride; _ }, x) -> (
      let addr = u addr and x = u x in
      match stride with
      | 4L ->
          push st ?order []
            (Mir_sel.Op.Machine (Rv32_op.Vec (V.Vse32 (addr, x))))
      | 0L -> unsupported ()
      | s ->
          let stride = materialize st Mir_type.i32 s in
          push st ?order []
            (Mir_sel.Op.Machine (Rv32_op.Vec (V.Vsse32 (addr, stride, x)))))
  | _ -> unsupported ()
