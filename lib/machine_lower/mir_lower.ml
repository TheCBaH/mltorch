open Ssa_ir
open Machine_ir
include Mir_lower_core
module B = Mir_builder
module L = Mir_layout_map

type result = {
  program : Mir_verify.Generic.t;
  layout : L.Entry.t list;
  planning : Mir_planning.t;
}

let subject p = Digest.to_hex (Digest.string (Ssa_pp.to_string p))

let summary ?target ?fma (plan : Ssa_plan.t) =
  let contraction = Ssa_numerics.contraction_permitted plan.Ssa_plan.numerics in
  let fma =
    match fma with
    | Some f -> f
    | None -> (
        match target with
        | Some (t : Ssa_target.t) when contraction && t.Ssa_target.fma ->
            Mir_planning.Fma.Exact
        | Some t when contraction && t.Ssa_target.relaxed_madd ->
            Mir_planning.Fma.Relaxed_madd
        | Some _ | None -> Mir_planning.Fma.Forbidden)
  in
  let precision =
    match plan.Ssa_plan.precision with
    | Ssa_numerics.Precision.F32 -> Mir_planning.Precision.F32
    | Ssa_numerics.Precision.F64 -> Mir_planning.Precision.F64
  in
  let lanes, schedule, capabilities =
    match target with
    | None -> (Mir_type.Lanes.of_int 1, "scalar", [])
    | Some t ->
        ( Mir_type.Lanes.of_int (Ssa_type.Lanes.to_int t.Ssa_target.lanes),
          Printf.sprintf "%s/row_block=%d" t.Ssa_target.name
            t.Ssa_target.row_block,
          Mir_planning.Capability.Vector_bits
            (Int64.of_int t.Ssa_target.vector_bits)
          ::
          (if fma = Mir_planning.Fma.Exact then
             [ Mir_planning.Capability.Fused_multiply_add ]
           else []) )
  in
  Mir_planning.make
    ~subject:(subject plan.Ssa_plan.program)
    ~policy:(Ssa_numerics.identity plan.Ssa_plan.numerics)
    ~schedule ~precision ~lanes ~fma ~capabilities

(* binary16 bits (in an i32) to binary64 by primitives: a normal or an
   infinity/NaN by its binary32 bit pattern, a subnormal or zero as its
   significand times 2^-24, the sign applied last. *)
let f16_decode st h =
  let i32 x = i32k st x in
  let iar o a b = emit st (Mir_op.Iarith (o, a, b)) in
  let s = iar Mir_op.Iarith.Shr_u h (i32 15L) in
  let e =
    iar Mir_op.Iarith.And (iar Mir_op.Iarith.Shr_u h (i32 10L)) (i32 0x1FL)
  in
  let m = iar Mir_op.Iarith.And h (i32 0x3FFL) in
  let sign = iar Mir_op.Iarith.Shl s (i32 31L) in
  let mant = iar Mir_op.Iarith.Shl m (i32 13L) in
  let f32_of bits =
    emit st
      (Mir_op.Fconvert
         ( Mir_op.Fconvert.F32_to_f64,
           emit st (Mir_op.Bitcast (Mir_type.F32, bits)) ))
  in
  let normal =
    f32_of
      (iar Mir_op.Iarith.Or sign
         (iar Mir_op.Iarith.Or
            (iar Mir_op.Iarith.Shl
               (iar Mir_op.Iarith.Add e (i32 112L))
               (i32 23L))
            mant))
  in
  let special =
    f32_of
      (iar Mir_op.Iarith.Or sign (iar Mir_op.Iarith.Or (i32 0x7F80_0000L) mant))
  in
  let tiny =
    emit st
      (Mir_op.Fbinary
         ( Mir_op.Fbinary.Mul,
           emit st (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, sext st m)),
           f64k st (Float.ldexp 1. (-24)) ))
  in
  let tiny =
    select st
      (icmp st Mir_op.Icmp.Eq s (i32 0L))
      tiny
      (emit st (Mir_op.Funary (Mir_op.Funary.Neg, tiny)))
  in
  select st
    (icmp st Mir_op.Icmp.Eq e (i32 0L))
    tiny
    (select st (icmp st Mir_op.Icmp.Eq e (i32 0x1FL)) special normal)

let zext32 st v = emit st (Mir_op.Iext (Mir_op.Iext.Zext, Mir_width.W32, v))
let sext32 st v = emit st (Mir_op.Iext (Mir_op.Iext.Sext, Mir_width.W32, v))

(* [scale * (q - zero_point)], the parameters per tensor, or per channel from
   the buffer's parameter table at its C coordinate. *)
let dequantize st op (e : L.Entry.t) q ~channel =
  let q = sext st (sext32 st q) in
  let s, z =
    match
      ( Ssa_format.quant e.L.Entry.buffer.Ssa_buffer.format,
        L.Params.view e,
        channel )
    with
    | Some (Ssa_format.Per_tensor { scale; zero_point }), _, _ ->
        (f64k st scale, i64k st (Int64.of_int zero_point))
    | Some (Ssa_format.Per_channel _), Some table, Some c ->
        let base = emit st (Mir_op.Addr table) in
        let c = if mutated st Mutation.Channel_zero then i32k st 0L else c in
        let off =
          emit st (Mir_op.Iarith (Mir_op.Iarith.Mul, sext st c, i64k st 8L))
        in
        let at k =
          emit st
            (Mir_op.Ptr_add
               ( base,
                 emit st (Mir_op.Iarith (Mir_op.Iarith.Add, off, i64k st k)) ))
        in
        let load a =
          emit st
            (Mir_op.Load
               { Mir_op.Access.width = Mir_width.W64; addr = a; align = 8L })
        in
        let scale = emit st (Mir_op.Bitcast (Mir_type.F64, load (at 0L))) in
        (scale, load (at (L.Params.zero_point_offset e)))
    | _ -> unsupported st op
  in
  let diff = emit st (Mir_op.Iarith (Mir_op.Iarith.Sub, q, z)) in
  emit st
    (Mir_op.Fbinary
       ( Mir_op.Fbinary.Mul,
         s,
         emit st (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, diff)) ))

let decode ?entry ?channel st op addr (d : Ssa_op.Decode.t) =
  with_role st Mir_origin.Role.Decode;
  let load width align =
    emit st (Mir_op.Load { Mir_op.Access.width; addr; align })
  in
  match d with
  | Ssa_op.Decode.Bf16_to_f64 ->
      (* the high half of a binary32 *)
      let bits =
        emit st
          (Mir_op.Iarith
             (Mir_op.Iarith.Shl, zext32 st (load Mir_width.W16 2L), i32k st 16L))
      in
      emit st
        (Mir_op.Fconvert
           ( Mir_op.Fconvert.F32_to_f64,
             emit st (Mir_op.Bitcast (Mir_type.F32, bits)) ))
  | Ssa_op.Decode.Bool_to_f64 ->
      let b = zext32 st (load Mir_width.W8 1L) in
      select st
        (icmp st Mir_op.Icmp.Ne b (i32k st 0L))
        (f64k st 1.) (f64k st 0.)
  | Ssa_op.Decode.F16_to_f64 ->
      f16_decode st (zext32 st (load Mir_width.W16 2L))
  | Ssa_op.Decode.I32_to_f64 ->
      emit st
        (Mir_op.Fconvert
           (Mir_op.Fconvert.S64_to_f64, sext st (load Mir_width.W32 4L)))
  | Ssa_op.Decode.I16_dequant | Ssa_op.Decode.I8_dequant -> (
      match entry with
      | None -> unsupported st op
      | Some e ->
          let q =
            match d with
            | Ssa_op.Decode.I8_dequant -> load Mir_width.W8 1L
            | _ -> load Mir_width.W16 2L
          in
          dequantize st op e q ~channel)
  | Ssa_op.Decode.F32_to_f64 ->
      let bits = load Mir_width.W32 4L in
      let f = emit st (Mir_op.Bitcast (Mir_type.F32, bits)) in
      emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, f))
  | Ssa_op.Decode.F64_to_f64 ->
      emit st (Mir_op.Bitcast (Mir_type.F64, load Mir_width.W64 8L))
  | Ssa_op.Decode.I64 -> load Mir_width.W64 8L
  | Ssa_op.Decode.I64_to_f64 ->
      emit st
        (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, load Mir_width.W64 8L))

let encode st addr (enc : Ssa_op.Encode.t) v =
  with_role st Mir_origin.Role.Encode;
  let store width align x =
    emit_unit st (Mir_op.Store ({ Mir_op.Access.width; addr; align }, x))
  in
  match enc with
  | Ssa_op.Encode.F32_round ->
      let r = emit st (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, v)) in
      store Mir_width.W32 4L (emit st (Mir_op.Bitcast (Mir_type.i32, r)))
  | Ssa_op.Encode.I64 -> store Mir_width.W64 8L v
  | Ssa_op.Encode.Bool_nonzero ->
      (* [v <> 0.]: a NaN stores 1 *)
      let zero = emit st (Mir_op.Fcmp (Mir_op.Fcmp.Eq, v, f64k st 0.)) in
      let b = select st zero (i32k st 0L) (i32k st 1L) in
      store Mir_width.W8 1L (emit st (Mir_op.Itrunc (Mir_width.W8, b)))

(* [0 <= x < extent] for an index [x], in i64 so any extent compares. *)
let within st x extent =
  let x = sext st x in
  pand st
    (icmp st Mir_op.Icmp.Sle (i64k st 0L) x)
    (icmp st Mir_op.Icmp.Slt x (i64k st extent))

let local_object st (l : Ssa_value.t) =
  match Hashtbl.find_opt st.locals (l.Ssa_value.id :> int) with
  | Some o -> o
  | None -> refuse st (Refusal.Local_object l.Ssa_value.id)

(* The address of cell [at] of the object [base] points to. *)
let cell st base at =
  with_role st Mir_origin.Role.Address;
  let off =
    emit st (Mir_op.Iarith (Mir_op.Iarith.Mul, sext st at, i64k st 8L))
  in
  emit st (Mir_op.Ptr_add (base, off))

(* A scan meter word: [off] bytes into the meter. *)
let meter_word st off =
  with_role st Mir_origin.Role.Address;
  let base = emit st (Mir_op.Addr (snd L.Scratch.meter).Mir_view.id) in
  emit st (Mir_op.Ptr_add (base, i64k st off))

let load64 st addr =
  emit st
    (Mir_op.Load { Mir_op.Access.width = Mir_width.W64; addr; align = 8L })

let store64 st addr x =
  emit_unit st
    (Mir_op.Store ({ Mir_op.Access.width = Mir_width.W64; addr; align = 8L }, x))

(* A fresh meter: the whole update budget, no live state. *)
let meter_reset st =
  store64 st
    (meter_word st L.Scratch.meter_remaining)
    (i64k st (Expr.Scan_limits.max_updates st.limits));
  store64 st (meter_word st L.Scratch.meter_live) (i64k st 0L)

(* One update: fails when none is left, before the body it guards runs. *)
let meter_charge st =
  let at = meter_word st L.Scratch.meter_remaining in
  with_role st Mir_origin.Role.Compute;
  let r = load64 st at in
  with_role st Mir_origin.Role.Guard;
  guard st
    ~ok:(icmp st Mir_op.Icmp.Slt (i64k st 0L) r)
    (Mir_failure.Scan_meter Mir_failure.Meter.Updates_exhausted)
    (fun () -> [ i64k st (Expr.Scan_limits.max_updates st.limits) ]);
  with_role st Mir_origin.Role.Compute;
  store64 st at (emit st (Mir_op.Iarith (Mir_op.Iarith.Sub, r, i64k st 1L)))

(* A call to a math helper, which the program then declares. *)
let math st fn x =
  if not (List.mem fn st.helpers) then st.helpers <- fn :: st.helpers;
  let d = Mir_math.descriptor fn in
  let signature = function
    | Mir_op.Callee.Helper _ ->
        Some
          {
            Mir_typing.Signature.params = d.Mir_helper.params;
            results = d.Mir_helper.results;
          }
    | Mir_op.Callee.Func _ -> None
  in
  match
    B.op ~origin:st.origin st.bld st.cur ~signature
      (Mir_op.Call (Mir_op.Callee.Helper d.Mir_helper.id, [ x ]))
  with
  | Ok [ r ] -> r
  | _ -> invalid_arg "Mir_lower: a math helper call"

let fbin st o a b = emit st (Mir_op.Fbinary (o, a, b))

(* The error function, owned: the reference's Abramowitz-Stegun form in its
   own operation order, [exp] the only call. At binary32 every step is a
   binary32 operation (each rounds once, as the reference rounds each binary64
   step) and [exp] runs in binary64 between a widening and one rounding. *)
let rec erf st x =
  if
    mutated st Mutation.Erf_single_rounding
    && Mir_type.equal x.Mir_value.ty Mir_type.F32
  then
    emit st
      (Mir_op.Fconvert
         ( Mir_op.Fconvert.F64_to_f32,
           erf st (emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, x))) ))
  else erf_steps st x

and erf_steps st x =
  let ty = x.Mir_value.ty in
  let f32 = Mir_type.equal ty Mir_type.F32 in
  let k c =
    const st
      (if f32 then Mir_const.f32 (Ssa_const.round_f32 c) else Mir_const.f64 c)
  in
  let abs =
    let ity = if f32 then Mir_type.i32 else Mir_type.i64 in
    let mask = if f32 then 0x7FFF_FFFFL else Int64.max_int in
    let bits = emit st (Mir_op.Bitcast (ity, x)) in
    let m = const st { Mir_const.ty = ity; bits = mask } in
    emit st
      (Mir_op.Bitcast (ty, emit st (Mir_op.Iarith (Mir_op.Iarith.And, bits, m))))
  in
  let sign = select st (fcmp st Mir_op.Fcmp.Lt x (k 0.)) (k (-1.)) (k 1.) in
  let mul = fbin st Mir_op.Fbinary.Mul and add = fbin st Mir_op.Fbinary.Add in
  let t =
    fbin st Mir_op.Fbinary.Div (k 1.) (add (k 1.) (mul (k 0.3275911) abs))
  in
  let poly =
    List.fold_left
      (fun acc a -> mul t (add (k a) acc))
      (mul t (k 1.061405429))
      [ -1.453152027; 1.421413741; -0.284496736 ]
  in
  let poly =
    if mutated st Mutation.Erf_distributed then
      add (mul t (k 0.254829592)) (mul t poly)
    else mul t (add (k 0.254829592) poly)
  in
  let sq = emit st (Mir_op.Funary (Mir_op.Funary.Neg, mul abs abs)) in
  let e =
    if f32 then
      emit st
        (Mir_op.Fconvert
           ( Mir_op.Fconvert.F64_to_f32,
             math st Mir_math.Fn.Exp
               (emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, sq))) ))
    else math st Mir_math.Fn.Exp sq
  in
  mul sign (fbin st Mir_op.Fbinary.Sub (k 1.) (mul poly e))

let fbinary = function
  | Expr.Value.Add -> Mir_op.Fbinary.Add
  | Expr.Value.Div -> Mir_op.Fbinary.Div
  | Expr.Value.Mul -> Mir_op.Fbinary.Mul
  | Expr.Value.Sub -> Mir_op.Fbinary.Sub

let compare_op = function
  | Ssa_op.Compare.Eq -> (Mir_op.Icmp.Eq, Mir_op.Fcmp.Eq)
  | Ssa_op.Compare.Lt -> (Mir_op.Icmp.Slt, Mir_op.Fcmp.Lt)

(* One SSA operation; [result] receives its value, if it has one. *)
let rec instr st (i : Ssa_instr.t) =
  let op = i.Ssa_instr.op in
  let v = value st in
  let result m =
    match List.filter (fun r -> not (is_effect r)) i.Ssa_instr.results with
    | [ r ] -> bind st r m
    | _ -> invalid_arg "Mir_lower: a value result is missing"
  in
  with_role st Mir_origin.Role.Compute;
  match op with
  | Ssa_op.Check_access { buffer; at = Ssa_access.Coord c } ->
      coord_guards st (entry_of st buffer) c
  | Ssa_op.Const c ->
      result
        (const st
           (match c with
           | Ssa_const.F32 x -> Mir_const.f32 x
           | Ssa_const.F64 x -> Mir_const.f64 x
           | Ssa_const.I64 x -> Mir_const.i64 x
           | Ssa_const.Index x -> Mir_const.i32 x
           | Ssa_const.Pred b -> Mir_const.pred b))
  | Ssa_op.Convert (c, a) ->
      result
        (match c with
        | Ssa_op.Convert.F32_to_f64 ->
            emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, v a))
        | Ssa_op.Convert.F64_to_f32 ->
            emit st (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, v a))
        | Ssa_op.Convert.Index_to_f64 ->
            emit st
              (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, sext st (v a)))
        | Ssa_op.Convert.Index_to_i64 -> sext st (v a)
        | Ssa_op.Convert.I64_to_f32 ->
            if mutated st Mutation.Double_rounding then
              let d =
                emit st (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, v a))
              in
              emit st (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, d))
            else emit st (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f32, v a))
        | Ssa_op.Convert.I64_to_f64 ->
            emit st (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, v a)))
  | Ssa_op.Float_binary (o, a, b) ->
      let a, b =
        match o with
        | (Expr.Value.Div | Expr.Value.Sub)
          when mutated st Mutation.Operand_order ->
            (b, a)
        | _ -> (a, b)
      in
      result (emit st (Mir_op.Fbinary (fbinary o, v a, v b)))
  | Ssa_op.Float_compare (c, a, b) ->
      result (emit st (Mir_op.Fcmp (snd (compare_op c), v a, v b)))
  | Ssa_op.Float_unary (Expr.Value.Sqrt, a) ->
      result (emit st (Mir_op.Funary (Mir_op.Funary.Sqrt, v a)))
  | Ssa_op.Float_unary (Expr.Value.Trunc, a) ->
      result (emit st (Mir_op.Funary (Mir_op.Funary.Trunc, v a)))
  | Ssa_op.Float_unary (Expr.Value.Erf, a) -> result (erf st (v a))
  | Ssa_op.Float_unary
      ( ((Expr.Value.Cos | Expr.Value.Exp | Expr.Value.Log | Expr.Value.Sin) as u),
        a ) ->
      let fn =
        match u with
        | Expr.Value.Cos -> Mir_math.Fn.Cos
        | Expr.Value.Exp -> Mir_math.Fn.Exp
        | Expr.Value.Log -> Mir_math.Fn.Log
        | _ -> Mir_math.Fn.Sin
      in
      let x = v a in
      (* a binary32 operand: the binary64 result rounded once *)
      result
        (if Mir_type.equal x.Mir_value.ty Mir_type.F32 then
           emit st
             (Mir_op.Fconvert
                ( Mir_op.Fconvert.F64_to_f32,
                  math st fn
                    (emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, x)))
                ))
         else math st fn x)
  | Ssa_op.Check_access { at = Ssa_access.Flat _; _ } ->
      (* no source failure: outside the buffer is a defect, and the access
         that follows reports it *)
      ()
  | Ssa_op.I64_arith (o, a, b) ->
      let o =
        match o with
        | Ssa_op.I64_op.Add -> Mir_op.Iarith.Add
        | Ssa_op.I64_op.Mul -> Mir_op.Iarith.Mul
        | Ssa_op.I64_op.Sub -> Mir_op.Iarith.Sub
      in
      result (emit st (Mir_op.Iarith (o, v a, v b)))
  | Ssa_op.I64_compare (c, a, b) | Ssa_op.Index_compare (c, a, b) ->
      result (icmp st (fst (compare_op c)) (v a) (v b))
  | Ssa_op.Check_gather { raw; extent } ->
      let x = v raw in
      let e = i64k st extent in
      with_role st Mir_origin.Role.Guard;
      let ok =
        pand st
          (icmp st Mir_op.Icmp.Sle (i64k st (Int64.neg extent)) x)
          (icmp st Mir_op.Icmp.Slt x e)
      in
      guard st ~ok Mir_failure.Gather_index_out_of_range (fun () -> [ x; e ])
  | Ssa_op.Float_fma (a, b, c) -> result (emit st (Mir_op.Ffma (v a, v b, v c)))
  | Ssa_op.Float_max (a, b) ->
      result (emit st (Mir_op.Fbinary (Mir_op.Fbinary.Max, v a, v b)))
  | Ssa_op.Float_to_i64 a ->
      let x = v a in
      with_role st Mir_origin.Role.Guard;
      let nan () =
        guard st
          ~ok:(pnot st (fcmp st Mir_op.Fcmp.Unordered x x))
          Mir_failure.I64_from_float_nan
          (fun () -> [])
      in
      let infinite () =
        let inf =
          emit st
            (Mir_op.Pbinary
               ( Mir_op.Pbinary.Or,
                 fcmp st Mir_op.Fcmp.Eq x (f64k st Float.infinity),
                 fcmp st Mir_op.Fcmp.Eq x (f64k st Float.neg_infinity) ))
        in
        guard st ~ok:(pnot st inf) Mir_failure.I64_from_float_infinite
          (fun () -> [])
      in
      let range () =
        let ok =
          pand st
            (fcmp st Mir_op.Fcmp.Le (f64k st (-9223372036854775808.)) x)
            (fcmp st Mir_op.Fcmp.Lt x (f64k st 9223372036854775808.))
        in
        guard st ~ok Mir_failure.I64_from_float_out_of_range (fun () -> [ x ])
      in
      (* NaN, then an infinity, then the range: a NaN or an infinity also
         fails the range test, so this order is the row *)
      if mutated st Mutation.Conversion_order then (
        range ();
        nan ();
        infinite ())
      else (
        nan ();
        infinite ();
        range ());
      with_role st Mir_origin.Role.Compute;
      result (emit st (Mir_op.Fto_sint x))
  | Ssa_op.I64_div (a, b) ->
      let x = v a and y = v b in
      with_role st Mir_origin.Role.Guard;
      guard st
        ~ok:(pnot st (icmp st Mir_op.Icmp.Eq y (i64k st 0L)))
        Mir_failure.I64_division_by_zero
        (fun () -> []);
      let both =
        pand st
          (icmp st Mir_op.Icmp.Eq x (i64k st Int64.min_int))
          (icmp st Mir_op.Icmp.Eq y (i64k st (-1L)))
      in
      guard st ~ok:(pnot st both) Mir_failure.I64_division_overflow (fun () ->
          []);
      with_role st Mir_origin.Role.Compute;
      result (emit st (Mir_op.Idiv (Mir_op.Idiv.Div_s, x, y)))
  | Ssa_op.Index_floor_div (k, a) | Ssa_op.Index_ceil_div (k, a) ->
      (* by a positive literal: the truncated quotient, adjusted by the
         remainder's sign *)
      let n = sext st (v a) and d = i64k st k in
      let q = emit st (Mir_op.Idiv (Mir_op.Idiv.Div_s, n, d)) in
      let r = emit st (Mir_op.Idiv (Mir_op.Idiv.Rem_s, n, d)) in
      let zero = i64k st 0L and one = i64k st 1L in
      let q =
        match op with
        | Ssa_op.Index_floor_div _ ->
            select st
              (icmp st Mir_op.Icmp.Slt r zero)
              (emit st (Mir_op.Iarith (Mir_op.Iarith.Sub, q, one)))
              q
        | _ ->
            select st
              (icmp st Mir_op.Icmp.Slt zero r)
              (emit st (Mir_op.Iarith (Mir_op.Iarith.Add, q, one)))
              q
      in
      result (emit st (Mir_op.Narrow (Mir_width.W32, q)))
  | Ssa_op.Index_of_i64 a ->
      result (emit st (Mir_op.Narrow (Mir_width.W32, v a)))
  | Ssa_op.Pool_better (best, value) ->
      let b = v best and x = v value in
      result
        (emit st
           (Mir_op.Pbinary
              ( Mir_op.Pbinary.Or,
                fcmp st Mir_op.Fcmp.Lt b x,
                fcmp st Mir_op.Fcmp.Unordered x x )))
  | Ssa_op.Index_add (a, b) ->
      let x = sext st (v a) and y = sext st (v b) in
      let sum = emit st (Mir_op.Iarith (Mir_op.Iarith.Add, x, y)) in
      with_role st Mir_origin.Role.Guard;
      guard st ~ok:(in_index_domain st sum)
        (Mir_failure.Index_overflow Mir_failure.Overflow_op.Add) (fun () ->
          [ x; y ]);
      with_role st Mir_origin.Role.Compute;
      result (emit st (Mir_op.Itrunc (Mir_width.W32, sum)))
  | Ssa_op.Index_add_in_domain (a, b) ->
      let sum =
        emit st
          (Mir_op.Iarith (Mir_op.Iarith.Add, sext st (v a), sext st (v b)))
      in
      result (emit st (Mir_op.Narrow (Mir_width.W32, sum)))
  | Ssa_op.Index_scale (k, a) ->
      let kk = i64k st k and y = sext st (v a) in
      let prod = emit st (Mir_op.Iarith (Mir_op.Iarith.Mul, kk, y)) in
      with_role st Mir_origin.Role.Guard;
      guard st ~ok:(in_index_domain st prod)
        (Mir_failure.Index_overflow Mir_failure.Overflow_op.Mul) (fun () ->
          [ kk; y ]);
      with_role st Mir_origin.Role.Compute;
      result (emit st (Mir_op.Itrunc (Mir_width.W32, prod)))
  | Ssa_op.Index_scale_in_domain (k, a) ->
      let prod =
        emit st (Mir_op.Iarith (Mir_op.Iarith.Mul, i64k st k, sext st (v a)))
      in
      result (emit st (Mir_op.Narrow (Mir_width.W32, prod)))
  | Ssa_op.Index_clamp_low a ->
      let x = v a in
      let zero = i32k st 0L in
      result (select st (icmp st Mir_op.Icmp.Slt x zero) zero x)
  | Ssa_op.Index_max (a, b) ->
      let x = v a and y = v b in
      result (select st (icmp st Mir_op.Icmp.Slt x y) y x)
  | Ssa_op.Index_min (a, b) ->
      let x = v a and y = v b in
      result (select st (icmp st Mir_op.Icmp.Slt x y) x y)
  | Ssa_op.Load { buffer; at; decode = d } ->
      let e = entry_of st buffer in
      let channel =
        match at with
        | Ssa_access.Coord c -> Some (value st c.Expr.Coord.c)
        | Ssa_access.Flat _ -> None
      in
      if mutated st Mutation.Eager_load then (
        let x = decode ~entry:e ?channel st op (address st e at) d in
        (match at with
        | Ssa_access.Coord c -> coord_guards st e c
        | Ssa_access.Flat _ -> ());
        result x)
      else (
        (match at with
        | Ssa_access.Coord c -> coord_guards st e c
        | Ssa_access.Flat _ -> ());
        result (decode ~entry:e ?channel st op (address st e at) d))
  | Ssa_op.Load_in_bounds { buffer; at; decode = d } ->
      let e = entry_of st buffer in
      let channel =
        match at with
        | Ssa_access.Coord c -> Some (value st c.Expr.Coord.c)
        | Ssa_access.Flat _ -> None
      in
      result (decode ~entry:e ?channel st op (address st e at) d)
  | Ssa_op.Check_local { var; at; extent } ->
      with_role st Mir_origin.Role.Guard;
      guard st
        ~ok:(within st (v at) extent)
        (Mir_failure.Unbound_local var)
        (fun () -> [])
  | Ssa_op.Check_scan { var; row; lane; row_extent; lane_extent } ->
      (* the row, then the lane: the row wins a simultaneous failure *)
      let r = v row and l = v lane in
      with_role st Mir_origin.Role.Guard;
      List.iter
        (fun (which, x, extent) ->
          guard st ~ok:(within st x extent)
            (Mir_failure.Scan_projection { Mir_failure.Scan.which; var })
            (fun () -> [ sext st r; sext st l; i64k st extent ]))
        [
          (Mir_failure.Scan_axis.Row, r, row_extent);
          (Mir_failure.Scan_axis.Lane, l, lane_extent);
        ]
  | Ssa_op.Local_alloc { slots; var } ->
      let id =
        match List.filter (fun r -> not (is_effect r)) i.Ssa_instr.results with
        | [ r ] -> r.Ssa_value.id
        | _ -> invalid_arg "Mir_lower: a local allocation's result"
      in
      let bytes =
        match Mir_layout.mul slots 8L with
        | Some b -> b
        | None -> refuse st (Refusal.Local_storage id)
      in
      (match Mir_layout.add st.local_bytes bytes with
      | Some total when Int64.compare total L.Scratch.local_limit <= 0 ->
          st.local_bytes <- total
      | _ -> refuse st (Refusal.Local_storage id));
      let region, view = L.Scratch.local (Hashtbl.length st.locals) ~bytes in
      st.objects <- (region, view) :: st.objects;
      Hashtbl.replace st.locals (id :> int) { Local_object.slots; var };
      (* every run of the site begins a fresh object *)
      if not (mutated st Mutation.Stale_local) then
        emit_unit st (Mir_op.Undef view.Mir_view.id);
      result (emit st (Mir_op.Addr view.Mir_view.id))
  | Ssa_op.Local_read { local; at } ->
      let o = local_object st local and x = v at in
      (* outside a named variable's object is its unbound-local failure;
         outside an anonymous one is a defect the byte range check reports *)
      (match o.Local_object.var with
      | Some var ->
          with_role st Mir_origin.Role.Guard;
          guard st ~ok:(within st x o.Local_object.slots)
            (Mir_failure.Unbound_local var) (fun () -> [])
      | None -> ());
      let a = cell st (v local) x in
      with_role st Mir_origin.Role.Compute;
      result (emit st (Mir_op.Bitcast (Mir_type.F64, load64 st a)))
  | Ssa_op.Local_write { local; at; value = x } ->
      ignore (local_object st local);
      let a = cell st (v local) (v at) in
      with_role st Mir_origin.Role.Compute;
      store64 st a (emit st (Mir_op.Bitcast (Mir_type.i64, v x)))
  | Ssa_op.Meter_charge ->
      if mutated st Mutation.Charge_after_body then
        st.deferred_charges <- st.deferred_charges + 1
      else meter_charge st
  | Ssa_op.Meter_release width ->
      let at = meter_word st L.Scratch.meter_live in
      with_role st Mir_origin.Role.Compute;
      let live = load64 st at in
      store64 st at
        (emit st
           (Mir_op.Iarith (Mir_op.Iarith.Sub, live, i64k st (Int64.mul 2L width))))
  | Ssa_op.Meter_reserve width ->
      let at = meter_word st L.Scratch.meter_live in
      with_role st Mir_origin.Role.Compute;
      let live =
        emit st
          (Mir_op.Iarith
             (Mir_op.Iarith.Add, load64 st at, i64k st (Int64.mul 2L width)))
      in
      let limit = Int64.of_int (Expr.Scan_limits.max_state st.limits) in
      with_role st Mir_origin.Role.Guard;
      guard st
        ~ok:(icmp st Mir_op.Icmp.Sle live (i64k st limit))
        (Mir_failure.Scan_meter Mir_failure.Meter.State_over_limit)
        (fun () -> [ i64k st limit ]);
      with_role st Mir_origin.Role.Compute;
      store64 st at live
  | Ssa_op.Meter_reset -> meter_reset st
  | Ssa_op.Mark m ->
      emit_unit st (Mir_op.Event (Mir_census_event.of_mark m, 1L))
  | Ssa_op.Mark_lanes { mark; lanes } ->
      emit_unit st
        (Mir_op.Event
           ( Mir_census_event.of_mark mark,
             Int64.of_int (Ssa_type.Lanes.to_int lanes) ))
  | Ssa_op.Pred_not a -> result (emit st (Mir_op.Pnot (v a)))
  | Ssa_op.Pred_or (a, b) ->
      result (emit st (Mir_op.Pbinary (Mir_op.Pbinary.Or, v a, v b)))
  | Ssa_op.Select (p, a, b) -> result (select st (v p) (v a) (v b))
  | Ssa_op.Store { buffer; at; encode = enc; value = x } ->
      let e = entry_of st buffer in
      encode st (address st e at) enc (v x)
  | Ssa_op.Lanewise inner ->
      (* the scalar expansion, on vector operands *)
      if Mir_lower_vector.liftable inner then instr st { i with op = inner }
      else unsupported st op
  | Ssa_op.Vec_extract { lane; vector } ->
      result
        (emit st
           (Mir_op.Vextract
              (Mir_type.Lane.of_int (Ssa_type.Lane.to_int lane), v vector)))
  | Ssa_op.Vec_insert { lane; vector; element } ->
      result
        (emit st
           (Mir_op.Vinsert
              ( Mir_type.Lane.of_int (Ssa_type.Lane.to_int lane),
                v vector,
                v element )))
  | Ssa_op.Vec_iota { base; step; lanes } ->
      result (Mir_lower_vector.iota st ~base:(v base) ~step ~lanes)
  | Ssa_op.Vec_load { buffer; at; steps; decode = d; lanes } ->
      result
        (Mir_lower_vector.load st op (entry_of st buffer) ~at ~steps ~lanes d)
  | Ssa_op.Vec_splat { element; lanes } ->
      result
        (emit st
           (Mir_op.Vsplat
              (Mir_type.Lanes.of_int (Ssa_type.Lanes.to_int lanes), v element)))
  | Ssa_op.Vec_store { buffer; at; steps; encode = enc; value = x; lanes } ->
      Mir_lower_vector.store st op (entry_of st buffer) ~at ~steps ~lanes enc
        (v x)

let data_values vs = List.filter (fun v -> not (is_effect v)) vs

let edge st (e : Ssa_cfg_edge.t) =
  let target = Hashtbl.find st.heads (e.Ssa_cfg_edge.target :> int) in
  let args = List.map (value st) (data_values e.Ssa_cfg_edge.args) in
  let args =
    if not (mutated st Mutation.Sequential_transfer) then args
    else
      (* argument [i] naming the target's parameter [j < i] reads the value
         parameter [j] was just given, as a sequential copy would *)
      let params = B.param target in
      let given = Array.of_list args in
      List.mapi
        (fun i (a : Mir_value.t) ->
          let rec index k = function
            | [] -> None
            | p :: rest ->
                if Mir_value.equal a p then Some k else index (k + 1) rest
          in
          match index 0 params with Some j when j < i -> given.(j) | _ -> a)
        args
  in
  (target, args)

let block st (b : Ssa_cfg_block.t) =
  st.cur <- Hashtbl.find st.heads (b.Ssa_cfg_block.id :> int);
  List.iteri
    (fun k (i : Ssa_instr.t) ->
      st.origin <-
        {
          Mir_origin.cfg =
            Some
              {
                Mir_origin.Cfg_site.block = (b.Ssa_cfg_block.id :> int);
                instr = k;
              };
          output =
            (match i.Ssa_instr.origin with
            | Ssa_origin.Output o -> Some (Expr.Source.create (o :> int))
            | Ssa_origin.Unknown -> None);
          role = Mir_origin.Role.Compute;
          clone = 0;
        };
      instr st i)
    b.Ssa_cfg_block.body;
  for _ = 1 to st.deferred_charges do
    meter_charge st
  done;
  st.deferred_charges <- 0;
  match b.Ssa_cfg_block.terminator with
  | Ssa_cfg_terminator.Jump e ->
      let target, args = edge st e in
      B.jump st.cur target args
  | Ssa_cfg_terminator.Branch { cond; then_; else_ } ->
      B.branch st.cur (value st cond) (edge st then_) (edge st else_)
  | Ssa_cfg_terminator.Return _ -> B.return st.cur []

(* Binary32 arithmetic or a fused multiply-add anywhere in the graph. *)
let scan (cfg : Ssa_cfg.t) =
  List.fold_left
    (fun (f32, fma) (b : Ssa_cfg_block.t) ->
      List.fold_left
        (fun (f32, fma) (i : Ssa_instr.t) ->
          let is32 (r : Ssa_value.t) =
            match r.Ssa_value.ty with
            | Ssa_type.Scalar Ssa_type.F32 | Ssa_type.Vec (Ssa_type.F32, _) ->
                true
            | _ -> false
          in
          let rec fused = function
            | Ssa_op.Float_fma _ -> true
            | Ssa_op.Lanewise o -> fused o
            | _ -> false
          in
          (* binary32 arithmetic, not a binary32 value a conversion rounds *)
          let rec arithmetic = function
            | Ssa_op.Float_binary _ | Ssa_op.Float_fma _ | Ssa_op.Float_max _
            | Ssa_op.Float_unary _ ->
                true
            | Ssa_op.Lanewise o -> arithmetic o
            | _ -> false
          in
          ( f32
            || arithmetic i.Ssa_instr.op
               && List.exists is32 i.Ssa_instr.results,
            fma || fused i.Ssa_instr.op ))
        (f32, fma) b.Ssa_cfg_block.body)
    (false, false) cfg.Ssa_cfg.blocks

let program ?mutation ~planning (p : Ssa_program.t) =
  Err.Escape.with_escape @@ fun esc ->
  let refuse r = Err.Escape.throw esc r in
  (match Err.payload (Ssa_verify.check p) with
  | Error (`Invalid_program d) -> refuse (Refusal.Invalid_program d)
  | Ok () -> ());
  let cfg =
    match Ssa_cfg_lower.program p with
    | Ok c -> c
    | Error e -> refuse (Refusal.Cfg e)
  in
  (match Err.payload (Ssa_cfg_verify.check cfg) with
  | Error (`Invalid_cfg d) -> refuse (Refusal.Invalid_cfg d)
  | Ok () -> ());
  let f32, fma = scan cfg in
  let planning =
    match Mir_planning.admit planning ~subject:(subject p) ~contracts:fma with
    | Ok s -> s
    | Error m -> refuse (Refusal.Planning m)
  in
  if f32 && planning.Mir_planning.precision = Mir_planning.Precision.F64 then
    refuse (Refusal.Precision planning.Mir_planning.precision);
  let layout =
    match L.entries cfg.Ssa_cfg.buffers with
    | Ok l -> l
    | Error b -> refuse (Refusal.Buffer_layout b)
  in
  let bld = B.create () in
  let entry = B.new_block bld [] in
  let st =
    {
      esc;
      bld;
      layout;
      values = Hashtbl.create 256;
      constants = Hashtbl.create 64;
      heads = Hashtbl.create 32;
      locals = Hashtbl.create 8;
      objects = [];
      local_bytes = 0L;
      limits = p.Ssa_program.scan_limits;
      deferred_charges = 0;
      helpers = [];
      cur = entry;
      origin = Mir_origin.unknown;
      mutation;
    }
  in
  let rpo =
    List.filter_map (Ssa_cfg.find_block cfg) (Ssa_cfg.reverse_postorder cfg)
  in
  List.iter
    (fun (b : Ssa_cfg_block.t) ->
      let params = data_values b.Ssa_cfg_block.params in
      let mb =
        B.new_block bld
          (List.map
             (fun (v : Ssa_value.t) -> machine_type st v.Ssa_value.ty)
             params)
      in
      List.iter2 (bind st) params (B.param mb);
      Hashtbl.replace st.heads (b.Ssa_cfg_block.id :> int) mb)
    rpo;
  (* the invocation's meter starts fresh *)
  let metered =
    List.exists
      (fun (b : Ssa_cfg_block.t) ->
        List.exists
          (fun (i : Ssa_instr.t) ->
            match
              (Mir_census.op_row i.Ssa_instr.op).Mir_census.Row.observable
            with
            | Mir_census.Observable.Meter -> true
            | _ -> false)
          b.Ssa_cfg_block.body)
      cfg.Ssa_cfg.blocks
  in
  if metered then meter_reset st;
  B.jump st.cur (Hashtbl.find st.heads (cfg.Ssa_cfg.entry :> int)) [];
  (* blocks in reverse postorder, so every use follows its definition *)
  List.iter (block st) rpo;
  let fn = Mir_id.Func.of_int 0 in
  let f = B.func bld ~id:fn ~name:"kernel" ~entry ~results:[] in
  let scratch =
    List.rev st.objects @ if metered then [ L.Scratch.meter ] else []
  in
  let mir =
    B.program
      ~regions:
        (List.map L.region layout
        @ List.concat_map (fun e -> List.map fst (L.Params.objects e)) layout
        @ List.map fst scratch)
      ~views:
        (List.map L.view layout
        @ List.concat_map (fun e -> List.map snd (L.Params.objects e)) layout
        @ List.map snd scratch)
      ~helpers:
        (List.filter_map
           (fun fn ->
             if List.mem fn st.helpers then Some (Mir_math.descriptor fn)
             else None)
           Mir_math.Fn.all)
      ~planning [ f ] ~main:fn
  in
  match Err.payload (Mir_verify.generic mir) with
  | Ok g -> { program = g; layout; planning }
  | Error d -> refuse (Refusal.Invalid_lowering d)
