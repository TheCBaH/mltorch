(* The vector operations of a planned SSA program as generic Machine IR
   vectors: a lane access is one strided access whose byte stride is the
   steps' row-major element offset, and its decode or encode is a lanewise
   conversion, or none for an exact binary32 access. Any other float decode,
   a dequantization included, builds its vector lane by lane through the scalar
   decode; an integer vector and any other encode are a typed refusal. *)

open Ssa_ir
open Machine_ir
open Mir_lower_core

(* The bytes between consecutive lanes: the steps' row-major element offset
   times the element size, refused when it does not fit an i64. *)
let stride st op (e : L.Entry.t) (steps : int64 Expr.Coord.t) =
  let extents = e.L.Entry.buffer.Ssa_buffer.extents in
  let ( let* ) = Option.bind in
  let elements =
    List.fold_left
      (fun acc axis ->
        let* acc = acc in
        let* scaled = Mir_layout.mul acc (Expr.Coord.get extents axis) in
        Mir_layout.add scaled (Expr.Coord.get steps axis))
      (Some 0L) Expr.Axis.all
  in
  match Option.bind elements (Mir_layout.mul e.L.Entry.elem_bytes) with
  | Some s when mutated st Mutation.Vector_stride -> Int64.mul 2L s
  | Some s -> s
  | None -> unsupported st op

let vaccess st op (e : L.Entry.t) at steps lanes elem =
  {
    Mir_op.Vaccess.elem;
    lanes = Mir_type.Lanes.of_int (Ssa_type.Lanes.to_int lanes);
    addr = address st e (Ssa_access.Coord at);
    stride = stride st op e steps;
    align = Mir_type.Elem.bytes elem;
  }

(* The vector element a decode produces: its scalar result's. *)
let result_elem st op (d : Ssa_op.Decode.t) =
  match Ssa_op.Decode.result d with
  | Ssa_type.Scalar Ssa_type.F32 -> Mir_type.Elem.F32
  | Ssa_type.Scalar Ssa_type.F64 -> Mir_type.Elem.F64
  | _ -> unsupported st op

(* A decode with no vector load, one lane at a time: each lane's scalar decode at
   its own address, put into the vector the lanes build up. [decode] is the
   scalar decode of an address. *)
let by_lanes st op (e : L.Entry.t) ~at ~steps ~lanes ~decode d =
  let acc = vaccess st op e at steps lanes (result_elem st op d) in
  let n = Mir_type.Lanes.to_int acc.Mir_op.Vaccess.lanes in
  let lane k =
    let addr =
      if k = 0 then acc.Mir_op.Vaccess.addr
      else
        emit st
          (Mir_op.Ptr_add
             ( acc.Mir_op.Vaccess.addr,
               i64k st (Int64.mul (Int64.of_int k) acc.Mir_op.Vaccess.stride) ))
    in
    decode addr d
  in
  let first = lane 0 in
  let rec go v k =
    if k >= n then v
    else
      go (emit st (Mir_op.Vinsert (Mir_type.Lane.of_int k, v, lane k))) (k + 1)
  in
  go (emit st (Mir_op.Vsplat (acc.Mir_op.Vaccess.lanes, first))) 1

(* A binary32 dequantization of contiguous [i8] or [i16] cells as integer
   lanes: the cells loaded whole registers at a time, each lane sign-extended
   to int32 and converted (exactly, [|q| < 2^15]), then minus the zero point
   and times the scale, each a binary32 lane operation rounding once, as the
   scalar decode. The parameters are the buffer's, or the access's channel's
   (the same in every lane). *)
let dequant32 st op (e : L.Entry.t) ~at ~steps ~lanes ~channel w =
  let acc = vaccess st op e at steps lanes (Mir_type.Elem.Int w) in
  let n = acc.Mir_op.Vaccess.lanes in
  with_role st Mir_origin.Role.Decode;
  let q = emit st (Mir_op.Vload acc) in
  let wide = emit st (Mir_op.Iext (Mir_op.Iext.Sext, Mir_width.W32, q)) in
  let qf = emit st (Mir_op.Fconvert (Mir_op.Fconvert.S32_to_f32, wide)) in
  let s, z = Mir_lower_f32.dequant_params st op e ~channel ~f32:true in
  let splat x = emit st (Mir_op.Vsplat (n, x)) in
  emit st
    (Mir_op.Fbinary
       ( Mir_op.Fbinary.Mul,
         emit st (Mir_op.Fbinary (Mir_op.Fbinary.Sub, qf, splat z)),
         splat s ))

(* Contiguous [f16] or [bf16] cells read straight to binary32 as integer
   lanes, each cell zero-extended to an int32 [h]. A [bf16] is the high half
   of its binary32. An [f16]'s exponent and mantissa move to the binary32's
   ([(h << 17) >> 4]) and times 2^112 rebias exactly, a subnormal included
   (an operand below 2^-126 is not flushed); a product at or above 2^16 came
   from an infinity or a NaN, whose exponent field is then all ones with the
   mantissa kept (a NaN is not quieted); the sign last. *)
let half32 st op (e : L.Entry.t) ~at ~steps ~lanes (d : Ssa_op.Decode.t) =
  let acc = vaccess st op e at steps lanes (Mir_type.Elem.Int Mir_width.W16) in
  let n = acc.Mir_op.Vaccess.lanes in
  with_role st Mir_origin.Role.Decode;
  let h =
    emit st
      (Mir_op.Iext (Mir_op.Iext.Zext, Mir_width.W32, emit st (Mir_op.Vload acc)))
  in
  let splat x = emit st (Mir_op.Vsplat (n, x)) in
  let int k = splat (i32k st k)
  and f32 x = splat (const st (Mir_const.f32 x)) in
  let ints = Mir_type.Vec (Mir_type.Elem.Int Mir_width.W32, n)
  and floats = Mir_type.Vec (Mir_type.Elem.F32, n) in
  let ia o a b = emit st (Mir_op.Iarith (o, a, b)) in
  let as_f v = emit st (Mir_op.Bitcast (floats, v))
  and as_i v = emit st (Mir_op.Bitcast (ints, v)) in
  match d with
  | Ssa_op.Decode.Bf16_to_f32 -> as_f (ia Mir_op.Iarith.Shl h (int 16L))
  | _ ->
      let em =
        ia Mir_op.Iarith.Shr_u (ia Mir_op.Iarith.Shl h (int 17L)) (int 4L)
      in
      let f =
        emit st
          (Mir_op.Fbinary (Mir_op.Fbinary.Mul, as_f em, f32 (Float.ldexp 1. 112)))
      in
      let special = emit st (Mir_op.Fcmp (Mir_op.Fcmp.Le, f32 65536., f)) in
      let all_ones =
        as_f (ia Mir_op.Iarith.Or (as_i f) (as_i (f32 Float.infinity)))
      in
      let r = emit st (Mir_op.Select (special, all_ones, f)) in
      let sign =
        ia Mir_op.Iarith.Shl (ia Mir_op.Iarith.Shr_u h (int 15L)) (int 31L)
      in
      as_f (ia Mir_op.Iarith.Or (as_i r) sign)

let load st op (e : L.Entry.t) ~at ~steps ~lanes ~decode ~channel
    (d : Ssa_op.Decode.t) =
  let vload elem =
    let acc = vaccess st op e at steps lanes elem in
    with_role st Mir_origin.Role.Decode;
    emit st (Mir_op.Vload acc)
  in
  (* contiguous cells filling whole 16-byte registers *)
  let whole w =
    Int64.equal (stride st op e steps) e.L.Entry.elem_bytes
    && Ssa_type.Lanes.to_int lanes * Int64.to_int (Mir_width.bytes w) mod 16 = 0
  in
  match d with
  | Ssa_op.Decode.I8_dequant32 when whole Mir_width.W8 ->
      dequant32 st op e ~at ~steps ~lanes ~channel Mir_width.W8
  | Ssa_op.Decode.I16_dequant32 when whole Mir_width.W16 ->
      dequant32 st op e ~at ~steps ~lanes ~channel Mir_width.W16
  | (Ssa_op.Decode.Bf16_to_f32 | Ssa_op.Decode.F16_to_f32)
    when whole Mir_width.W16 ->
      half32 st op e ~at ~steps ~lanes d
  | Ssa_op.Decode.F32_to_f64 ->
      let v = vload Mir_type.Elem.F32 in
      emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, v))
  | Ssa_op.Decode.F32_exact -> vload Mir_type.Elem.F32
  | Ssa_op.Decode.F64_to_f64 -> vload Mir_type.Elem.F64
  | Ssa_op.Decode.Bf16_to_f32 | Ssa_op.Decode.Bf16_to_f64
  | Ssa_op.Decode.Bool_to_f32 | Ssa_op.Decode.Bool_to_f64
  | Ssa_op.Decode.F16_to_f32 | Ssa_op.Decode.F16_to_f64
  | Ssa_op.Decode.I32_to_f32 | Ssa_op.Decode.I32_to_f64
  | Ssa_op.Decode.I64_to_f64 | Ssa_op.Decode.I16_dequant
  | Ssa_op.Decode.I16_dequant32 | Ssa_op.Decode.I8_dequant
  | Ssa_op.Decode.I8_dequant32 ->
      by_lanes st op e ~at ~steps ~lanes ~decode d
  | Ssa_op.Decode.I64 -> unsupported st op

let store st op (e : L.Entry.t) ~at ~steps ~lanes (enc : Ssa_op.Encode.t) v =
  match enc with
  | Ssa_op.Encode.F32_round ->
      let acc = vaccess st op e at steps lanes Mir_type.Elem.F32 in
      with_role st Mir_origin.Role.Encode;
      let r = emit st (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, v)) in
      emit_unit st (Mir_op.Vstore (acc, r))
  | Ssa_op.Encode.F32_exact ->
      let acc = vaccess st op e at steps lanes Mir_type.Elem.F32 in
      with_role st Mir_origin.Role.Encode;
      emit_unit st (Mir_op.Vstore (acc, v))
  | Ssa_op.Encode.Bf16_round | Ssa_op.Encode.Bool_nonzero | Ssa_op.Encode.I64 ->
      unsupported st op

(* Lane [k] is binary64 [base + k * step]: [base] an index and [k * step]
   below 2^37, so both and their sum are exact in binary64. *)
let iota st ~base ~step ~lanes =
  let n = Ssa_type.Lanes.to_int lanes in
  let lanes = Mir_type.Lanes.of_int n in
  let offsets =
    List.fold_left
      (fun acc k ->
        emit st
          (Mir_op.Vinsert
             ( Mir_type.Lane.of_int k,
               acc,
               f64k st (Int64.to_float (Int64.mul (Int64.of_int k) step)) )))
      (emit st (Mir_op.Vsplat (lanes, f64k st 0.)))
      (List.init n Fun.id)
  in
  let b =
    emit st (Mir_op.Fconvert (Mir_op.Fconvert.S64_to_f64, sext st base))
  in
  emit st
    (Mir_op.Fbinary
       (Mir_op.Fbinary.Add, emit st (Mir_op.Vsplat (lanes, b)), offsets))

(* The operations a [Lanewise] may lift whose scalar expansion is itself
   lanewise: generic float and predicate primitives on its operands only. *)
let liftable (op : Ssa_op.t) =
  match op with
  | Ssa_op.Convert ((Ssa_op.Convert.F32_to_f64 | Ssa_op.Convert.F64_to_f32), _)
  | Ssa_op.Float_binary _ | Ssa_op.Float_compare _ | Ssa_op.Float_fma _
  | Ssa_op.Float_max _
  | Ssa_op.Float_unary ((Expr.Value.Sqrt | Expr.Value.Trunc), _)
  | Ssa_op.Pool_better _ | Ssa_op.Pred_not _ | Ssa_op.Pred_or _
  | Ssa_op.Select _ ->
      true
  | _ -> false
