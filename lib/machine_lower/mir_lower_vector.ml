(* The vector operations of a planned SSA program as generic Machine IR
   vectors: a lane access is one strided access whose byte stride is the
   steps' row-major element offset, and its decode or encode is a lanewise
   conversion. Only the binary32 and binary64 storage of the vector slice is
   admitted; any other decode or encode is a typed refusal. *)

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

(* A decode with no vector load, one lane at a time: each lane's scalar decode at
   its own address, put into the vector the lanes build up. [decode] is the
   scalar decode of an address. *)
let by_lanes st op (e : L.Entry.t) ~at ~steps ~lanes ~decode d =
  let acc = vaccess st op e at steps lanes Mir_type.Elem.F64 in
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

let load st op (e : L.Entry.t) ~at ~steps ~lanes ~decode (d : Ssa_op.Decode.t) =
  let vload elem =
    let acc = vaccess st op e at steps lanes elem in
    with_role st Mir_origin.Role.Decode;
    emit st (Mir_op.Vload acc)
  in
  match d with
  | Ssa_op.Decode.F32_to_f64 ->
      let v = vload Mir_type.Elem.F32 in
      emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, v))
  | Ssa_op.Decode.F64_to_f64 -> vload Mir_type.Elem.F64
  | Ssa_op.Decode.Bf16_to_f64 | Ssa_op.Decode.Bool_to_f64
  | Ssa_op.Decode.F16_to_f64 | Ssa_op.Decode.I32_to_f64
  | Ssa_op.Decode.I64_to_f64 ->
      by_lanes st op e ~at ~steps ~lanes ~decode d
  | Ssa_op.Decode.I16_dequant | Ssa_op.Decode.I64 | Ssa_op.Decode.I8_dequant ->
      unsupported st op

let store st op (e : L.Entry.t) ~at ~steps ~lanes (enc : Ssa_op.Encode.t) v =
  match enc with
  | Ssa_op.Encode.F32_round ->
      let acc = vaccess st op e at steps lanes Mir_type.Elem.F32 in
      with_role st Mir_origin.Role.Encode;
      let r = emit st (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, v)) in
      emit_unit st (Mir_op.Vstore (acc, r))
  | Ssa_op.Encode.Bool_nonzero | Ssa_op.Encode.I64 -> unsupported st op

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
