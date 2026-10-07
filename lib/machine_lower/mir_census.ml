(* The opcode and observable census: every SSA operation, decode, encode and
   type with its generic expansion, helper requirement or refusal, the plan
   slice that admits it, and each failing condition classified as a typed
   language failure (which ends a generic block at its guard) or a compiler
   invariant (a verified proof or an interpreter defect check, never a failure
   row). The matches are exhaustive: a new SSA constructor is a compile error
   here until it is classified. *)

open Ssa_ir

module Slice = struct
  (* The plan task that admits a row. Ordered by the delivery sequence, which
     is the order a reader follows. *)
  type t = First | Integer | Storage | Locals | Helpers | Vectors

  let name = function
    | First -> "M3"
    | Integer -> "M4.1"
    | Storage -> "M4.2"
    | Locals -> "M4.3"
    | Helpers -> "M4.4"
    | Vectors -> "M11"
end

module Outcome = struct
  type t =
    | Defect  (** a compiler invariant: interpreter defect or verified proof *)
    | Failure of string  (** the language failure kind it reports *)
end

module Condition = struct
  type t = { test : string; outcome : Outcome.t }
end

module Disposition = struct
  type t =
    | Helper of string  (** a named helper call with a modelled algorithm *)
    | Primitive  (** generic primitives only *)
end

module Observable = struct
  type t = Event | Local_state | Meter | Pure | Reads | Writes

  let name = function
    | Event -> "event"
    | Local_state -> "local"
    | Meter -> "meter"
    | Pure -> "-"
    | Reads -> "reads"
    | Writes -> "writes"
end

module Row = struct
  type t = {
    op : string;
    slice : Slice.t;
    disposition : Disposition.t;
    expansion : string;
    conditions : Condition.t list;
    observable : Observable.t;
  }
end

let row ?(conditions = []) ?(observable = Observable.Pure)
    ?(disposition = Disposition.Primitive) op slice expansion =
  { Row.op; slice; disposition; expansion; conditions; observable }

let fail test kind = { Condition.test; outcome = Outcome.Failure kind }
let defect test = { Condition.test; outcome = Outcome.Defect }

let coord_guard =
  fail "coordinate outside its axis, first axis in N T D H W C order"
    "coord_out_of_range"

let decode_slice = function
  | Ssa_op.Decode.F32_to_f64 | Ssa_op.Decode.F64_to_f64 -> Slice.First
  | Ssa_op.Decode.I64 | Ssa_op.Decode.I64_to_f64 -> Slice.Integer
  | Ssa_op.Decode.Bf16_to_f64 | Ssa_op.Decode.Bool_to_f64
  | Ssa_op.Decode.F16_to_f64 | Ssa_op.Decode.I16_dequant
  | Ssa_op.Decode.I32_to_f64 | Ssa_op.Decode.I8_dequant ->
      Slice.Storage

let decode_expansion = function
  | Ssa_op.Decode.Bf16_to_f64 -> "load.i16; shl 16; bitcast.f32; fext"
  | Ssa_op.Decode.Bool_to_f64 -> "load.i8; icmp.ne 0; select 1.0 0.0"
  | Ssa_op.Decode.F16_to_f64 -> "load.i16; binary16 decode by primitives"
  | Ssa_op.Decode.F32_to_f64 -> "load.i32; bitcast.f32; fext.f32.f64"
  | Ssa_op.Decode.F64_to_f64 -> "load.i64; bitcast.f64"
  | Ssa_op.Decode.I16_dequant ->
      "load.i16; sext; sub zero_point; scvt; fmul scale (channel from C)"
  | Ssa_op.Decode.I32_to_f64 -> "load.i32; sext; scvt.i64.f64"
  | Ssa_op.Decode.I64 -> "load.i64"
  | Ssa_op.Decode.I64_to_f64 -> "load.i64; scvt.i64.f64"
  | Ssa_op.Decode.I8_dequant ->
      "load.i8; sext; sub zero_point; scvt; fmul scale (channel from C)"

let encode_slice = function
  | Ssa_op.Encode.F32_round | Ssa_op.Encode.I64 -> Slice.First
  | Ssa_op.Encode.Bool_nonzero -> Slice.Storage

let encode_expansion = function
  | Ssa_op.Encode.Bool_nonzero -> "fcmp.oeq 0.0; select 0 1; store.i8"
  | Ssa_op.Encode.F32_round -> "fround.f64.f32; bitcast.i32; store.i32"
  | Ssa_op.Encode.I64 -> "store.i64"

let max_slice a b = if compare a b >= 0 then a else b

let rec op_row (op : Ssa_op.t) =
  match op with
  | Ssa_op.Check_access { at = Ssa_access.Coord _; _ } ->
      row "check_access.coord" Slice.First "per-axis sle/slt guard chain"
        ~conditions:[ coord_guard ]
  | Ssa_op.Check_access { at = Ssa_access.Flat _; _ } ->
      row "check_access.flat" Slice.First "nothing: no source failure"
        ~conditions:[ defect "flat offset outside the buffer" ]
  | Ssa_op.Check_gather _ ->
      row "check_gather" Slice.Integer "slt -extent; sle extent guards"
        ~conditions:
          [ fail "raw outside [-extent, extent)" "gather_index_out_of_range" ]
  | Ssa_op.Check_local _ ->
      row "check_local" Slice.Locals "slt/sle guard"
        ~conditions:[ fail "position outside [0, extent)" "unbound_local" ]
  | Ssa_op.Check_scan _ ->
      row "check_scan" Slice.Locals "row guard, then lane guard"
        ~conditions:
          [
            fail "row outside its extent (wins)" "scan_projection";
            fail "lane outside its extent" "scan_projection";
          ]
  | Ssa_op.Const _ -> row "const" Slice.First "const (exact bits)"
  | Ssa_op.Convert (c, _) -> (
      let name = "convert." ^ Ssa_op.Convert.name c in
      match c with
      | Ssa_op.Convert.F32_to_f64 -> row name Slice.First "fext.f32.f64"
      | Ssa_op.Convert.F64_to_f32 -> row name Slice.First "fround.f64.f32"
      | Ssa_op.Convert.I64_to_f32 ->
          row name Slice.Integer "scvt.i64.f32 (one rounding)"
      | Ssa_op.Convert.I64_to_f64 -> row name Slice.Integer "scvt.i64.f64"
      | Ssa_op.Convert.Index_to_f64 ->
          row name Slice.First "sext.i64; scvt.i64.f64"
      | Ssa_op.Convert.Index_to_i64 -> row name Slice.First "sext.i64")
  | Ssa_op.Float_binary (o, _, _) ->
      row
        ("float." ^ Ssa_op.binary_name o)
        Slice.First
        ("f" ^ Ssa_op.binary_name o ^ " at the operand precision")
  | Ssa_op.Float_compare (c, _, _) ->
      row
        ("float.compare." ^ Ssa_op.Compare.name c)
        Slice.First
        (match c with
        | Ssa_op.Compare.Eq -> "fcmp.oeq"
        | Ssa_op.Compare.Lt -> "fcmp.olt")
  | Ssa_op.Float_fma _ ->
      row "float.fma" Slice.Integer
        "ffma (one rounding; the planning summary must permit contraction)"
  | Ssa_op.Float_max _ ->
      row "float.max" Slice.Integer
        "fmax (IEEE maximum: NaN propagates, +0 > -0)"
  | Ssa_op.Float_to_i64 _ ->
      row "float.to_i64" Slice.Integer
        "uno guard; |x| = inf guard; range guard; fcvt.f64.i64"
        ~conditions:
          [
            fail "NaN" "i64_from_float_nan";
            fail "an infinity" "i64_from_float_infinite";
            fail "outside [-2^63, 2^63)" "i64_from_float_out_of_range";
          ]
  | Ssa_op.Float_unary (u, _) -> (
      let name = "float." ^ Ssa_op.unary_name u in
      match u with
      | Expr.Value.Sqrt -> row name Slice.First "fsqrt (correctly rounded)"
      | Expr.Value.Trunc -> row name Slice.First "ftrunc"
      | Expr.Value.Erf ->
          row name Slice.Helpers
            "owned: abs; A-S polynomial; call exp (binary32 steps at binary32)"
            ~disposition:(Disposition.Helper "exp")
      | Expr.Value.Cos | Expr.Value.Exp | Expr.Value.Log | Expr.Value.Sin ->
          row name Slice.Helpers "call libm (binary32: fext; call; fround)"
            ~disposition:(Disposition.Helper (Ssa_op.unary_name u)))
  | Ssa_op.I64_arith (o, _, _) ->
      row
        ("i64." ^ Ssa_op.I64_op.name o)
        Slice.First
        (Ssa_op.I64_op.name o ^ ".i64 (modular)")
  | Ssa_op.I64_compare (c, _, _) ->
      row
        ("i64.compare." ^ Ssa_op.Compare.name c)
        Slice.First
        (match c with
        | Ssa_op.Compare.Eq -> "icmp.eq"
        | Ssa_op.Compare.Lt -> "icmp.slt")
  | Ssa_op.I64_div _ ->
      row "i64.div" Slice.Integer "zero guard; min/-1 guard; sdiv"
        ~conditions:
          [
            fail "divisor zero (first)" "i64_division_by_zero";
            fail "min_int / -1" "i64_division_overflow";
          ]
  | Ssa_op.Index_add _ ->
      row "index.add" Slice.First "sext.i64 x2; add; domain guard; trunc.i32"
        ~conditions:[ fail "sum outside the index domain" "index_overflow" ]
  | Ssa_op.Index_add_in_domain _ ->
      row "index.add_in_domain" Slice.First "sext.i64 x2; add; narrow.i32"
        ~conditions:[ defect "sum outside the index domain (proof)" ]
  | Ssa_op.Index_ceil_div _ ->
      row "index.ceil_div" Slice.Integer
        "sdiv/srem by the positive literal; adjust"
  | Ssa_op.Index_clamp_low _ ->
      row "index.clamp_low" Slice.First "icmp.slt 0; select"
  | Ssa_op.Index_compare (c, _, _) ->
      row
        ("index.compare." ^ Ssa_op.Compare.name c)
        Slice.First
        (match c with
        | Ssa_op.Compare.Eq -> "icmp.eq"
        | Ssa_op.Compare.Lt -> "icmp.slt")
  | Ssa_op.Index_floor_div _ ->
      row "index.floor_div" Slice.Integer
        "sdiv/srem by the positive literal; adjust"
  | Ssa_op.Index_max _ -> row "index.max" Slice.First "icmp.slt; select"
  | Ssa_op.Index_min _ -> row "index.min" Slice.First "icmp.slt; select"
  | Ssa_op.Index_of_i64 _ ->
      row "index.of_i64" Slice.Integer "narrow.i32"
        ~conditions:[ defect "int64 outside the index domain" ]
  | Ssa_op.Index_scale _ ->
      row "index.scale" Slice.First "sext.i64; mul; domain guard; trunc.i32"
        ~conditions:[ fail "product outside the index domain" "index_overflow" ]
  | Ssa_op.Index_scale_in_domain _ ->
      row "index.scale_in_domain" Slice.First "sext.i64; mul; narrow.i32"
        ~conditions:[ defect "product outside the index domain (proof)" ]
  | Ssa_op.Lanewise inner ->
      let r = op_row inner in
      row ("lanes." ^ r.Row.op) Slice.Vectors
        "lane-wise generic vector operation"
  | Ssa_op.Load { at = Ssa_access.Coord _; decode; _ } ->
      row "load.coord" (decode_slice decode)
        ("axis guards; row-major byte offset; " ^ decode_expansion decode)
        ~conditions:[ coord_guard ] ~observable:Observable.Reads
  | Ssa_op.Load { at = Ssa_access.Flat _; decode; _ } ->
      row "load.flat" (decode_slice decode)
        ("byte offset; " ^ decode_expansion decode)
        ~conditions:[ defect "flat offset outside the buffer" ]
        ~observable:Observable.Reads
  | Ssa_op.Load_in_bounds { decode; _ } ->
      row "load.in_bounds" (decode_slice decode)
        ("byte offset; " ^ decode_expansion decode)
        ~conditions:[ defect "coordinate outside (proof; byte range checked)" ]
        ~observable:Observable.Reads
  | Ssa_op.Local_alloc _ ->
      row "local.alloc" Slice.Locals "per-site scratch region; undef; addr"
        ~observable:Observable.Local_state
  | Ssa_op.Local_read _ ->
      row "local.read" Slice.Locals "bounds guard when named; load.i64; bitcast"
        ~conditions:
          [
            fail "outside a named local's object" "unbound_local";
            defect "outside an anonymous local's object";
            defect "a cell never written";
          ]
        ~observable:Observable.Local_state
  | Ssa_op.Local_write _ ->
      row "local.write" Slice.Locals "bitcast; store.i64"
        ~conditions:[ defect "outside its object" ]
        ~observable:Observable.Local_state
  | Ssa_op.Mark _ ->
      row "mark" Slice.First "event x1" ~observable:Observable.Event
  | Ssa_op.Mark_lanes _ ->
      row "mark_lanes" Slice.First "event x lanes" ~observable:Observable.Event
  | Ssa_op.Meter_charge ->
      row "meter.charge" Slice.Locals
        "load remaining; slt 0 guard; store remaining-1"
        ~conditions:[ fail "no update left (before the body)" "scan_meter" ]
        ~observable:Observable.Meter
  | Ssa_op.Meter_release _ ->
      row "meter.release" Slice.Locals "load live; sub; store"
        ~observable:Observable.Meter
  | Ssa_op.Meter_reserve _ ->
      row "meter.reserve" Slice.Locals "load live; add; sle limit guard; store"
        ~conditions:[ fail "live state over the peak" "scan_meter" ]
        ~observable:Observable.Meter
  | Ssa_op.Meter_reset ->
      row "meter.reset" Slice.Locals "store limit; store 0"
        ~observable:Observable.Meter
  | Ssa_op.Pool_better _ ->
      row "pool_better" Slice.Integer "fcmp.olt best value; fcmp.uno value; por"
  | Ssa_op.Pred_not _ -> row "pred.not" Slice.First "pnot"
  | Ssa_op.Pred_or _ -> row "pred.or" Slice.First "por (both computed)"
  | Ssa_op.Select _ -> row "select" Slice.First "select (both computed)"
  | Ssa_op.Store { encode; _ } ->
      row "store" (encode_slice encode)
        ("byte offset; " ^ encode_expansion encode)
        ~conditions:[ defect "a store outside its buffer" ]
        ~observable:Observable.Writes
  | Ssa_op.Vec_extract _ -> row "vec.extract" Slice.Vectors "lane extract"
  | Ssa_op.Vec_insert _ -> row "vec.insert" Slice.Vectors "lane insert"
  | Ssa_op.Vec_iota _ ->
      row "vec.iota" Slice.Vectors "lane constants + splat add"
  | Ssa_op.Vec_load { decode; _ } ->
      row "vec.load"
        (max_slice Slice.Vectors (decode_slice decode))
        "contiguous/broadcast/strided lanes"
        ~conditions:[ defect "a lane outside the buffer (proof)" ]
        ~observable:Observable.Reads
  | Ssa_op.Vec_splat _ -> row "vec.splat" Slice.Vectors "splat"
  | Ssa_op.Vec_store _ ->
      row "vec.store" Slice.Vectors "lanes in order"
        ~conditions:[ defect "a lane outside the buffer (proof)" ]
        ~observable:Observable.Writes

(* The machine type of an SSA type: [Index] keeps its checked signed 32-bit
   domain in an i32; [Local] becomes an object pointer; an effect becomes the
   order state. *)
let machine_type : Ssa_type.t -> (Machine_ir.Mir_type.t, Slice.t) result =
  function
  | Ssa_type.Effect -> Ok Machine_ir.Mir_type.Order
  | Ssa_type.Local -> Ok Machine_ir.Mir_type.Ptr
  | Ssa_type.Mask n ->
      Ok
        (Machine_ir.Mir_type.Mask
           (Machine_ir.Mir_type.Lanes.of_int (Ssa_type.Lanes.to_int n)))
  | Ssa_type.Vec (s, n) -> (
      let n = Machine_ir.Mir_type.Lanes.of_int (Ssa_type.Lanes.to_int n) in
      match s with
      | Ssa_type.F32 ->
          Ok (Machine_ir.Mir_type.Vec (Machine_ir.Mir_type.Elem.F32, n))
      | Ssa_type.F64 ->
          Ok (Machine_ir.Mir_type.Vec (Machine_ir.Mir_type.Elem.F64, n))
      | Ssa_type.I64 | Ssa_type.Index | Ssa_type.Pred -> Error Slice.Vectors)
  | Ssa_type.Scalar Ssa_type.F32 -> Ok Machine_ir.Mir_type.F32
  | Ssa_type.Scalar Ssa_type.F64 -> Ok Machine_ir.Mir_type.F64
  | Ssa_type.Scalar Ssa_type.I64 -> Ok Machine_ir.Mir_type.i64
  | Ssa_type.Scalar Ssa_type.Index -> Ok Machine_ir.Mir_type.i32
  | Ssa_type.Scalar Ssa_type.Pred -> Ok Machine_ir.Mir_type.Pred
