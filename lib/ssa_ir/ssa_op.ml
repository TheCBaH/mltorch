(* The operation schema. An operation is operands plus attributes; its result
   types come from {!Ssa_typing}, and an effectful one is sequenced by the
   instruction's effect operand. Every consumer matches this type exhaustively
   and alphabetically: a new opcode is a compile error in the verifier, the
   printer and the interpreter until each handles it. *)

module Convert = struct
  (* Total, pure conversions between working types. *)
  type t =
    | F32_to_f64
    | F64_to_f32
    | I64_to_f32
    | I64_to_f64
    | Index_to_f64
    | Index_to_i64

  let name = function
    | F32_to_f64 -> "f32_to_f64"
    | F64_to_f32 -> "f64_to_f32"
    | I64_to_f32 -> "i64_to_f32"
    | I64_to_f64 -> "i64_to_f64"
    | Index_to_f64 -> "index_to_f64"
    | Index_to_i64 -> "index_to_i64"
end

module Compare = struct
  (* Ordered comparisons. A float comparison with a NaN is false, and signed
     zeros are equal. *)
  type t = Eq | Lt

  let name = function Eq -> "eq" | Lt -> "lt"
end

module I64_op = struct
  (* Modular two's-complement arithmetic: it wraps and cannot fail. Division is
     its own checked operation. *)
  type t = Add | Mul | Sub

  let name = function Add -> "add" | Mul -> "mul" | Sub -> "sub"
end

module Decode = struct
  (* A load's decode: a stored cell to the working type it produces. A
     quantized cell is [scale * (q - zero_point)] with the buffer's parameters,
     the channel taken from the access's C coordinate. *)
  type t =
    | Bf16_to_f64
    | Bool_to_f64
    | F16_to_f64
    | F32_to_f64
    | F64_to_f64
    | I16_dequant
    | I32_to_f64
    | I64
    | I64_to_f64
    | I8_dequant

  let name = function
    | Bf16_to_f64 -> "bf16_to_f64"
    | Bool_to_f64 -> "bool_to_f64"
    | F16_to_f64 -> "f16_to_f64"
    | F32_to_f64 -> "f32_to_f64"
    | F64_to_f64 -> "f64_to_f64"
    | I16_dequant -> "i16_dequant"
    | I32_to_f64 -> "i32_to_f64"
    | I64 -> "i64"
    | I64_to_f64 -> "i64_to_f64"
    | I8_dequant -> "i8_dequant"

  (* The family of buffer a decode reads. *)
  let family = function
    | Bf16_to_f64 -> Ssa_format.Family.Bf16
    | Bool_to_f64 -> Ssa_format.Family.Bool
    | F16_to_f64 -> Ssa_format.Family.F16
    | F32_to_f64 -> Ssa_format.Family.F32
    | F64_to_f64 -> Ssa_format.Family.F64
    | I16_dequant -> Ssa_format.Family.I16
    | I32_to_f64 -> Ssa_format.Family.I32
    | I64 | I64_to_f64 -> Ssa_format.Family.I64
    | I8_dequant -> Ssa_format.Family.I8

  let result = function
    | I64 -> Ssa_type.Scalar Ssa_type.I64
    | Bf16_to_f64 | Bool_to_f64 | F16_to_f64 | F32_to_f64 | F64_to_f64
    | I16_dequant | I32_to_f64 | I64_to_f64 | I8_dequant ->
        Ssa_type.Scalar Ssa_type.F64

  (* The decode that reads a buffer of this format as a float. *)
  let to_float = function
    | Ssa_format.Bf16 -> Bf16_to_f64
    | Ssa_format.Bool -> Bool_to_f64
    | Ssa_format.F16 -> F16_to_f64
    | Ssa_format.F32 -> F32_to_f64
    | Ssa_format.F64 -> F64_to_f64
    | Ssa_format.I16 _ -> I16_dequant
    | Ssa_format.I32 -> I32_to_f64
    | Ssa_format.I64 -> I64_to_f64
    | Ssa_format.I8 _ -> I8_dequant
end

module Encode = struct
  (* A store's encode: a working value to a stored cell. [Bool_nonzero] writes
     [v <> 0.] as a canonical 0/1 byte; [F32_round] rounds to binary32. *)
  type t = Bool_nonzero | F32_round | I64

  let name = function
    | Bool_nonzero -> "bool_nonzero"
    | F32_round -> "f32_round"
    | I64 -> "i64"

  let family = function
    | Bool_nonzero -> Ssa_format.Family.Bool
    | F32_round -> Ssa_format.Family.F32
    | I64 -> Ssa_format.Family.I64

  let operand = function
    | Bool_nonzero | F32_round -> Ssa_type.Scalar Ssa_type.F64
    | I64 -> Ssa_type.Scalar Ssa_type.I64
end

type t =
  | Check_access of { buffer : Ssa_id.Buffer.t; at : Ssa_access.t }
      (** The bounds check a coordinate load performs, with no read: fails with
          the first axis outside, in axis order. *)
  | Check_gather of { raw : Ssa_value.t; extent : int64 }
      (** A gather's raw index must lie in [-extent, extent - 1], checked in the
          int64 domain before it is normalized and narrowed. *)
  | Check_local of { var : Expr.Local_var.t; at : Ssa_value.t; extent : int64 }
      (** A read position must lie in [0, extent) of the local variable it
          names; outside it fails as the reference's unbound local does. *)
  | Check_scan of {
      var : Expr.Local_var.t option;
      row : Ssa_value.t;
      lane : Ssa_value.t;
      row_extent : int64;
      lane_extent : int64;
    }
      (** A scan projection's row, then its lane, against their extents: the row
          wins a simultaneous failure. [var] names a materialized trace, [None]
          an inline scan. *)
  | Const of Ssa_const.t
  | Convert of Convert.t * Ssa_value.t
  | Float_binary of Expr.Value.binary_op * Ssa_value.t * Ssa_value.t
  | Float_compare of Compare.t * Ssa_value.t * Ssa_value.t
  | Float_fma of Ssa_value.t * Ssa_value.t * Ssa_value.t
      (** [a * b + c] with one rounding, at the operands' type: binary64
          [Float.fma], binary32 [fmaf] (see {!Ssa_numerics.fma32}). Only a plan
          whose numerical policy permits contraction contains one. *)
  | Float_max of Ssa_value.t * Ssa_value.t
      (** [Expr.Max_op.Float_max]: [Float.max], with its NaN and signed-zero
          behavior. *)
  | Float_to_i64 of Ssa_value.t
      (** Checked: NaN, an infinity or a value outside [-2^63, 2^63) fails. *)
  | Float_unary of Expr.Value.unary_op * Ssa_value.t
  | I64_arith of I64_op.t * Ssa_value.t * Ssa_value.t
  | I64_compare of Compare.t * Ssa_value.t * Ssa_value.t
  | I64_div of Ssa_value.t * Ssa_value.t
      (** Truncating toward zero; a zero divisor fails first, then
          [min_int / -1]. *)
  | Index_add of Ssa_value.t * Ssa_value.t
      (** Checked: leaving the index domain fails at this operation with its
          operands. *)
  | Index_add_in_domain of Ssa_value.t * Ssa_value.t
      (** [Index_add] whose sum is proved to stay in the index domain: pure and
          total, and valid only where {!Ssa_verify} can re-derive the proof. *)
  | Index_ceil_div of int64 * Ssa_value.t
      (** Mathematical ceiling by a positive literal divisor, never truncation.
      *)
  | Index_clamp_low of Ssa_value.t
  | Index_compare of Compare.t * Ssa_value.t * Ssa_value.t
  | Index_floor_div of int64 * Ssa_value.t
  | Index_max of Ssa_value.t * Ssa_value.t
  | Index_min of Ssa_value.t * Ssa_value.t
  | Index_of_i64 of Ssa_value.t
      (** Narrowing an int64 already inside the index domain; outside it is a
          defect of the program, not a failure row. *)
  | Index_scale of int64 * Ssa_value.t
      (** Checked multiplication by a literal that is itself an index. *)
  | Index_scale_in_domain of int64 * Ssa_value.t
      (** [Index_scale] whose product is proved to stay in the index domain. *)
  | Lanewise of t
      (** A pure scalar operation applied to every lane at once: its operands
          are vectors (or masks) of one width, lane [k] of the result is the
          scalar operation on lane [k] of each operand, evaluated by the same
          function the scalar operation is. Only the float and predicate
          operations of the scalar surface may be lifted, and a lane is never a
          failure site: none of them can fail. *)
  | Load of { buffer : Ssa_id.Buffer.t; at : Ssa_access.t; decode : Decode.t }
  | Load_in_bounds of {
      buffer : Ssa_id.Buffer.t;
      at : Ssa_access.t;
      decode : Decode.t;
    }
      (** [Load] whose coordinate is proved inside the buffer: it checks
          nothing, still reads memory, and is valid only where {!Ssa_verify} can
          re-derive the proof. *)
  | Local_alloc of { slots : int64; var : Expr.Local_var.t option }
      (** A fresh scratch object of [slots] binary64 cells, every cell unset. A
          read of an unset cell is a defect, never an undefined value. *)
  | Local_read of { local : Ssa_value.t; at : Ssa_value.t }
      (** Outside the object fails as an unbound local when the object names a
          variable, and is a defect when it does not. *)
  | Local_write of {
      local : Ssa_value.t;
      at : Ssa_value.t;
      value : Ssa_value.t;
    }
  | Mark of Ssa_mark.t
  | Mark_lanes of { mark : Ssa_mark.t; lanes : Ssa_type.Lanes.t }
      (** [lanes] marks at once: what a vector iteration counts for the scalar
          iterations it covers. *)
  | Meter_charge
      (** One update against the scan meter, before the update body runs:
          exactly the limit succeed and the next fails. *)
  | Meter_release of int64
      (** Gives back the [2 * width] live state a reserve took. *)
  | Meter_reserve of int64
      (** [width] lanes of two rolling rows against the nesting peak of live
          state. *)
  | Meter_reset  (** A fresh meter: the full update budget, no live state. *)
  | Pool_better of Ssa_value.t * Ssa_value.t
      (** [Expr.Max_op.pool_better ~best ~value]: the one predicate a paired
          argmax state advances under. *)
  | Pred_not of Ssa_value.t
  | Pred_or of Ssa_value.t * Ssa_value.t
      (** Both operands are already computed: this is not a short-circuit. A
          lazy right-hand side is an [if]. *)
  | Select of Ssa_value.t * Ssa_value.t * Ssa_value.t
      (** Chooses between values already computed. It never makes an arm lazy:
          an arm with a load, a checked operation or a loop is an [if]. *)
  | Store of {
      buffer : Ssa_id.Buffer.t;
      at : Ssa_access.t;
      encode : Encode.t;
      value : Ssa_value.t;
    }
  | Vec_extract of { lane : Ssa_type.Lane.t; vector : Ssa_value.t }
  | Vec_insert of {
      lane : Ssa_type.Lane.t;
      vector : Ssa_value.t;
      element : Ssa_value.t;
    }
  | Vec_iota of { base : Ssa_value.t; step : int64; lanes : Ssa_type.Lanes.t }
      (** Lane [k] is the binary64 value of [base + k * step], computed in
          int64: [base] is an index and [step] a literal index, so it cannot
          wrap. *)
  | Vec_load of {
      buffer : Ssa_id.Buffer.t;
      at : Ssa_value.t Expr.Coord.t;
      steps : int64 Expr.Coord.t;
      decode : Decode.t;
      lanes : Ssa_type.Lanes.t;
    }
      (** Lane [k] reads the cell at [at + k * steps], axis by axis. Every lane
          is read, and every lane must be inside the buffer: the verifier
          accepts the operation only where the range analysis proves it, so a
          mask never hides a lane that would have failed. *)
  | Vec_splat of { element : Ssa_value.t; lanes : Ssa_type.Lanes.t }
  | Vec_store of {
      buffer : Ssa_id.Buffer.t;
      at : Ssa_value.t Expr.Coord.t;
      steps : int64 Expr.Coord.t;
      encode : Encode.t;
      value : Ssa_value.t;
      lanes : Ssa_type.Lanes.t;
    }
      (** Lane [k] writes the cell at [at + k * steps]; the lanes are written in
          order and the verifier requires all of them in bounds. *)

(* Checked operations and accesses are effectful even when their result looks
   like a scalar expression: they sequence on the effect chain. *)
let rec effectful = function
  | Lanewise op -> effectful op
  | Const _ | Convert _ | Float_binary _ | Float_compare _ | Float_fma _
  | Float_max _ | Float_unary _ | I64_arith _ | I64_compare _
  | Index_add_in_domain _ | Index_ceil_div _ | Index_clamp_low _
  | Index_compare _ | Index_floor_div _ | Index_max _ | Index_min _
  | Index_scale_in_domain _ | Pool_better _ | Pred_not _ | Pred_or _ | Select _
  | Vec_extract _ | Vec_insert _ | Vec_iota _ | Vec_splat _ ->
      false
  | Check_access _ | Check_gather _ | Check_local _ | Check_scan _
  | Float_to_i64 _ | I64_div _ | Index_add _ | Index_of_i64 _ | Index_scale _
  | Load _ | Load_in_bounds _ | Local_alloc _ | Local_read _ | Local_write _
  | Mark _ | Mark_lanes _ | Meter_charge | Meter_release _ | Meter_reserve _
  | Meter_reset | Store _ | Vec_load _ | Vec_store _ ->
      true

let rec operands = function
  | Lanewise op -> operands op
  | Const _ | Local_alloc _ | Mark _ | Mark_lanes _ | Meter_charge
  | Meter_release _ | Meter_reserve _ | Meter_reset ->
      []
  | Check_access { at; _ } | Load { at; _ } | Load_in_bounds { at; _ } ->
      Ssa_access.operands at
  | Check_gather { raw = a; _ }
  | Check_local { at = a; _ }
  | Convert (_, a)
  | Float_to_i64 a
  | Float_unary (_, a)
  | Index_ceil_div (_, a)
  | Index_clamp_low a
  | Index_floor_div (_, a)
  | Index_of_i64 a
  | Index_scale (_, a)
  | Index_scale_in_domain (_, a)
  | Pred_not a ->
      [ a ]
  | Float_binary (_, a, b)
  | Float_compare (_, a, b)
  | Float_max (a, b)
  | I64_arith (_, a, b)
  | I64_compare (_, a, b)
  | I64_div (a, b)
  | Index_add (a, b)
  | Index_add_in_domain (a, b)
  | Index_compare (_, a, b)
  | Index_max (a, b)
  | Index_min (a, b)
  | Pool_better (a, b)
  | Pred_or (a, b) ->
      [ a; b ]
  | Check_scan { row; lane; _ } -> [ row; lane ]
  | Local_read { local; at } -> [ local; at ]
  | Local_write { local; at; value } -> [ local; at; value ]
  | Select (p, a, b) -> [ p; a; b ]
  | Store { at; value; _ } -> Ssa_access.operands at @ [ value ]
  | Float_fma (a, b, c) -> [ a; b; c ]
  | Vec_extract { vector; _ } -> [ vector ]
  | Vec_insert { vector; element; _ } -> [ vector; element ]
  | Vec_iota { base; _ } -> [ base ]
  | Vec_load { at; _ } -> Expr.Coord.to_list at
  | Vec_splat { element; _ } -> [ element ]
  | Vec_store { at; value; _ } -> Expr.Coord.to_list at @ [ value ]

let rec map_operands f = function
  | Lanewise op -> Lanewise (map_operands f op)
  | ( Const _ | Local_alloc _ | Mark _ | Mark_lanes _ | Meter_charge
    | Meter_release _ | Meter_reserve _ | Meter_reset ) as op ->
      op
  | Check_access { buffer; at } ->
      Check_access { buffer; at = Ssa_access.map f at }
  | Check_gather { raw; extent } -> Check_gather { raw = f raw; extent }
  | Check_local { var; at; extent } -> Check_local { var; at = f at; extent }
  | Check_scan { var; row; lane; row_extent; lane_extent } ->
      let row = f row in
      let lane = f lane in
      Check_scan { var; row; lane; row_extent; lane_extent }
  | Local_read { local; at } ->
      let local = f local in
      let at = f at in
      Local_read { local; at }
  | Local_write { local; at; value } ->
      let local = f local in
      let at = f at in
      let value = f value in
      Local_write { local; at; value }
  | Convert (c, a) -> Convert (c, f a)
  | Float_binary (op, a, b) ->
      let a = f a in
      let b = f b in
      Float_binary (op, a, b)
  | Float_compare (c, a, b) ->
      let a = f a in
      let b = f b in
      Float_compare (c, a, b)
  | Float_fma (a, b, c) ->
      let a = f a in
      let b = f b in
      let c = f c in
      Float_fma (a, b, c)
  | Float_max (a, b) ->
      let a = f a in
      let b = f b in
      Float_max (a, b)
  | Float_to_i64 a -> Float_to_i64 (f a)
  | Float_unary (op, a) -> Float_unary (op, f a)
  | I64_arith (op, a, b) ->
      let a = f a in
      let b = f b in
      I64_arith (op, a, b)
  | I64_compare (c, a, b) ->
      let a = f a in
      let b = f b in
      I64_compare (c, a, b)
  | I64_div (a, b) ->
      let a = f a in
      let b = f b in
      I64_div (a, b)
  | Index_add (a, b) ->
      let a = f a in
      let b = f b in
      Index_add (a, b)
  | Index_add_in_domain (a, b) ->
      let a = f a in
      let b = f b in
      Index_add_in_domain (a, b)
  | Index_ceil_div (k, a) -> Index_ceil_div (k, f a)
  | Index_clamp_low a -> Index_clamp_low (f a)
  | Index_compare (c, a, b) ->
      let a = f a in
      let b = f b in
      Index_compare (c, a, b)
  | Index_floor_div (k, a) -> Index_floor_div (k, f a)
  | Index_max (a, b) ->
      let a = f a in
      let b = f b in
      Index_max (a, b)
  | Index_min (a, b) ->
      let a = f a in
      let b = f b in
      Index_min (a, b)
  | Index_of_i64 a -> Index_of_i64 (f a)
  | Index_scale (k, a) -> Index_scale (k, f a)
  | Index_scale_in_domain (k, a) -> Index_scale_in_domain (k, f a)
  | Load_in_bounds { buffer; at; decode } ->
      Load_in_bounds { buffer; at = Ssa_access.map f at; decode }
  | Load { buffer; at; decode } ->
      Load { buffer; at = Ssa_access.map f at; decode }
  | Pool_better (a, b) ->
      let a = f a in
      let b = f b in
      Pool_better (a, b)
  | Pred_not a -> Pred_not (f a)
  | Pred_or (a, b) ->
      let a = f a in
      let b = f b in
      Pred_or (a, b)
  | Select (p, a, b) ->
      let p = f p in
      let a = f a in
      let b = f b in
      Select (p, a, b)
  | Store { buffer; at; encode; value } ->
      let at = Ssa_access.map f at in
      Store { buffer; at; encode; value = f value }
  | Vec_extract { lane; vector } -> Vec_extract { lane; vector = f vector }
  | Vec_insert { lane; vector; element } ->
      let vector = f vector in
      let element = f element in
      Vec_insert { lane; vector; element }
  | Vec_iota { base; step; lanes } -> Vec_iota { base = f base; step; lanes }
  | Vec_load { buffer; at; steps; decode; lanes } ->
      Vec_load { buffer; at = Expr.Coord.map f at; steps; decode; lanes }
  | Vec_splat { element; lanes } -> Vec_splat { element = f element; lanes }
  | Vec_store { buffer; at; steps; encode; value; lanes } ->
      let at = Expr.Coord.map f at in
      Vec_store { buffer; at; steps; encode; value = f value; lanes }

let binary_name = function
  | Expr.Value.Add -> "add"
  | Expr.Value.Div -> "div"
  | Expr.Value.Mul -> "mul"
  | Expr.Value.Sub -> "sub"

let unary_name = function
  | Expr.Value.Cos -> "cos"
  | Expr.Value.Erf -> "erf"
  | Expr.Value.Exp -> "exp"
  | Expr.Value.Log -> "log"
  | Expr.Value.Sin -> "sin"
  | Expr.Value.Sqrt -> "sqrt"
  | Expr.Value.Trunc -> "trunc"

let rec name = function
  | Lanewise op -> "lanes." ^ name op
  | Check_access _ -> "check_access"
  | Check_gather _ -> "check_gather"
  | Check_local _ -> "check_local"
  | Check_scan _ -> "check_scan"
  | Const _ -> "const"
  | Convert (c, _) -> "convert." ^ Convert.name c
  | Float_binary (op, _, _) -> "float." ^ binary_name op
  | Float_compare (c, _, _) -> "float.compare." ^ Compare.name c
  | Float_fma _ -> "float.fma"
  | Float_max _ -> "float.max"
  | Float_to_i64 _ -> "float.to_i64"
  | Float_unary (op, _) -> "float." ^ unary_name op
  | I64_arith (op, _, _) -> "i64." ^ I64_op.name op
  | I64_compare (c, _, _) -> "i64.compare." ^ Compare.name c
  | I64_div _ -> "i64.div"
  | Index_add _ -> "index.add"
  | Index_add_in_domain _ -> "index.add_in_domain"
  | Index_ceil_div _ -> "index.ceil_div"
  | Index_clamp_low _ -> "index.clamp_low"
  | Index_compare (c, _, _) -> "index.compare." ^ Compare.name c
  | Index_floor_div _ -> "index.floor_div"
  | Index_max _ -> "index.max"
  | Index_min _ -> "index.min"
  | Index_of_i64 _ -> "index.of_i64"
  | Index_scale _ -> "index.scale"
  | Index_scale_in_domain _ -> "index.scale_in_domain"
  | Load _ -> "load"
  | Load_in_bounds _ -> "load.in_bounds"
  | Local_alloc _ -> "local.alloc"
  | Local_read _ -> "local.read"
  | Local_write _ -> "local.write"
  | Mark _ -> "mark"
  | Mark_lanes _ -> "mark_lanes"
  | Meter_charge -> "meter.charge"
  | Meter_release _ -> "meter.release"
  | Meter_reserve _ -> "meter.reserve"
  | Meter_reset -> "meter.reset"
  | Pool_better _ -> "pool_better"
  | Pred_not _ -> "pred.not"
  | Pred_or _ -> "pred.or"
  | Select _ -> "select"
  | Store _ -> "store"
  | Vec_extract _ -> "vec.extract"
  | Vec_insert _ -> "vec.insert"
  | Vec_iota _ -> "vec.iota"
  | Vec_load _ -> "vec.load"
  | Vec_splat _ -> "vec.splat"
  | Vec_store _ -> "vec.store"

(* Two pure operations with the same key compute the same value from the same
   operands: the name carries the operator and the conversion, the rest of the
   attributes follow, and the operands are named by id. Constants compare by their
   bits, all NaNs as one, as {!Ssa_const.equal} does. *)
let key op =
  let id (v : Ssa_value.t) = string_of_int (v.Ssa_value.id :> int) in
  let attrs =
    match op with
    | Vec_extract { lane; _ } | Vec_insert { lane; _ } ->
        string_of_int (Ssa_type.Lane.to_int lane)
    | Vec_iota { step; lanes; _ } ->
        Int64.to_string step ^ "x" ^ string_of_int (Ssa_type.Lanes.to_int lanes)
    | Vec_splat { lanes; _ } -> string_of_int (Ssa_type.Lanes.to_int lanes)
    | Const c -> (
        let bits x =
          if Float.is_nan x then "nan"
          else Int64.to_string (Int64.bits_of_float x)
        in
        match c with
        | Ssa_const.F32 x -> "f32:" ^ bits x
        | Ssa_const.F64 x -> "f64:" ^ bits x
        | Ssa_const.I64 x -> "i64:" ^ Int64.to_string x
        | Ssa_const.Index x -> "index:" ^ Int64.to_string x
        | Ssa_const.Pred b -> "pred:" ^ string_of_bool b)
    | Index_add_in_domain _ | Index_ceil_div _ | Index_floor_div _
    | Index_scale _ | Index_scale_in_domain _ -> (
        match op with
        | Index_ceil_div (k, _)
        | Index_floor_div (k, _)
        | Index_scale (k, _)
        | Index_scale_in_domain (k, _) ->
            Int64.to_string k
        | _ -> "")
    | _ -> ""
  in
  String.concat " " ((name op ^ "[" ^ attrs ^ "]") :: List.map id (operands op))
