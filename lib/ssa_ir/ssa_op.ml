(* The operation schema. An operation is operands plus attributes; its result
   types come from {!Ssa_typing}, and an effectful one is sequenced by the
   instruction's effect operand. Every consumer matches this type exhaustively
   and alphabetically: a new opcode is a compile error in the verifier, the
   printer and the interpreter until each handles it. *)

module Convert = struct
  (* Total, pure conversions between working types. *)
  type t = F32_to_f64 | F64_to_f32 | Index_to_f64 | Index_to_i64

  let name = function
    | F32_to_f64 -> "f32_to_f64"
    | F64_to_f32 -> "f64_to_f32"
    | Index_to_f64 -> "index_to_f64"
    | Index_to_i64 -> "index_to_i64"
end

module Decode = struct
  (* A load's decode: a stored cell to the working type it produces. *)
  type t = Bool_to_f64 | F32_to_f64 | I64

  let name = function
    | Bool_to_f64 -> "bool_to_f64"
    | F32_to_f64 -> "f32_to_f64"
    | I64 -> "i64"

  (* The format a decode reads. *)
  let format = function
    | Bool_to_f64 -> Ssa_format.Bool
    | F32_to_f64 -> Ssa_format.F32
    | I64 -> Ssa_format.I64

  let result = function
    | Bool_to_f64 | F32_to_f64 -> Ssa_type.Scalar Ssa_type.F64
    | I64 -> Ssa_type.Scalar Ssa_type.I64
end

module Encode = struct
  (* A store's encode: a working value to a stored cell. [Bool_nonzero] writes
     [v <> 0.] as a canonical 0/1 byte; [F32_round] rounds to binary32. *)
  type t = Bool_nonzero | F32_round | I64

  let name = function
    | Bool_nonzero -> "bool_nonzero"
    | F32_round -> "f32_round"
    | I64 -> "i64"

  let format = function
    | Bool_nonzero -> Ssa_format.Bool
    | F32_round -> Ssa_format.F32
    | I64 -> Ssa_format.I64

  let operand = function
    | Bool_nonzero | F32_round -> Ssa_type.Scalar Ssa_type.F64
    | I64 -> Ssa_type.Scalar Ssa_type.I64
end

type t =
  | Const of Ssa_const.t
  | Convert of Convert.t * Ssa_value.t
  | Float_binary of Expr.Value.binary_op * Ssa_value.t * Ssa_value.t
  | Index_add of Ssa_value.t * Ssa_value.t
      (** Checked: leaving the index domain fails at this operation with its
          operands. *)
  | Index_scale of int64 * Ssa_value.t
      (** Checked multiplication by a literal that is itself an index. *)
  | Load of { buffer : Ssa_id.Buffer.t; at : Ssa_access.t; decode : Decode.t }
  | Mark of Ssa_mark.t
  | Store of {
      buffer : Ssa_id.Buffer.t;
      at : Ssa_access.t;
      encode : Encode.t;
      value : Ssa_value.t;
    }

(* Checked operations and accesses are effectful even when their result looks
   like a scalar expression: they sequence on the effect chain. *)
let effectful = function
  | Const _ | Convert _ | Float_binary _ -> false
  | Index_add _ | Index_scale _ | Load _ | Mark _ | Store _ -> true

let operands = function
  | Const _ | Mark _ -> []
  | Convert (_, a) | Index_scale (_, a) -> [ a ]
  | Float_binary (_, a, b) | Index_add (a, b) -> [ a; b ]
  | Load { at; _ } -> Ssa_access.operands at
  | Store { at; value; _ } -> Ssa_access.operands at @ [ value ]

let map_operands f = function
  | Const _ as op -> op
  | Convert (c, a) -> Convert (c, f a)
  | Float_binary (op, a, b) ->
      let a = f a in
      let b = f b in
      Float_binary (op, a, b)
  | Index_add (a, b) ->
      let a = f a in
      let b = f b in
      Index_add (a, b)
  | Index_scale (k, a) -> Index_scale (k, f a)
  | Load { buffer; at; decode } ->
      Load { buffer; at = Ssa_access.map f at; decode }
  | Mark _ as op -> op
  | Store { buffer; at; encode; value } ->
      let at = Ssa_access.map f at in
      Store { buffer; at; encode; value = f value }

let binary_name = function
  | Expr.Value.Add -> "add"
  | Expr.Value.Div -> "div"
  | Expr.Value.Mul -> "mul"
  | Expr.Value.Sub -> "sub"

let name = function
  | Const _ -> "const"
  | Convert (c, _) -> "convert." ^ Convert.name c
  | Float_binary (op, _, _) -> "float." ^ binary_name op
  | Index_add _ -> "index.add"
  | Index_scale _ -> "index.scale"
  | Load _ -> "load"
  | Mark _ -> "mark"
  | Store _ -> "store"
