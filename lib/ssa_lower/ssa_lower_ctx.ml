open Ssa_ir
(* What one value's lowering threads: the builder of the region it is in, what
   [Output a] and each reducer mean here, and the buffers a source can name. *)

type error = [ `Unsupported of Ssa_unsupported.t ]

(* What a source names here: a declared buffer, or a reason there is none. *)
type source =
  | Buffer of Ssa_buffer.t
  | Fill of Ssa_buffer.t * float
      (** A [Filled] input: no data, but a read still checks its coordinate
          against the declared shape and then folds to the value a materialized
          fill would decode to. *)
  | Fill_i64 of Ssa_buffer.t * int64
  | Unsupported_format of string

(* A Region local as its reads see it. A slot local is a scratch object of its
   own, with the shape its declaration fixes: a read dispatches on that shape,
   never on the slot count, since a vector of extent 1 has the count of a
   scalar. [Prev_row] is a scan update's [prev]: [width] cells of an object,
   starting at [base]. *)
type local =
  | Prev_row of {
      handle : Ssa_type.local Ssa_builder.value;
      base : Ssa_type.index Ssa_builder.value;
      width : int;
    }
  | Slots of {
      handle : Ssa_type.local Ssa_builder.value;
      count : int;
      shape : Region_local.Shape.t;
    }

type t = {
  esc : error Err.Escape.t;
  at : Tensor_id.t;  (** the value being lowered, for a refusal to name *)
  b : Ssa_builder.t;
  sources : source Tensor_id.Map.t;
  axes : Ssa_type.index Ssa_builder.value Expr.Coord.t option;
      (** what [Output a] means here: [None] outside a nest *)
  reducers : Ssa_type.index Ssa_builder.value Expr.Reduce_var.Map.t;
  locals : local Expr.Local_var.Map.t;
      (** the Region locals already written for the current key *)
  meter : bool ref;
      (** set when the value being lowered reads the scan meter, so a pixel nest
          starts each cell with a fresh meter *)
}

let refuse ctx construct =
  Err.Escape.throw ctx.esc
    (`Unsupported { Ssa_unsupported.at = ctx.at; construct } : error)

let buffer_id id = Ssa_id.Buffer.of_int (Tensor_id.to_int id)

(* A value's store through its boundary's encode. [x] is
   [Kernel.Result_conversion.before_store]'s value; a bfloat16 store takes a
   binary32 operand, which that value already is exactly. *)
let store_result b id ~(result : Kernel.Result_conversion.t) at x =
  match result with
  | Kernel.Result_conversion.Nonzero_bool ->
      Ssa_builder.store_f64 b (buffer_id id) ~encode:Ssa_op.Encode.Bool_nonzero
        at x
  | Kernel.Result_conversion.Round_bf16 ->
      Ssa_builder.store_bf16 b (buffer_id id) at (Ssa_builder.f64_to_f32 b x)
  | Kernel.Result_conversion.Round_f32 ->
      Ssa_builder.store_f64 b (buffer_id id) ~encode:Ssa_op.Encode.F32_round at
        x
