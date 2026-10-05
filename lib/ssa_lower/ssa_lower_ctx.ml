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

type t = {
  esc : error Err.Escape.t;
  at : Tensor_id.t;  (** the value being lowered, for a refusal to name *)
  b : Ssa_builder.t;
  sources : source Tensor_id.Map.t;
  axes : Ssa_type.index Ssa_builder.value Expr.Coord.t option;
      (** what [Output a] means here: [None] outside a nest *)
  reducers : Ssa_type.index Ssa_builder.value Expr.Reduce_var.Map.t;
}

let refuse ctx construct =
  Err.Escape.throw ctx.esc
    (`Unsupported { Ssa_unsupported.at = ctx.at; construct } : error)

let buffer_id id = Ssa_id.Buffer.of_int (Tensor_id.to_int id)
