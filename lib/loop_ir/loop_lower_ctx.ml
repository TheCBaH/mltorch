(* The state one value's lowering threads: what is in scope, the ids it may mint,
   and the block statements are appended to. Statements are emitted in
   EVALUATION order, which is what keeps the first failure the one the reference
   reports first. *)

type error = [ `Unsupported of Loop_unsupported.t ]

type supply = {
  mutable arrays : Loop_array.Next.t;
  mutable vars : Loop_var.Next.t;
  mutable temps : Loop_temp.Next.t;
}

(* A Region local as its reads see it. A slot local is the one flat slot array,
   the range the executor's own [Region_slots] layout gives it, and the shape its
   declaration fixes: a read dispatches on that shape, never on the slot count,
   since a vector of extent 1 has the same range as a scalar. [Prev_row] is a
   scan update's [prev]: [width] cells of a trace, or of a rolling row, starting
   at [base]. *)
type local =
  | Prev_row of { array : Loop_array.t; base : Loop_index.t; width : int }
  | Slots of {
      array : Loop_array.t;
      range : Slot.Range.t;
      shape : Region_local.Shape.t;
    }

type source =
  | Buffer of Loop_buffer.t
  | Fill of Loop_buffer.t * float
      (** A [Filled] input: no buffer exists, and a load folds to the value a
          materialized fill would decode to. The record still names the shape a
          load is bounds-checked against. *)
  | Fill_i64 of Loop_buffer.t * int64
      (** A [Filled_i64] input: exact, never through a float. *)

type t = {
  esc : error Err.Escape.t;
  at : Tensor_id.t;  (** the value being lowered, for a refusal to name *)
  sources : source Tensor_id.Map.t;
  axes : Loop_index.coord;  (** what [Output a] means here *)
  reducers : Loop_index.t Expr.Reduce_var.Map.t;
  locals : local Expr.Local_var.Map.t;
      (** the Region locals already written for the current key *)
  ranges : Loop_range.Env.t;
  supply : supply;
  meter : bool ref;
      (** set when the value being lowered reads the scan meter, so its nest
          starts each meter scope with [Reset_meter] *)
  hoisted : Loop_stmt.t list ref;
      (** allocations every evaluation reuses, placed before the first nest *)
  block : Loop_stmt.t list ref;  (** the current block, most recent first *)
}

let refuse ctx construct =
  Err.Escape.throw ctx.esc
    (`Unsupported { Loop_unsupported.at = ctx.at; construct } : error)

let fresh_array ctx =
  let a, next = Loop_array.Next.alloc ctx.supply.arrays in
  ctx.supply.arrays <- next;
  a

let fresh_var ctx =
  let v, next = Loop_var.Next.alloc ctx.supply.vars in
  ctx.supply.vars <- next;
  v

let fresh_temp ctx =
  let t, next = Loop_temp.Next.alloc ctx.supply.temps in
  ctx.supply.temps <- next;
  t

let emit ctx s = ctx.block := s :: !(ctx.block)

(* Runs [f] against a fresh block and returns what it appended, in order. The
   scope (ranges, reducers) is the caller's to extend by passing a derived
   context; only the block is replaced here. *)
let in_block ctx f =
  let inner = { ctx with block = ref [] } in
  let result = f inner in
  (result, List.rev !(inner.block))

let output_buffer (sg : Tensor_sig.t) =
  { Loop_buffer.id = sg.Tensor_sig.id; sg; role = Loop_buffer.Output }

(* A value's storage encode: the F32 boundary rounds on write, the Bool boundary
   writes a working value's truth as a canonical byte. The conversion itself is
   already inside the expression; this only names how the buffer takes it. *)
let stored_of (v : Kernel.Value.t) e : Loop_stored.t =
  match v.Kernel.Value.result with
  | Kernel.Result_conversion.Round_f32 -> Loop_stored.F32 e
  | Kernel.Result_conversion.Nonzero_bool -> Loop_stored.Bool e

let format_name (b : Loop_buffer.t) =
  let sg = b.Loop_buffer.sg in
  let (Payload.Fmt f) = sg.Tensor_sig.fmt in
  Payload.fmt_name f
