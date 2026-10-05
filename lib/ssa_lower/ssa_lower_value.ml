open Ssa_ir
(* Lowering a float [Expr.Value.t] to SSA values, in evaluation order: every
   check, load and reduction is built before the operation that consumes it. *)

open Ssa_lower_ctx

(* The decode a load of this source performs, or a refusal naming the format. *)
let decode ctx (b : Ssa_buffer.t) =
  match b.Ssa_buffer.format with
  | Ssa_format.Bool -> Ssa_op.Decode.Bool_to_f64
  | Ssa_format.F32 -> Ssa_op.Decode.F32_to_f64
  | Ssa_format.I64 -> refuse ctx (Ssa_unsupported.Load_format "i64")

let rec value ctx : float Expr.Value.t -> Ssa_type.f64 Ssa_builder.value =
  function
  | Expr.Value.Binary (op, a, b) ->
      let a = value ctx a in
      let b = value ctx b in
      Ssa_builder.f64_binary ctx.b op a b
  | Expr.Value.Const x -> Ssa_builder.f64 ctx.b x
  | Expr.Value.I64_to_float _ -> refuse ctx Ssa_unsupported.Int64_value
  | Expr.Value.Intrinsic _ -> refuse ctx Ssa_unsupported.Intrinsic
  | Expr.Value.Load (src, coord) -> load ctx src coord
  | Expr.Value.Local _ | Expr.Value.Local_at _ ->
      refuse ctx Ssa_unsupported.Local_read
  | Expr.Value.Local_scan_at _ | Expr.Value.Scan_at _ ->
      refuse ctx Ssa_unsupported.Scan_read
  | Expr.Value.Reduce r -> reduce ctx r
  | Expr.Value.Round_f32 a ->
      let a = value ctx a in
      Ssa_builder.f32_to_f64 ctx.b (Ssa_builder.f64_to_f32 ctx.b a)
  | Expr.Value.Select _ -> refuse ctx Ssa_unsupported.Select
  | Expr.Value.Unary _ -> refuse ctx Ssa_unsupported.Unary_operation
  | Expr.Value.Value_of_index i ->
      Ssa_builder.index_to_f64 ctx.b (Ssa_lower_index.index ctx i)

and load ctx src coord =
  match Tensor_id.Map.find_opt (Expr_bridge.id_of_source src) ctx.sources with
  | None -> refuse ctx Ssa_unsupported.Unmaterialized_source
  | Some Filled -> refuse ctx Ssa_unsupported.Filled_input
  | Some (Unsupported_format name) ->
      refuse ctx (Ssa_unsupported.Load_format name)
  | Some (Buffer b) ->
      let decode = decode ctx b in
      let coord = Ssa_lower_index.coord ctx coord in
      Ssa_builder.load_f64 ctx.b b.Ssa_buffer.id ~decode
        (Ssa_builder.Coord coord)

(* The ordered half-open left fold [Expr.Eval] specifies, seeded at [+0.]
   (never the first element, which would make an all-[-0.] sum [-0.]). One
   reduction mark per term, before the term is evaluated. *)
and reduce ctx (r : Expr.Reduction.t) =
  match r.Expr.Reduction.kind with
  | Expr.Reduction.Sum ->
      let lo = Ssa_lower_index.index ctx r.Expr.Reduction.lo in
      let hi = Ssa_lower_index.index ctx r.Expr.Reduction.hi in
      let seed = Ssa_builder.f64 ctx.b 0. in
      Ssa_builder.ordered_sum ctx.b ~lo ~hi ~seed (fun b i ->
          Ssa_builder.mark b Ssa_mark.Reduction;
          let inner =
            {
              ctx with
              b;
              reducers =
                Expr.Reduce_var.Map.add r.Expr.Reduction.var i ctx.reducers;
            }
          in
          value inner r.Expr.Reduction.body)
  | Expr.Reduction.Max -> refuse ctx Ssa_unsupported.Max_reduction
  | Expr.Reduction.Argmax_index | Expr.Reduction.Argmax_value ->
      refuse ctx Ssa_unsupported.Argmax_reduction
