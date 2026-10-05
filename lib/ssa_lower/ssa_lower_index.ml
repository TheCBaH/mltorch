open Ssa_ir
(* Index lowering. Every [Add] and [Scale] is a checked SSA operation at the
   source's evaluation point, so the first intermediate overflow fails with the
   operands [Expr.Eval.index] reports, even if a later operation would return
   the value to range. No range proof removes a check here: that is the
   analyses' job, over a program that is already correct. *)

open Ssa_lower_ctx

(* A literal the index domain holds; the reference accepts any host [int], so a
   wider one is refused rather than silently narrowed. *)
let literal ctx n =
  let n = Int64.of_int n in
  if not (Ssa_const.in_index_domain n) then
    refuse ctx Ssa_unsupported.Index_literal
  else n

(* A divisor the language accepts is a positive literal; anything else is
   refused, since the reference reports it only when the index is evaluated. *)
let divisor ctx d =
  let d' = Int64.of_int d in
  if d > 0 && Ssa_const.in_index_domain d' then d'
  else refuse ctx Ssa_unsupported.Index_divisor

let rec index : type role.
    Ssa_lower_ctx.t -> role Expr.Index.t -> Ssa_type.index Ssa_builder.value =
 fun ctx -> function
  | Expr.Index.Add (a, b) ->
      let a = index ctx a in
      let b = index ctx b in
      Ssa_builder.index_add ctx.b a b
  | Expr.Index.Assume_position a -> index ctx a
  | Expr.Index.Ceil_div_pos (a, d) ->
      let d = divisor ctx d in
      Ssa_builder.index_ceil_div ctx.b d (index ctx a)
  | Expr.Index.Clamp_low a -> Ssa_builder.index_clamp_low ctx.b (index ctx a)
  | Expr.Index.Floor_div_pos (a, d) ->
      let d = divisor ctx d in
      Ssa_builder.index_floor_div ctx.b d (index ctx a)
  | Expr.Index.Max (a, b) ->
      let a = index ctx a in
      let b = index ctx b in
      Ssa_builder.index_max ctx.b a b
  | Expr.Index.Min (a, b) ->
      let a = index ctx a in
      let b = index ctx b in
      Ssa_builder.index_min ctx.b a b
  | Expr.Index.Const n -> Ssa_builder.index ctx.b (literal ctx n)
  | Expr.Index.Data (src, c, extent) -> gather ctx src c ~extent
  | Expr.Index.Of_position a -> index ctx a
  | Expr.Index.Output axis -> (
      match ctx.axes with
      | Some axes -> Expr.Coord.get axes axis
      | None -> invalid_arg "Ssa_lower: Output outside a nest")
  | Expr.Index.Reduce v -> (
      match Expr.Reduce_var.Map.find_opt v ctx.reducers with
      | Some i -> i
      | None -> invalid_arg "Ssa_lower: reducer is not in scope")
  | Expr.Index.Scale (k, a) ->
      let k = literal ctx k in
      let a = index ctx a in
      Ssa_builder.index_scale ctx.b k a
  | Expr.Index.Zero -> Ssa_builder.index ctx.b 0L

(* A load's coordinate, every component lowered in [Expr.Axis.all] order: the
   order [Expr.Eval] evaluates them, which an explicit list fixes where a
   record's field order would not. *)
and coord : type role.
    Ssa_lower_ctx.t ->
    role Expr.Index.t Expr.Coord.t ->
    Ssa_type.index Ssa_builder.value Expr.Coord.t =
 fun ctx c ->
  let components =
    List.map (fun a -> (a, index ctx (Expr.Coord.get c a))) Expr.Axis.all
  in
  Expr.Coord.of_fn (fun a -> List.assoc a components)

(* [Index.Data]: a runtime gather. The source's raw stored int64 is read (its
   coordinate lowered and bounds-checked like any load), checked against ATen's
   valid range [-extent, extent - 1] in the int64 domain, normalized (a negative
   value gains [extent]), and only then narrowed to an index. Narrowing first
   would let a value near [Int64.min_int] wrap into a spuriously in-range
   index, which is the defect the check before the narrowing exists to
   prevent. *)
and gather : type role.
    Ssa_lower_ctx.t ->
    Expr.Source.t ->
    role Expr.Index.t Expr.Coord.t ->
    extent:int ->
    Ssa_type.index Ssa_builder.value =
 fun ctx src c ~extent ->
  let id = Expr_bridge.id_of_source src in
  let raw =
    match Tensor_id.Map.find_opt id ctx.sources with
    | None -> refuse ctx Ssa_unsupported.Unmaterialized_source
    | Some (Buffer b)
      when Ssa_format.family b.Ssa_buffer.format = Ssa_format.Family.I64 ->
        Ssa_builder.load_i64 ctx.b b.Ssa_buffer.id
          (Ssa_builder.Coord (coord ctx c))
    | Some (Fill_i64 (b, v)) ->
        Ssa_builder.check_access ctx.b b.Ssa_buffer.id
          (Ssa_builder.Coord (coord ctx c));
        Ssa_builder.i64 ctx.b v
    | Some (Buffer b | Fill (b, _)) ->
        refuse ctx
          (Ssa_unsupported.Load_format (Ssa_format.name b.Ssa_buffer.format))
    | Some (Unsupported_format name) ->
        refuse ctx (Ssa_unsupported.Load_format name)
  in
  let bound = Int64.of_int extent in
  Ssa_builder.check_gather ctx.b raw ~extent:bound;
  let zero = Ssa_builder.i64 ctx.b 0L in
  let negative = Ssa_builder.i64_compare ctx.b Ssa_op.Compare.Lt raw zero in
  let shifted =
    Ssa_builder.i64_arith ctx.b Ssa_op.I64_op.Add raw
      (Ssa_builder.i64 ctx.b bound)
  in
  Ssa_builder.index_of_i64 ctx.b (Ssa_builder.select ctx.b negative shifted raw)
