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

let rec index : type role.
    Ssa_lower_ctx.t -> role Expr.Index.t -> Ssa_type.index Ssa_builder.value =
 fun ctx -> function
  | Expr.Index.Add (a, b) ->
      let a = index ctx a in
      let b = index ctx b in
      Ssa_builder.index_add ctx.b a b
  | Expr.Index.Assume_position a -> index ctx a
  | Expr.Index.Ceil_div_pos _ | Expr.Index.Clamp_low _
  | Expr.Index.Floor_div_pos _ | Expr.Index.Max _ | Expr.Index.Min _ ->
      refuse ctx Ssa_unsupported.Index_operation
  | Expr.Index.Const n -> Ssa_builder.index ctx.b (literal ctx n)
  | Expr.Index.Data _ -> refuse ctx Ssa_unsupported.Gather_index
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
let coord : type role.
    Ssa_lower_ctx.t ->
    role Expr.Index.t Expr.Coord.t ->
    Ssa_type.index Ssa_builder.value Expr.Coord.t =
 fun ctx c ->
  let components =
    List.map (fun a -> (a, index ctx (Expr.Coord.get c a))) Expr.Axis.all
  in
  Expr.Coord.of_fn (fun a -> List.assoc a components)
