(* The oracle adapter: an SSA interpreter failure row as the Machine IR row it
   must equal — kind, decoded static identity and payload. Machine IR records
   are compared through this, never through raw Loop or SSA site numbers. *)

open Machine_ir

let i64 n = Mir_const.i64 (Int64.of_int n)

let row ?invocation (f : Ssa_ir.Ssa_interp.failure) =
  let failure, payload =
    match f with
    | `Coord_out_of_range (source, axis, _, c) ->
        ( Mir_failure.Coord_out_of_range { Mir_failure.Coord.source; axis },
          List.map i64 (Expr.Coord.to_list c) )
    | `Gather_index_out_of_range
        { Expr.Eval.Gather_index_out_of_range.raw; extent } ->
        ( Mir_failure.Gather_index_out_of_range,
          [ Mir_const.i64 raw; i64 extent ] )
    | `I64_division_by_zero -> (Mir_failure.I64_division_by_zero, [])
    | `I64_division_overflow -> (Mir_failure.I64_division_overflow, [])
    | `I64_from_float_infinite -> (Mir_failure.I64_from_float_infinite, [])
    | `I64_from_float_nan -> (Mir_failure.I64_from_float_nan, [])
    | `I64_from_float_out_of_range x ->
        (Mir_failure.I64_from_float_out_of_range, [ Mir_const.f64 x ])
    | `Index_overflow { Expr.Index_overflow.op; lhs; rhs } ->
        ( Mir_failure.Index_overflow
            (match op with
            | `Add -> Mir_failure.Overflow_op.Add
            | `Mul -> Mir_failure.Overflow_op.Mul
            | `Sub ->
                invalid_arg
                  "Mir_ssa_rows: the SSA IR never reports a sub overflow"),
          [ i64 lhs; i64 rhs ] )
    | `Scan_meter (Expr.Scan_meter.Updates_exhausted { limit }) ->
        ( Mir_failure.Scan_meter Mir_failure.Meter.Updates_exhausted,
          [ Mir_const.i64 limit ] )
    | `Scan_meter (Expr.Scan_meter.State_over_limit { limit }) ->
        ( Mir_failure.Scan_meter Mir_failure.Meter.State_over_limit,
          [ i64 limit ] )
    | `Scan_projection e ->
        let ( which,
              {
                Expr.Eval.Scan_bounds.projection =
                  { Expr.Eval.Scan_projection.local; row; lane };
                extent;
              } ) =
          match e with
          | Expr.Eval.Lane_out_of_range b -> (Mir_failure.Scan_axis.Lane, b)
          | Expr.Eval.Row_out_of_range b -> (Mir_failure.Scan_axis.Row, b)
          | Expr.Eval.Unknown_local _ ->
              invalid_arg
                "Mir_ssa_rows: the SSA IR never reports an unknown local"
        in
        ( Mir_failure.Scan_projection { Mir_failure.Scan.which; var = local },
          [ i64 row; i64 lane; i64 extent ] )
    | `Unbound_local v -> (Mir_failure.Unbound_local v, [])
  in
  { Mir_observation.Row.failure; payload; invocation; site = None }
