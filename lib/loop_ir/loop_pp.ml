(* Names are assigned on first appearance, per call. The traversal order is the
   printed order, so the assignment is a function of the text alone. *)
type names = {
  vars : (int, int) Hashtbl.t;
  temps : (int, int) Hashtbl.t;
  index_temps : (int, int) Hashtbl.t;
  arrays : (int, int) Hashtbl.t;
}

let fresh_names () =
  {
    vars = Hashtbl.create 8;
    temps = Hashtbl.create 8;
    index_temps = Hashtbl.create 8;
    arrays = Hashtbl.create 8;
  }

let ordinal table key =
  match Hashtbl.find_opt table key with
  | Some n -> n
  | None ->
      let n = Hashtbl.length table in
      Hashtbl.add table key n;
      n

let pp_var nm fmt v = Fmt.pf fmt "i%d" (ordinal nm.vars (Loop_var.to_int v))
let pp_temp nm fmt t = Fmt.pf fmt "x%d" (ordinal nm.temps (Loop_temp.to_int t))

(* An index temporary, [o] as in [Loop_js]. *)
let pp_index_temp nm fmt t =
  Fmt.pf fmt "o%d" (ordinal nm.index_temps (Loop_temp.to_int t))

let pp_array nm fmt a =
  Fmt.pf fmt "a%d" (ordinal nm.arrays (Loop_array.to_int a))

(* [%.17g] round-trips every finite binary64; the special values print the way
   the OCaml runtime does, so a golden does not depend on the platform's [printf]
   for them. *)
let pp_float fmt x =
  if Float.is_nan x then Fmt.string fmt "nan"
  else if x = Float.infinity then Fmt.string fmt "inf"
  else if x = Float.neg_infinity then Fmt.string fmt "-inf"
  else if x = 0. && 1. /. x < 0. then Fmt.string fmt "-0."
  else Fmt.pf fmt "%.17g" x

let rec pp_index nm fmt : Loop_index.t -> unit = function
  | Loop_index.Add (a, b) ->
      Fmt.pf fmt "(%a + %a)" (pp_index nm) a (pp_index nm) b
  | Loop_index.Ceil_div_pos (a, d) ->
      Fmt.pf fmt "ceil_div(%a, %d)" (pp_index nm) a d
  | Loop_index.Clamp_low a -> Fmt.pf fmt "clamp_low(%a)" (pp_index nm) a
  | Loop_index.Const n -> Fmt.int fmt n
  | Loop_index.Floor_div_pos (a, d) ->
      Fmt.pf fmt "floor_div(%a, %d)" (pp_index nm) a d
  | Loop_index.Max (a, b) ->
      Fmt.pf fmt "max(%a, %a)" (pp_index nm) a (pp_index nm) b
  | Loop_index.Min (a, b) ->
      Fmt.pf fmt "min(%a, %a)" (pp_index nm) a (pp_index nm) b
  | Loop_index.Scale (k, a) -> Fmt.pf fmt "(%d * %a)" k (pp_index nm) a
  | Loop_index.Temp t -> pp_index_temp nm fmt t
  | Loop_index.Var v -> pp_var nm fmt v

let pp_coord nm fmt (c : Loop_index.coord) = Expr.Coord.pp (pp_index nm) fmt c

let rec pp_expr : type a. names -> Format.formatter -> a Loop_expr.t -> unit =
 fun nm fmt -> function
  | Loop_expr.Array_get (a, i) ->
      Fmt.pf fmt "%a[%a]" (pp_array nm) a (pp_index nm) i
  | Loop_expr.Binary (op, a, b) ->
      Fmt.pf fmt "(%a %s %a)" (pp_expr nm) a (Expr.Value.binary_sym op)
        (pp_expr nm) b
  | Loop_expr.Const x -> pp_float fmt x
  | Loop_expr.Float_max (a, b) ->
      Fmt.pf fmt "float_max(%a, %a)" (pp_expr nm) a (pp_expr nm) b
  | Loop_expr.Float_to_i64 a -> Fmt.pf fmt "float_to_i64(%a)" (pp_expr nm) a
  | Loop_expr.I64_binary (op, a, b) ->
      Fmt.pf fmt "(%a %s %a)" (pp_expr nm) a
        (Expr.Value.i64_binary_sym op)
        (pp_expr nm) b
  | Loop_expr.I64_const n -> Fmt.pf fmt "%LdL" n
  | Loop_expr.I64_of_index i -> Fmt.pf fmt "i64_of_index(%a)" (pp_index nm) i
  | Loop_expr.I64_to_float a -> Fmt.pf fmt "i64_to_float(%a)" (pp_expr nm) a
  | Loop_expr.Load (b, c) ->
      Fmt.pf fmt "load %a[%a]" Tensor_id.pp b.Loop_buffer.id (pp_coord nm) c
  | Loop_expr.Load_flat (b, i) ->
      Fmt.pf fmt "load %a[@%a]" Tensor_id.pp b.Loop_buffer.id (pp_index nm) i
  | Loop_expr.Load_i64 (b, c) ->
      Fmt.pf fmt "load_i64 %a[%a]" Tensor_id.pp b.Loop_buffer.id (pp_coord nm) c
  | Loop_expr.Load_i64_flat (b, i) ->
      Fmt.pf fmt "load_i64 %a[@%a]" Tensor_id.pp b.Loop_buffer.id (pp_index nm)
        i
  | Loop_expr.Round_f32 a -> Fmt.pf fmt "round_f32(%a)" (pp_expr nm) a
  | Loop_expr.Select (p, a, b) ->
      Fmt.pf fmt "(%a ? %a : %a)" (pp_pred nm) p (pp_expr nm) a (pp_expr nm) b
  | Loop_expr.Temp (_, t) -> pp_temp nm fmt t
  | Loop_expr.Unary (op, a) ->
      Fmt.pf fmt "%s(%a)" (Expr.Value.unary_name op) (pp_expr nm) a
  | Loop_expr.Value_of_index i ->
      Fmt.pf fmt "float_of_index(%a)" (pp_index nm) i

and pp_pred nm fmt : Loop_expr.pred -> unit = function
  | Loop_bool.I64_eq (a, b) ->
      Fmt.pf fmt "%a == %a" (pp_expr nm) a (pp_expr nm) b
  | Loop_bool.I64_lt (a, b) ->
      Fmt.pf fmt "%a < %a" (pp_expr nm) a (pp_expr nm) b
  | Loop_bool.Index_eq (a, b) ->
      Fmt.pf fmt "%a == %a" (pp_index nm) a (pp_index nm) b
  | Loop_bool.Index_lt (a, b) ->
      Fmt.pf fmt "%a < %a" (pp_index nm) a (pp_index nm) b
  | Loop_bool.Index_overflows i ->
      Fmt.pf fmt "index_overflows(%a)" (pp_index nm) i
  | Loop_bool.Not p -> Fmt.pf fmt "!(%a)" (pp_pred nm) p
  | Loop_bool.Or (p, q) -> Fmt.pf fmt "(%a || %a)" (pp_pred nm) p (pp_pred nm) q
  | Loop_bool.Out_of_range (i, n) ->
      Fmt.pf fmt "out_of_range(%a, %d)" (pp_index nm) i n
  | Loop_bool.Pool_better (a, b) ->
      Fmt.pf fmt "pool_better(%a, %a)" (pp_expr nm) a (pp_expr nm) b
  | Loop_bool.Value_eq (a, b) ->
      Fmt.pf fmt "%a == %a" (pp_expr nm) a (pp_expr nm) b
  | Loop_bool.Value_lt (a, b) ->
      Fmt.pf fmt "%a < %a" (pp_expr nm) a (pp_expr nm) b

let pp_failure nm fmt : Loop_failure.t -> unit = function
  | Loop_failure.Gather_out_of_range { raw; extent } ->
      Fmt.pf fmt "gather_out_of_range(%a, %d)" (pp_expr nm) raw extent
  | Loop_failure.I64_division_by_zero -> Fmt.string fmt "i64_division_by_zero"
  | Loop_failure.I64_division_overflow -> Fmt.string fmt "i64_division_overflow"
  | Loop_failure.I64_from_float { value } ->
      Fmt.pf fmt "i64_from_float(%a)" (pp_expr nm) value
  | Loop_failure.Index_overflow { index } ->
      Fmt.pf fmt "index_overflow(%a)" (pp_index nm) index
  | Loop_failure.Load_out_of_range { buffer; coord } ->
      Fmt.pf fmt "load_out_of_range(%a[%a])" Tensor_id.pp buffer.Loop_buffer.id
        (pp_coord nm) coord
  | Loop_failure.Scan_lane_out_of_range { local; row; lane; extent } ->
      Fmt.pf fmt "scan_lane_out_of_range(%a, row=%a, lane=%a, extent=%d)"
        (Fmt.option ~none:(Fmt.any "inline") Expr.Local_var.pp)
        local (pp_index nm) row (pp_index nm) lane extent
  | Loop_failure.Scan_row_out_of_range { local; row; lane; extent } ->
      Fmt.pf fmt "scan_row_out_of_range(%a, row=%a, lane=%a, extent=%d)"
        (Fmt.option ~none:(Fmt.any "inline") Expr.Local_var.pp)
        local (pp_index nm) row (pp_index nm) lane extent
  | Loop_failure.Local_out_of_range { local; index; extent } ->
      Fmt.pf fmt "local_out_of_range(%a, %a, %d)" Expr.Local_var.pp local
        (pp_index nm) index extent

let pp_stored nm fmt : Loop_stored.t -> unit = function
  | Loop_stored.Bool e -> Fmt.pf fmt "bool(%a)" (pp_expr nm) e
  | Loop_stored.F32 e -> Fmt.pf fmt "f32(%a)" (pp_expr nm) e
  | Loop_stored.I64 e -> Fmt.pf fmt "i64(%a)" (pp_expr nm) e

let rec pp_stmt nm fmt : Loop_stmt.t -> unit = function
  | Loop_stmt.Alloc (a, n) ->
      Fmt.pf fmt "alloc %a : float64[%d]" (pp_array nm) a (n :> int)
  | Loop_stmt.Array_set (a, i, e) ->
      Fmt.pf fmt "%a[%a] = %a" (pp_array nm) a (pp_index nm) i (pp_expr nm) e
  | Loop_stmt.Assign (_, t, e) ->
      Fmt.pf fmt "%a = %a" (pp_temp nm) t (pp_expr nm) e
  | Loop_stmt.Assign_index (t, i) ->
      Fmt.pf fmt "%a = %a" (pp_index_temp nm) t (pp_index nm) i
  | Loop_stmt.Assign_index_of_i64 (t, e) ->
      Fmt.pf fmt "%a = index_of_i64(%a)" (pp_index_temp nm) t (pp_expr nm) e
  | Loop_stmt.Charge_scan_update -> Fmt.string fmt "charge_scan_update"
  | Loop_stmt.Release_scan_state w -> Fmt.pf fmt "release_scan_state %d" w
  | Loop_stmt.Reserve_scan_state w -> Fmt.pf fmt "reserve_scan_state %d" w
  | Loop_stmt.Reset_meter -> Fmt.string fmt "reset_meter"
  | Loop_stmt.Fail_if (p, f) ->
      Fmt.pf fmt "fail_if %a -> %a" (pp_pred nm) p (pp_failure nm) f
  | Loop_stmt.For { var; lo; hi; body } ->
      Fmt.pf fmt "@[<v 2>for %a in [%a, %a):%a@]" (pp_var nm) var (pp_index nm)
        lo (pp_index nm) hi (pp_block nm) body
  | Loop_stmt.If (p, yes, no) ->
      Fmt.pf fmt "@[<v 2>if %a:%a@]" (pp_pred nm) p (pp_block nm) yes;
      if no <> [] then Fmt.pf fmt "@,@[<v 2>else:%a@]" (pp_block nm) no
  | Loop_stmt.Mark m -> Fmt.pf fmt "mark %s" (Loop_mark.name m)
  | Loop_stmt.Store { buffer; coord; value } ->
      Fmt.pf fmt "store %a[%a] = %a" Tensor_id.pp buffer.Loop_buffer.id
        (pp_coord nm) coord (pp_stored nm) value
  | Loop_stmt.Store_flat { buffer; offset; value } ->
      Fmt.pf fmt "store %a[@%a] = %a" Tensor_id.pp buffer.Loop_buffer.id
        (pp_index nm) offset (pp_stored nm) value

and pp_block nm fmt body =
  List.iter (fun s -> Fmt.pf fmt "@,%a" (pp_stmt nm) s) body

let pp_buffer fmt (b : Loop_buffer.t) =
  Fmt.pf fmt "%s %a %s %a"
    (Loop_buffer.role_name b.role)
    Tensor_id.pp b.id
    (let (Payload.Fmt f) = b.sg.Tensor_sig.fmt in
     Payload.fmt_name f)
    Vec6.pp_shape b.sg.Tensor_sig.shape

let stmts fmt body =
  let nm = fresh_names () in
  Fmt.pf fmt "@[<v>%a@]" (Fmt.list ~sep:Fmt.cut (pp_stmt nm)) body

let program fmt (p : Loop_program.t) =
  let nm = fresh_names () in
  Fmt.pf fmt "@[<v>";
  List.iter (fun b -> Fmt.pf fmt "%a@," pp_buffer b) p.buffers;
  Fmt.pf fmt "%a@]" (Fmt.list ~sep:Fmt.cut (pp_stmt nm)) p.body
