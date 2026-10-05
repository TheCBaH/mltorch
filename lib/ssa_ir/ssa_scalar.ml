(* The meaning of the pure, total scalar operations, defined once. The
   interpreter evaluates through it and the constant folder through the same
   function, so a fold can only produce what execution would. Operations that
   fail, read memory, touch the meter or depend on a proof are not here. *)

type t = F of float | I of int64 | P of bool

let floor_div n d =
  let q = Int64.div n d and r = Int64.rem n d in
  if Int64.compare r 0L < 0 then Int64.pred q else q

let ceil_div n d = Int64.neg (floor_div (Int64.neg n) d)

let of_const : Ssa_const.t -> t = function
  | Ssa_const.F32 x | Ssa_const.F64 x -> F x
  | Ssa_const.I64 x | Ssa_const.Index x -> I x
  | Ssa_const.Pred b -> P b

let to_const (ty : Ssa_type.t) v : Ssa_const.t option =
  match (ty, v) with
  | Ssa_type.Scalar Ssa_type.F32, F x -> Some (Ssa_const.F32 x)
  | Ssa_type.Scalar Ssa_type.F64, F x -> Some (Ssa_const.F64 x)
  | Ssa_type.Scalar Ssa_type.I64, I x -> Some (Ssa_const.I64 x)
  | Ssa_type.Scalar Ssa_type.Index, I x -> Some (Ssa_const.Index x)
  | Ssa_type.Scalar Ssa_type.Pred, P b -> Some (Ssa_const.Pred b)
  | _ -> None

let is_f32 (ty : Ssa_type.t) = Ssa_type.equal ty (Ssa_type.Scalar Ssa_type.F32)

(* [get] reads an operand's value; [result] is the type of the operation's
   result. [None] for an operation that is not pure and total. *)
let eval (op : Ssa_op.t) ~(result : Ssa_type.t) ~(get : Ssa_value.t -> t) :
    t option =
  let float a =
    match get a with F x -> x | I _ | P _ -> invalid_arg "Ssa_scalar: float"
  in
  let int a =
    match get a with I x -> x | F _ | P _ -> invalid_arg "Ssa_scalar: int"
  in
  let bool a =
    match get a with P b -> b | F _ | I _ -> invalid_arg "Ssa_scalar: pred"
  in
  match op with
  | Ssa_op.Const c -> Some (of_const c)
  | Ssa_op.Convert (c, a) ->
      Some
        (match c with
        | Ssa_op.Convert.F32_to_f64 -> F (float a)
        | Ssa_op.Convert.F64_to_f32 -> F (Ssa_const.round_f32 (float a))
        | Ssa_op.Convert.I64_to_f32 -> F (Ssa_const.round32_of_i64 (int a))
        | Ssa_op.Convert.I64_to_f64 -> F (Int64.to_float (int a))
        | Ssa_op.Convert.Index_to_f64 -> F (Int64.to_float (int a))
        | Ssa_op.Convert.Index_to_i64 -> I (int a))
  | Ssa_op.Float_binary (op, a, b) ->
      let x = float a in
      let y = float b in
      let r = Expr.Value.apply_binary op x y in
      Some (F (if is_f32 result then Ssa_const.round_f32 r else r))
  | Ssa_op.Float_compare (c, a, b) ->
      let x = float a in
      let y = float b in
      Some
        (P
           (match c with
           | Ssa_op.Compare.Eq -> x = y
           | Ssa_op.Compare.Lt -> x < y))
  | Ssa_op.Float_max (a, b) ->
      let x = float a in
      let y = float b in
      Some (F (Expr.Max_op.apply Expr.Max_op.Float_max x y))
  | Ssa_op.Float_fma (a, b, c) ->
      let x = float a in
      let y = float b in
      let z = float c in
      Some
        (F (if is_f32 result then Ssa_numerics.fma32 x y z else Float.fma x y z))
  | Ssa_op.Float_unary (op, a) ->
      let x = float a in
      Some
        (F
           (if not (is_f32 result) then Expr.Value.apply_unary op x
            else
              match op with
              | Expr.Value.Erf -> Ssa_numerics.erf32 x
              | Expr.Value.Cos | Expr.Value.Exp | Expr.Value.Log
              | Expr.Value.Sin | Expr.Value.Sqrt | Expr.Value.Trunc ->
                  Ssa_const.round_f32 (Expr.Value.apply_unary op x)))
  | Ssa_op.I64_arith (op, a, b) ->
      let x = int a in
      let y = int b in
      Some
        (I
           (match op with
           | Ssa_op.I64_op.Add -> Int64.add x y
           | Ssa_op.I64_op.Mul -> Int64.mul x y
           | Ssa_op.I64_op.Sub -> Int64.sub x y))
  | Ssa_op.I64_compare (c, a, b) | Ssa_op.Index_compare (c, a, b) ->
      let x = int a in
      let y = int b in
      Some
        (P
           (match c with
           | Ssa_op.Compare.Eq -> Int64.equal x y
           | Ssa_op.Compare.Lt -> Int64.compare x y < 0))
  | Ssa_op.Index_ceil_div (k, a) -> Some (I (ceil_div (int a) k))
  | Ssa_op.Index_clamp_low a -> Some (I (Stdlib.max 0L (int a)))
  | Ssa_op.Index_floor_div (k, a) -> Some (I (floor_div (int a) k))
  | Ssa_op.Index_max (a, b) ->
      let x = int a in
      let y = int b in
      Some (I (Stdlib.max x y))
  | Ssa_op.Index_min (a, b) ->
      let x = int a in
      let y = int b in
      Some (I (Stdlib.min x y))
  | Ssa_op.Pool_better (a, b) ->
      let best = float a in
      let value = float b in
      Some (P (Expr.Max_op.pool_better ~best ~value))
  | Ssa_op.Pred_not a -> Some (P (not (bool a)))
  | Ssa_op.Pred_or (a, b) ->
      let x = bool a in
      let y = bool b in
      Some (P (x || y))
  | Ssa_op.Select (p, a, b) -> Some (get (if bool p then a else b))
  | Ssa_op.Check_access _ | Ssa_op.Check_gather _ | Ssa_op.Check_local _
  | Ssa_op.Check_scan _ | Ssa_op.Float_to_i64 _ | Ssa_op.I64_div _
  | Ssa_op.Index_add _ | Ssa_op.Index_add_in_domain _ | Ssa_op.Index_of_i64 _
  | Ssa_op.Index_scale _ | Ssa_op.Index_scale_in_domain _ | Ssa_op.Load _
  | Ssa_op.Load_in_bounds _ | Ssa_op.Local_alloc _ | Ssa_op.Local_read _
  | Ssa_op.Lanewise _ | Ssa_op.Local_write _ | Ssa_op.Mark _
  | Ssa_op.Mark_lanes _ | Ssa_op.Meter_charge | Ssa_op.Meter_release _
  | Ssa_op.Meter_reserve _ | Ssa_op.Meter_reset | Ssa_op.Store _
  | Ssa_op.Vec_extract _ | Ssa_op.Vec_insert _ | Ssa_op.Vec_iota _
  | Ssa_op.Vec_load _ | Ssa_op.Vec_splat _ | Ssa_op.Vec_store _ ->
      None
