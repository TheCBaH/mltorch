module Make () = struct
  type t = Expr.t
  type 'role index = Expr.index_expr
  type input = Tensor_sig.t

  let const x = Expr.Const x
  let add a b = Expr.Binary (Add, a, b)
  let sub a b = Expr.Binary (Sub, a, b)
  let mul a b = Expr.Binary (Mul, a, b)
  let div a b = Expr.Binary (Div, a, b)
  let exp a = Expr.Unary (Exp, a)
  let sqrt a = Expr.Unary (Sqrt, a)

  type b = Expr.bool_expr

  let lt a b = Expr.Cmp (Lt, a, b)
  let select c a b = Expr.Select (c, a, b)
  let index_zero = Expr.Index_const 0
  let index_extent (e : Dim.extent Dim.t) = Expr.Index_const (e :> int)
  let index_const n = Expr.Index_const n
  let of_index i = i
  let index_add a b = Expr.Index_add (a, b)
  let index_scale k a = Expr.Index_scale (k, a)
  let index_min a b = Expr.Index_min (a, b)
  let clamp_low x = Expr.Index_max (Expr.Index_const 0, x)
  let assume_index x = x

  let load s idx =
    Expr.Load
      ( s,
        [|
          idx Axis.N; idx Axis.T; idx Axis.D; idx Axis.H; idx Axis.W; idx Axis.C;
        |] )

  let c = ref 0

  let sum ~lo ~hi f =
    incr c;
    let v = !c in
    Expr.Reduce { kind = Sum; var = v; lo; hi; body = f (Expr.Reduce_var v) }

  let max_reduce ~lo ~hi f =
    incr c;
    let v = !c in
    Expr.Reduce
      { kind = Max_reduce; var = v; lo; hi; body = f (Expr.Reduce_var v) }
end

let out_coord a = Expr.Index_var a
