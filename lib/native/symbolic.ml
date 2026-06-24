module Make () = struct
  type t = Expr.t
  type 'role ix = Expr.iexpr
  type input = Tensor_sig.t

  let const x = Expr.Const x
  let add a b = Expr.Bin (Add, a, b)
  let sub a b = Expr.Bin (Sub, a, b)
  let mul a b = Expr.Bin (Mul, a, b)
  let div a b = Expr.Bin (Div, a, b)
  let exp a = Expr.Un (Exp, a)
  let sqrt a = Expr.Un (Sqrt, a)

  type b = Expr.bexpr

  let lt a b = Expr.Cmp (Lt, a, b)
  let select c a b = Expr.Select (c, a, b)
  let izero = Expr.Iconst 0
  let iext (e : Dim.extent Dim.t) = Expr.Iconst (e :> int)
  let iconst n = Expr.Iconst n
  let of_index i = i
  let iadd a b = Expr.Iadd (a, b)
  let iscale k a = Expr.Iscale (k, a)
  let imin a b = Expr.Imin (a, b)
  let clamp_low x = Expr.Imax (Expr.Iconst 0, x)
  let assume_index x = x

  let load (s : input) (idx : Axis.t -> _ ix) : t =
    Expr.Load (s, [| idx N; idx T; idx D; idx H; idx W; idx C |])

  let c = ref 0

  let sum ~lo ~hi f =
    incr c;
    let v = !c in
    Expr.Reduce { kind = Sum; var = v; lo; hi; body = f (Expr.Rvar v) }

  let maxr ~lo ~hi f =
    incr c;
    let v = !c in
    Expr.Reduce { kind = Max_r; var = v; lo; hi; body = f (Expr.Rvar v) }
end

let out_coord a = Expr.Ivar a
