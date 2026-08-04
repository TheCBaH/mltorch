module Make () = struct
  type t = Symbolic_expr.t
  type 'role index = Symbolic_expr.index_expr
  type input = Tensor_sig.t

  let const x = Symbolic_expr.Const x
  let add a b = Symbolic_expr.Binary (Add, a, b)
  let sub a b = Symbolic_expr.Binary (Sub, a, b)
  let mul a b = Symbolic_expr.Binary (Mul, a, b)
  let div a b = Symbolic_expr.Binary (Div, a, b)
  let exp a = Symbolic_expr.Unary (Exp, a)
  let sqrt a = Symbolic_expr.Unary (Sqrt, a)

  type b = Symbolic_expr.bool_expr

  let lt a b = Symbolic_expr.Cmp (Lt, a, b)
  let select c a b = Symbolic_expr.Select (c, a, b)
  let index_zero = Symbolic_expr.Index_const 0
  let index_extent (e : Dim.extent Dim.t) = Symbolic_expr.Index_const (e :> int)
  let index_const n = Symbolic_expr.Index_const n
  let of_index i = i
  let index_add a b = Symbolic_expr.Index_add (a, b)
  let index_scale k a = Symbolic_expr.Index_scale (k, a)

  let index_floor_div_pos a d =
    if (d : Op_config.Pos.t :> int) = 1 then a
    else Symbolic_expr.Index_floor_div_pos (a, (d :> int))

  let index_ceil_div_pos a d =
    if (d : Op_config.Pos.t :> int) = 1 then a
    else Symbolic_expr.Index_ceil_div_pos (a, (d :> int))

  let index_min a b = Symbolic_expr.Index_min (a, b)
  let index_eq a b = Symbolic_expr.Index_eq (a, b)
  let clamp_low x = Symbolic_expr.Index_max (Symbolic_expr.Index_const 0, x)
  let assume_index x = x
  let value_of_index x = Symbolic_expr.Value_of_index x
  let load s (v : Symbolic_expr.index_expr Vec6.t) = Symbolic_expr.Load (s, v)

  let load6 s ~n ~t ~d ~h ~w ~c =
    Symbolic_expr.Load (s, Vec6.make ~n ~t ~d ~h ~w ~c)

  let max_pool2d input ~x_shape:_ ~kernel ~stride ~pad out =
    Symbolic_expr.Max_pool { input; kernel; stride; pad; out; result = Value }

  let max_pool2d_index input ~x_shape:_ ~kernel ~stride ~pad out =
    Symbolic_expr.Max_pool { input; kernel; stride; pad; out; result = Index }

  let c = ref 0

  let sum ~lo ~hi f =
    incr c;
    let v = !c in
    Symbolic_expr.Reduce
      { kind = Sum; var = v; lo; hi; body = f (Symbolic_expr.Reduce_var v) }

  let max_reduce ~lo ~hi f =
    incr c;
    let v = !c in
    Symbolic_expr.Reduce
      {
        kind = Max_reduce;
        var = v;
        lo;
        hi;
        body = f (Symbolic_expr.Reduce_var v);
      }
end

let out_vec : Symbolic_expr.index_expr Vec6.t =
  Vec6.make ~n:(Symbolic_expr.Index_var Axis.N)
    ~t:(Symbolic_expr.Index_var Axis.T) ~d:(Symbolic_expr.Index_var Axis.D)
    ~h:(Symbolic_expr.Index_var Axis.H) ~w:(Symbolic_expr.Index_var Axis.W)
    ~c:(Symbolic_expr.Index_var Axis.C)
