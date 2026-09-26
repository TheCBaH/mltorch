open Js_ast

type num
type idx
type big
type bool_
type bits16
type bits32
type 'k arr
type 'k t = Js_ast.expr

let expr e = e
let id = Ident.v
let member o name = Member (o, id name)
let math name args = Call (member (Global Global.Math) name, args)

(* Every host-[int] to [Number] conversion goes through [float_of_int], exact
   for any value a 32-bit js_of_ocaml [int] can hold. *)
let lit n = Number (float_of_int n)

module Num = struct
  let abs a = math "abs" [ a ]
  let add a b = Binary (Add, a, b)
  let const x = Number x
  let cos a = math "cos" [ a ]
  let div a b = Binary (Div, a, b)
  let eq a b = Binary (Eq_strict, a, b)
  let exp a = math "exp" [ a ]
  let fround a = math "fround" [ a ]
  let is_finite a = Call (member (Global Global.Number) "isFinite", [ a ])
  let is_nan a = Call (member (Global Global.Number) "isNaN", [ a ])
  let log a = math "log" [ a ]
  let le a b = Binary (Le, a, b)
  let lt a b = Binary (Lt, a, b)
  let gt a b = Binary (Gt, a, b)
  let max a b = math "max" [ a; b ]
  let mul a b = Binary (Mul, a, b)
  let ne a b = Binary (Ne_strict, a, b)
  let neg a = Unary (Neg, a)
  let of_big b = Call (Global Global.Number, [ b ])
  let of_bits b = b

  (* An index [Number] can be -0 where an OCaml [int] cannot, and the
     interpreter's value is +0. [+ 0] maps -0 to +0 and leaves every other
     integer alone; this is the one place an index becomes a float. *)
  let of_idx = function
    | Number x as i when not (Float.sign_bit x) -> i
    | i -> add i (Number 0.)

  let pow a b = math "pow" [ a; b ]
  let sin a = math "sin" [ a ]
  let sqrt a = math "sqrt" [ a ]
  let sub a b = Binary (Sub, a, b)
  let trunc a = math "trunc" [ a ]
  let var v = Var v
end

module Idx = struct
  let is_zero = function Number x -> x = 0. | _ -> false

  (* [a + -k * b] prints as [a - k * b] and [a + -k] as [a - k]: every index is an integer within
     2^31, so both are exact in a [Number] and equal, [-0] included. *)
  let add a b =
    if is_zero a then b
    else if is_zero b then a
    else
      match b with
      | Number k when k < 0. -> Binary (Sub, a, Number (-.k))
      | Unary (Neg, c) -> Binary (Sub, a, c)
      | Binary (Mul, Number k, c) when k < 0. ->
          Binary (Sub, a, if k = -1. then c else Binary (Mul, Number (-.k), c))
      | _ -> Binary (Add, a, b)

  let ceil_div_pos a d = math "ceil" [ Binary (Div, a, lit d) ]
  let clamp_low a = math "max" [ lit 0; a ]
  let const n = lit n
  let eq a b = Binary (Eq_strict, a, b)
  let ge a b = Binary (Ge, a, b)
  let floor_div_pos a d = math "floor" [ Binary (Div, a, lit d) ]
  let lt a b = Binary (Lt, a, b)
  let max a b = math "max" [ a; b ]
  let min a b = math "min" [ a; b ]
  let of_big_bounded b = Call (Global Global.Number, [ b ])

  let out_of_range i n =
    Binary (Or, Binary (Lt, i, lit 0), Binary (Ge, i, lit n))

  (* 2^31 as a literal, exact in a [Number] and not an [int] a 32-bit backend
     could hold. *)
  let outside_int32 i =
    Binary
      ( Or,
        Binary (Lt, i, Number (-2147483648.)),
        Binary (Ge, i, Number 2147483648.) )

  let scale k a =
    if k = 1 then a
    else if is_zero a then a
    else if k = -1 then Unary (Neg, a)
    else Binary (Mul, lit k, a)

  let var v = Var v
end

module Big = struct
  let wrap e = Call (member (Global Global.Big_int) "asIntN", [ Number 64.; e ])
  let add_wrap a b = wrap (Binary (Add, a, b))
  let const n = Bigint n
  let div_unchecked a b = Binary (Div, a, b)
  let eq a b = Binary (Eq_strict, a, b)
  let lt a b = Binary (Lt, a, b)
  let mul_wrap a b = wrap (Binary (Mul, a, b))
  let of_idx i = Call (Global Global.Big_int, [ i ])
  let of_num_trunc x = Call (Global Global.Big_int, [ math "trunc" [ x ] ])
  let sub_wrap a b = wrap (Binary (Sub, a, b))
  let var v = Var v
end

module Bits = struct
  let and_ a b = Binary (Bit_and, a, b)
  let const n = lit n
  let eq a b = Binary (Eq_strict, a, b)
  let or_ a b = Binary (Bit_or, a, b)
  let shl16 a = Binary (Shl, a, Number 16.)
  let shr a n = Binary (Shr, a, lit n)
  let var v = Var v
end

module Pred = struct
  let false_ = Bool false
  let not_ p = Unary (Not, p)
  let or_ a b = Binary (Or, a, b)
end

let select p a b = Cond (p, a, b)

module Arr = struct
  let bits v = Var v
  let big v = Var v
  let idx v = Var v
  let literal es = Array es
  let new_float64 n = New (Global Global.Float64_array, [ lit n ])
  let new_uint32 n = New (Global Global.Uint32_array, [ lit n ])
  let num v = Var v
  let uint32 v = Var v
  let view_float32 a = New (Global Global.Float32_array, [ member a "buffer" ])
end

let load a i = Index (a, i)
let store a i v = Stmt.Assign (Lindex (a, i), Eq, v)

module Stmt = struct
  let assign v e = Stmt.Assign (Lvar v, Eq, e)
  let assign_big = assign
  let assign_idx = assign
  let assign_num = assign
  let const_arr v e = Stmt.Const (v, e)
  let const_bits v e = Stmt.Const (v, e)
  let const_idx v e = Stmt.Const (v, e)
  let const_num v e = Stmt.Const (v, e)
  let decr_num v e = Stmt.Assign (Lvar v, Minus_eq, e)

  let for_ v ~lo ~hi body =
    Stmt.For { var = v; init = lo; test = Binary (Lt, Var v, hi); body }

  let if_ p yes no = Stmt.If (p, yes, no)
  let incr_num v e = Stmt.Assign (Lvar v, Plus_eq, e)
  let let_big v e = Stmt.Let (v, e)
  let let_idx v e = Stmt.Let (v, e)
  let let_num v e = Stmt.Let (v, e)
  let return_ e = Stmt.Return (Some e)
  let return_null = Stmt.Return (Some Null)
end

let record fields = Object (List.map (fun (k, v) -> (id k, v)) fields)
let string s = String s
let bool b = Bool b
let to_string e = Call (member e "toString", [])

module Unsafe = struct
  let call f args = Call (Var f, args)
end
