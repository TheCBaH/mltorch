type t = Expr_repr.bool_expr =
  | I64_eq of int64 Expr_repr.value * int64 Expr_repr.value
  | I64_lt of int64 Expr_repr.value * int64 Expr_repr.value
  | Index_eq of Role.Delta.t Index.t * Role.Delta.t Index.t
  | Value_lt of float Expr_repr.value * float Expr_repr.value

let value_lt a b = Value_lt (a, b)
let index_eq a b = Index_eq (a, b)
let i64_eq a b = I64_eq (a, b)
let i64_lt a b = I64_lt (a, b)
