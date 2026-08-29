type t = Expr_repr.bool_expr =
  | Value_lt of Expr_repr.value * Expr_repr.value
  | Index_eq of Role.Delta.t Index.t * Role.Delta.t Index.t

let value_lt a b = Value_lt (a, b)
let index_eq a b = Index_eq (a, b)
