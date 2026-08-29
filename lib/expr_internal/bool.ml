type t = Expr_repr.bool_expr =
  | Index_eq of Role.Delta.t Index.t * Role.Delta.t Index.t
  | Value_lt of Expr_repr.value * Expr_repr.value

let value_lt a b = Value_lt (a, b)
let index_eq a b = Index_eq (a, b)
