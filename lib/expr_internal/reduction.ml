type kind = Expr_repr.reduction_kind = Max | Sum

type t = Expr_repr.reduction = {
  kind : kind;
  var : Reduce_var.t;
  lo : Role.Position.t Index.t;
  hi : Role.Delta.t Index.t;
  body : Expr_repr.value;
}

let kind_name = function Max -> "max_reduce" | Sum -> "sum"
