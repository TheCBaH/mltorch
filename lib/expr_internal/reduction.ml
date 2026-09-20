type kind = Expr_repr.reduction_kind = Argmax_index | Argmax_value | Max | Sum

type t = Expr_repr.reduction = {
  kind : kind;
  var : Reduce_var.t;
  lo : Role.Position.t Index.t;
  hi : Role.Delta.t Index.t;
  body : float Expr_repr.value;
}

(* The int64 carrier's reduction: sum, max or argmax over an int64 body. *)
type i64 = Expr_repr.i64_reduction = {
  i64_kind : kind;
  i64_var : Reduce_var.t;
  i64_lo : Role.Position.t Index.t;
  i64_hi : Role.Delta.t Index.t;
  i64_body : int64 Expr_repr.value;
}

let kind_name = function
  | Argmax_index -> "argmax_index"
  | Argmax_value -> "argmax_value"
  | Max -> "max_reduce"
  | Sum -> "sum"
