(* [Expr.Bool.t]'s predicates, parametrized by the float and int64 expression
   types so this module can sit below [Loop_expr], whose [Select] uses it. *)
type ('f, 'i) t =
  | I64_eq of 'i * 'i
  | I64_lt of 'i * 'i
  | Index_eq of Loop_index.t * Loop_index.t
  | Index_lt of Loop_index.t * Loop_index.t
  | Index_overflows of Loop_index.t
      (** some [Add] or [Scale] inside the index leaves the index domain
          ([Loop_range.domain]) for the values its variables now hold. *)
  | Not of ('f, 'i) t
  | Or of ('f, 'i) t * ('f, 'i) t
  | Out_of_range of Loop_index.t * int
      (** [index < 0 || index >= extent]: what a bounds check asks. *)
  | Pool_better of 'f * 'f
      (** [Max_op.pool_better best x]: the one predicate a paired argmax state
          advances under. *)
  | Value_eq of 'f * 'f
  | Value_lt of 'f * 'f
