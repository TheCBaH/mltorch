(** Facts about indices, expressions and statements shared by the vectorizer and
    the verifier. *)

val mentions : Loop_var.t -> Loop_index.t -> bool
(** Whether the index reads the loop variable. *)

val coefficient : Loop_var.t -> Loop_index.t -> (int, [> `Not_affine ]) result
(** The coefficient of the loop variable in an offset, provided the offset is an
    exact linear form ([Loop_linear]) whose only dependence on the variable is
    through [Var v] itself. *)

val format_ok : Loop_buffer.t -> bool
(** The buffer formats a vector load decodes: bool, f32, f64 and i32. *)

val assigned_temps : Loop_temp.Set.t -> Loop_stmt.t -> Loop_temp.Set.t
(** Adds every temporary the statement (nested statements included) assigns. *)

val expr_depends : Loop_var.t -> Loop_temp.Set.t -> 'a Loop_expr.t -> bool
(** Whether a scalar expression reads the loop variable or one of the
    temporaries. *)

val pred_depends : Loop_var.t -> Loop_temp.Set.t -> Loop_expr.pred -> bool

val expr_buffers : 'a Loop_expr.t -> Loop_buffer.t list
(** The buffers a scalar expression loads from. *)

val expr_temps : 'a Loop_expr.t -> Loop_temp.t list
(** Every temporary an expression reads, with multiplicity. *)

val stmt_temp_reads : Loop_temp.t list -> Loop_stmt.t -> Loop_temp.t list
(** Adds the temporaries a statement (nested statements included) reads. *)

val has_loop : Loop_stmt.t -> bool
(** Whether the statement is, or contains, a [For]. *)
