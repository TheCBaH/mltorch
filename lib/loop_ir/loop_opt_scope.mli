(** The scope a pass over an already-lowered program rebuilds: what the
    lowering's own [Loop_range.Env.t] held while it built the program. *)

val var_range : Loop_range.Env.t -> Loop_index.t -> Loop_index.t -> Loop_range.t
(** The values a loop variable over [\[lo, hi)] can take, as an enclosure; an
    empty loop gives the point [lo]. *)

val single_assignment : Loop_stmt.t list -> Loop_temp.t -> bool
(** Whether exactly one statement assigns the index temporary, so its range at
    that assignment covers every read. *)
