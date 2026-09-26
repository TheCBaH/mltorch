(* The value carriers an assignment can hold. Index temporaries are a third
   domain with their own [Loop_stmt.Assign_index], since an index is never a
   working float or an exact int64. *)
type _ t = Float : float t | Int64 : int64 t
