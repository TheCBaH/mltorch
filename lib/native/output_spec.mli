(* The storage boundary of a value the float path computes: which declared
   signatures it can land in, and how it lands. One table, so the Kernel's
   admission check, [Kernel_eval] and [Stage_program.ground] cannot disagree
   about it. Int64 values are not here: they are exact end to end and never pass
   through the float path (see [Kernel.Value_i64]). *)

val storable : Tensor_sig.t -> bool
(** [true] for an unquantized F32 or Bool signature -- the only formats a
    float-path value can be stored as. *)

val store : Tensor_sig.t -> Tensor.packed -> Tensor.packed
(** The tensor the float path produced, as its declared storage: a Bool
    signature is written as canonical Bool bytes, every other one stays the F32
    tensor it already is. *)
