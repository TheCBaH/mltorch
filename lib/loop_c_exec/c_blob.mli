(** Copies between a tensor's storage and a region of a file. *)

val copy : [ `In | `Out ] -> Unix.file_descr -> int -> Tensor.packed -> unit
(** [copy `In fd pos t] writes the tensor's cells into the file at byte [pos]
    (the file must already be long enough); [`Out] reads them back into the
    tensor. An empty tensor is a no-op: a zero-length map is never made. *)
