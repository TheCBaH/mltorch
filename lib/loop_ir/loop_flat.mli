(** A coordinate access as one dense row-major offset, the same linearisation
    the tensors use, folded as it is built. Shared by every backend and by the
    vectorizer, so they agree on what an access's offset is. *)

val offset : Loop_buffer.t -> Loop_index.coord -> Loop_index.t
