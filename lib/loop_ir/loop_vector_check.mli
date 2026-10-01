(** The verifier of a vector program: shape, types and the legality facts a
    vector loop needs, checked from the program and never from emitted text. A
    program that fails it is never given to a backend.

    What it proves per vector loop: lanes at least two; constant bounds; every
    temporary assigned before it is read; every access's stride equal to the
    loop variable's coefficient in its offset; a splat independent of the loop
    variable; no store through a broadcast; no buffer both stored and loaded
    unless at the identical access. That the vector body computes what the
    scalar loop computes is the oracle's claim ({!Loop_vector_expand}), checked
    by running both. *)

module Reason : sig
  type t =
    | Bad_lanes of int
    | Index_value_step_mismatch of { step : int; coefficient : int }
    | Non_constant_bounds
    | Offset_not_affine
    | Splat_depends_on_loop_variable
    | Splat_loads_stored_buffer of Loop_buffer.t
    | Store_through_broadcast
    | Store_loaded_elsewhere of Loop_buffer.t
    | Stores_overlap of Loop_buffer.t
    | Stride_mismatch of { stride : int; coefficient : int }
    | Temp_read_before_assigned of Loop_vector.Temp.t
    | Temp_assigned_twice of Loop_vector.Temp.t
    | Unsupported_load_format of Loop_buffer.t
end

type error = [ `Vector_invalid of Reason.t ]

val pp_error : Format.formatter -> [< error ] -> unit
val program : Loop_vector.program -> (unit, error) Err.t
