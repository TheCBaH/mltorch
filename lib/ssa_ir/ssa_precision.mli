(** Working precision made explicit. {!to_f32} rewrites a scalar program so that
    every binary64 working value is a binary32 value: types and constants
    change, nothing is implied. The result is what a binary32 kernel computes,
    bit for bit, when the interpreter evaluates it, because every binary32
    operation of the interpreter rounds once.

    - A float read decodes as before and is narrowed once ([f32], [f16], [bf16]
      and [bool] decode to exactly representable values; [f64], [i32] and [i64]
      round once), which {!Ssa_numerics.admit} states.
    - A binary64 constant is rounded once. A conversion from int64 or from an
      index is a single rounding to binary32, never through binary64.
    - A checked float-to-int64 conversion and a store widen their operand first,
      which is exact. A scratch cell is widened on write and narrowed on read.
    - A square root, truncation, [exp], [log], [sin], [cos] is the binary64
      function of the widened argument rounded once; [erf] is
      {!Ssa_numerics.erf32}.

    A program that already holds vectors is refused: precision is chosen first,
    then the planner vectorizes at that precision. *)

val to_f32 : Ssa_program.t -> Ssa_program.t
(** The program must verify and hold no vector operation, and
    {!Ssa_numerics.admit} must accept it; [Invalid_argument] otherwise. *)
