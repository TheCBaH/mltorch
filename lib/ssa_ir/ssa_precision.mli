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
    - A square root and a truncation are exact in binary32; [exp], [log], [sin],
      [cos] and [erf] are the sequences of {!F32_math}.

    A program that already holds vectors is refused: precision is chosen first,
    then the planner vectorizes at that precision. *)

val to_f32 : ?exact:bool -> Ssa_program.t -> Ssa_program.t
(** The program must verify and hold no vector operation, and
    {!Ssa_numerics.admit} must accept it; [Invalid_argument] otherwise.

    [exact] (default [false]) reads a binary32 cell with
    {!Ssa_op.Decode.F32_exact} and stores a binary32 value with
    {!Ssa_op.Encode.F32_exact}, so a kernel has no widening and narrowing around
    its accesses, scalar or vectorized afterwards: the [Fp32_ordered] form.
    Without it a read widens and narrows, the [Simd_fp32_*] form. *)
