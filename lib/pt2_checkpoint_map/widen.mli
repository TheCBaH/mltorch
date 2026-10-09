(** Exact widening of the two 16-bit float formats to binary32, by integer bit
    manipulation only: an OCaml [float] neither preserves NaN payloads portably
    nor survives js_of_ocaml unchanged, and a widening has no rounding to do.

    - [BF16] is the high half of a binary32, so every value, signed zero,
      subnormal, infinity and NaN (payload and quiet bit included) is the 16
      bits followed by 16 zero bits.
    - [F16] maps zero, subnormals (renormalized), normals and infinity by their
      fields. A NaN keeps its sign and payload and becomes quiet, as the
      hardware conversions torch uses do. *)

val f16_to_f32_halves : int -> int * int
(** The binary32 pattern of a binary16 pattern (0 to 65535), as its high and low
    16 bits. Both are below 2^16, so nothing here depends on the width of [int].
*)

val bf16_to_f32_halves : int -> int * int

val widen : Dtype.t -> src:Pt2_storage.t -> dst:Pt2_storage.t -> unit
(** [widen from ~src ~dst] converts each little-endian 16-bit element of [src]
    to a little-endian binary32 in [dst], whose length must be twice [src]'s.
    Raises [Invalid_argument] for any [from] other than [BF16] or [F16] or a
    wrong destination length. *)
