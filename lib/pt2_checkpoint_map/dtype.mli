(** The element types a checkpoint map names, in safetensors' codes. Closed and
    ordered alphabetically; a [BF16] or [F16] tensor exists only as a stored
    checkpoint value to be widened -- the graph-side {!Pt2_dtype} has neither.
*)

type t = BF16 | BOOL | F16 | F32 | F64 | I16 | I32 | I64 | I8 | U8

val all : t list
val equal : t -> t -> bool
val of_code : string -> t option
val to_code : t -> string

val of_torch_name : string -> t option
(** The names [captures.json] uses: ["float32"], ["int64"], ["bool"], ... *)

val of_pt2 : Pt2_dtype.t -> t

val byte_width : t -> int
(** Bytes per element. *)

val pp : t Fmt.t
