(** Binary PPM (P6, maxval 255): the one raw image format a bounded example can
    read without a codec. *)

type t = { width : int; height : int; rgb : Bytes.t }
(** [rgb] holds [3 * width * height] bytes, row-major, interleaved. *)

val of_string : string -> (t, string) result
