type t = (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

(* The compiler's own bigstring primitives: bounds-checked, and implemented by
   js_of_ocaml ([caml_ba_uint8_get32]/[64]) as well as natively. They read in
   host byte order, so [Sys.big_endian] fixes it up. *)
external get32 : t -> int -> int32 = "%caml_bigstring_get32"
external get64 : t -> int -> int64 = "%caml_bigstring_get64"
external swap32 : int32 -> int32 = "%bswap_int32"
external swap64 : int64 -> int64 = "%bswap_int64"
external set64u : t -> int -> int64 -> unit = "%caml_bigstring_set64u"

let empty = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 0
let length = Bigarray.Array1.dim

let of_string s =
  let n = String.length s in
  let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  (* Eight bytes at a time: a model is up to a gigabyte, and a byte loop would
     cost seconds per load. [String.get_int64_ne] and [set64u] agree on host
     order, so the bytes round-trip. *)
  let words = n / 8 in
  for i = 0 to words - 1 do
    set64u b (i * 8) (String.get_int64_ne s (i * 8))
  done;
  for i = words * 8 to n - 1 do
    Bigarray.Array1.unsafe_set b i (String.unsafe_get s i)
  done;
  b

let get_uint8 b i = Char.code (Bigarray.Array1.get b i)

let get_int32_le b i =
  let v = get32 b i in
  if Sys.big_endian then swap32 v else v

let get_int64_le b i =
  let v = get64 b i in
  if Sys.big_endian then swap64 v else v
