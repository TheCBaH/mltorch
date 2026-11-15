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

(* A byte offset past [max_int] cannot index a bigarray on this backend, and
   narrowing it silently would wrap: refuse it as the out-of-range read it is. *)
let narrow off =
  let v = Core.Storage_units.Byte_offset.to_int64 off in
  if Int64.compare v (Int64.of_int max_int) > 0 then
    invalid_arg "Pt2_storage: byte offset out of range"
  else Int64.to_int v

let length b =
  let length_bytes b =
    Core.Storage_units.Byte_size.of_int64 (Int64.of_int (Bigarray.Array1.dim b))
  in
  match length_bytes b with Ok s -> s | Error _ -> assert false

(* A cell is an element index; [cell * width] must not wrap, so the cell is
   bounded by the quotient before it is scaled. *)
let scaled cell width =
  if cell < 0 || cell > max_int / width then
    invalid_arg "Pt2_storage: cell out of range"
  else cell * width

let cell_count b element_bytes =
  Bigarray.Array1.dim b
  / Int64.to_int (Core.Storage_units.Element_bytes.to_int64 element_bytes)

let to_string b =
  String.init (Bigarray.Array1.dim b) (fun i -> Bigarray.Array1.unsafe_get b i)

let cell_uint8 b ~cell = get_uint8 b cell
let cell_int32_le b ~cell = get_int32_le b (scaled cell 4)
let cell_int64_le b ~cell = get_int64_le b (scaled cell 8)
let get_uint8 b off = get_uint8 b (narrow off)
let get_int32_le b off = get_int32_le b (narrow off)
let get_int64_le b off = get_int64_le b (narrow off)
