(* Tensor storage <-> the bytes of a file, by mapping the file with the very
   Bigarray kind of the payload. Host byte order; the C backend admits
   little-endian hosts only. *)

let map_region : type e b q.
    bool ->
    Unix.file_descr ->
    int ->
    (e, b, q) Payload.fmt ->
    int ->
    (e, b, Bigarray.c_layout) Bigarray.Array1.t =
 fun shared fd pos fmt n ->
  let m kind =
    Bigarray.array1_of_genarray
      (Unix.map_file fd ~pos:(Int64.of_int pos) kind Bigarray.c_layout shared
         [| n |])
  in
  match fmt with
  | Payload.BF16 -> m Bigarray.int16_unsigned
  | Payload.Bool -> m Bigarray.int8_unsigned
  | Payload.F16 -> m Bigarray.int16_unsigned
  | Payload.F32 -> m Bigarray.float32
  | Payload.F64 -> m Bigarray.float64
  | Payload.I16 -> m Bigarray.int16_signed
  | Payload.I32 -> m Bigarray.int32
  | Payload.I64 -> m Bigarray.int64
  | Payload.I8 -> m Bigarray.int8_signed

(* Copy a tensor's storage into the blob ([`In]) or the blob's back into the
   tensor ([`Out]). *)
let copy dir fd pos (Tensor.Tensor t) =
  let data = t.Tensor.payload.Payload.data in
  let n = Bigarray.Array1.dim data in
  if n > 0 then
    let region = map_region (dir = `In) fd pos t.Tensor.payload.Payload.fmt n in
    match dir with
    | `In -> Bigarray.Array1.blit data region
    | `Out -> Bigarray.Array1.blit region data
