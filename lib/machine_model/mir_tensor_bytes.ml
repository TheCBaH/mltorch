(* A tensor's cells as the little-endian bytes of its storage format, in
   [Vec6.offset] order — the row-major cell order an SSA buffer of the same
   signature addresses — and back. *)

let to_string (Tensor.Tensor t) =
  let p = t.Tensor.payload in
  let data = p.Payload.data in
  let n = Bigarray.Array1.dim data in
  let size = Payload.cell_bytes p.Payload.fmt in
  let b = Bytes.make (n * size) '\000' in
  let ints set v =
    for i = 0 to n - 1 do
      set b (i * size) (v i)
    done
  in
  (match p.Payload.fmt with
  | Payload.BF16 -> ints Bytes.set_uint16_le (fun i -> data.{i})
  | Payload.Bool -> ints Bytes.set_uint8 (fun i -> data.{i})
  | Payload.F16 -> ints Bytes.set_uint16_le (fun i -> data.{i})
  | Payload.F32 ->
      ints Bytes.set_int32_le (fun i -> Int32.bits_of_float data.{i})
  | Payload.F64 ->
      ints Bytes.set_int64_le (fun i -> Int64.bits_of_float data.{i})
  | Payload.I16 -> ints Bytes.set_int16_le (fun i -> data.{i})
  | Payload.I32 -> ints Bytes.set_int32_le (fun i -> data.{i})
  | Payload.I64 -> ints Bytes.set_int64_le (fun i -> data.{i})
  | Payload.I8 -> ints Bytes.set_int8 (fun i -> data.{i}));
  Bytes.to_string b

(* Fills [t] from defined bytes; [Error offset] at the first byte never
   written. *)
let fill (Tensor.Tensor t) (bytes : int option array) =
  let p = t.Tensor.payload in
  let data = p.Payload.data in
  let size = Payload.cell_bytes p.Payload.fmt in
  let rec undefined k =
    if k >= Array.length bytes then None
    else if Option.is_none bytes.(k) then Some k
    else undefined (k + 1)
  in
  match undefined 0 with
  | Some k -> Error k
  | None ->
      let b =
        Bytes.of_seq
          (Seq.map (fun x -> Char.chr (Option.get x)) (Array.to_seq bytes))
      in
      let cells get v =
        for i = 0 to Bigarray.Array1.dim data - 1 do
          v i (get b (i * size))
        done
      in
      (match p.Payload.fmt with
      | Payload.BF16 -> cells Bytes.get_uint16_le (fun i x -> data.{i} <- x)
      | Payload.Bool -> cells Bytes.get_uint8 (fun i x -> data.{i} <- x)
      | Payload.F16 -> cells Bytes.get_uint16_le (fun i x -> data.{i} <- x)
      | Payload.F32 ->
          cells Bytes.get_int32_le (fun i x ->
              data.{i} <- Int32.float_of_bits x)
      | Payload.F64 ->
          cells Bytes.get_int64_le (fun i x ->
              data.{i} <- Int64.float_of_bits x)
      | Payload.I16 -> cells Bytes.get_int16_le (fun i x -> data.{i} <- x)
      | Payload.I32 -> cells Bytes.get_int32_le (fun i x -> data.{i} <- x)
      | Payload.I64 -> cells Bytes.get_int64_le (fun i x -> data.{i} <- x)
      | Payload.I8 -> cells Bytes.get_int8 (fun i x -> data.{i} <- x));
      Ok ()
