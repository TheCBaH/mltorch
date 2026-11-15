(* Convert a parsed [Pt2_tensor.t] into a runnable [Aten_tensor.t]. We hand ATen
   PyTorch's native (storage, sizes, strides, offset) representation via
   [Aten_tensor.of_storage], so the layout — contiguous weights or channels-last
   inputs — is reproduced faithfully and the strided→dense work is done by ATen,
   not in OCaml. Float32, int64 and bool are needed by the current models; extend per
   dtype as required.

   Storage bytes are little-endian and the host is little-endian, so each cell
   can be decoded directly into the matching Bigarray element type. ATen takes
   native [int]s: this library is native-only, and a size, stride or offset
   past [int] is refused by the narrowing ([Invalid_argument]). *)

let sizes (t : Pt2_tensor.t) = List.map Pt2_tensor.Size.to_int t.sizes
let strides (t : Pt2_tensor.t) = List.map Pt2_tensor.Stride.to_int t.strides

let storage_offset (t : Pt2_tensor.t) =
  Int64.to_int (Core.Storage_units.Element_offset.to_int64 t.storage_offset)

let cells (t : Pt2_tensor.t) width =
  match Core.Storage_units.Element_bytes.of_int64 (Int64.of_int width) with
  | Ok w -> Pt2_storage.cell_count t.data w
  | Error _ -> invalid_arg "Pt2_aten.cells"

let to_tensor (t : Pt2_tensor.t) =
  match t.Pt2_tensor.dtype with
  | Pt2_dtype.Bool ->
      let data = t.Pt2_tensor.data in
      let ba =
        Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout
          (cells t 1)
      in
      for i = 0 to Bigarray.Array1.dim ba - 1 do
        ba.{i} <- Pt2_storage.cell_uint8 data ~cell:i
      done;
      Aten_tensor.of_storage Aten_dtype.bool ba ~sizes:(sizes t)
        ~strides:(strides t) ~storage_offset:(storage_offset t)
  | Pt2_dtype.Float32 ->
      let data = t.Pt2_tensor.data in
      let n = cells t 4 in
      let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
      for i = 0 to n - 1 do
        Bigarray.Array1.unsafe_set ba i
          (Int32.float_of_bits (Pt2_storage.cell_int32_le data ~cell:i))
      done;
      Aten_tensor.of_storage Aten_dtype.float32 ba ~sizes:(sizes t)
        ~strides:(strides t) ~storage_offset:(storage_offset t)
  | Pt2_dtype.Int64 ->
      let data = t.Pt2_tensor.data in
      let n = cells t 8 in
      let ba = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout n in
      for i = 0 to n - 1 do
        Bigarray.Array1.unsafe_set ba i (Pt2_storage.cell_int64_le data ~cell:i)
      done;
      Aten_tensor.of_storage Aten_dtype.int64 ba ~sizes:(sizes t)
        ~strides:(strides t) ~storage_offset:(storage_offset t)
  | d -> failwith ("Pt2_aten: unsupported dtype " ^ Pt2_dtype.to_string d)
