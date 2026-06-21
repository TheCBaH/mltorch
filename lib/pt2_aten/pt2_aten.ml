(* Convert a parsed [Pt2_tensor.t] into a runnable [Aten_tensor.t]. We hand ATen
   PyTorch's native (storage, sizes, strides, offset) representation via
   [Aten_tensor.of_storage], so the layout — contiguous weights or channels-last
   inputs — is reproduced faithfully and the strided→dense work is done by ATen,
   not in OCaml. Only float32 is needed by the current models; extend per dtype
   as required.

   The storage bytes are little-endian float32 and the host is little-endian, so
   each 4-byte cell is already the native float bit pattern. *)

let to_tensor (t : Pt2_tensor.t) =
  match t.Pt2_tensor.dtype with
  | Pt2_dtype.Float32 ->
      let data = t.Pt2_tensor.data in
      let n = Bytes.length data / 4 in
      let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
      for i = 0 to n - 1 do
        Bigarray.Array1.unsafe_set ba i
          (Int32.float_of_bits (Bytes.get_int32_le data (i * 4)))
      done;
      Aten_tensor.of_storage Aten_dtype.float32 ba ~sizes:t.Pt2_tensor.sizes
        ~strides:t.Pt2_tensor.strides
        ~storage_offset:t.Pt2_tensor.storage_offset
  | d -> failwith ("Pt2_aten: unsupported dtype " ^ Pt2_dtype.to_string d)
