(* Bidirectional conversion between ATen opaque handles and Native 6D tensors.

   ATen layout is strided row-major; native layout is 6D NHWC dense.  For
   contiguous tensors the two layouts share the same element order after
   right-alignment (Aten_shape.of_aten), so [of_aten] is a flat copy with no
   index remapping.  Non-contiguous ATen tensors are not yet supported — the
   caller must ensure the tensor is contiguous before converting.

   Only dtypes with a clean ctypes ↔ Bigarray element correspondence are
   supported (Float32 and Int64); others return an Error.

   The ATen tensor type [Interp_decode.tensor = [`atc_tensor_opaque] structure ptr]
   is the same phantom type ctypes assigns to the opaque C struct, identical to
   the type all Aten_tensor.* functions accept. *)

open Bigarray

(* Convert a contiguous ATen tensor to a Native packed tensor.  Copies the data
   into a new Bigarray so the result's lifetime is independent of the ATen handle.
   Returns Error if: the tensor is non-contiguous, the rank exceeds 6, or the
   dtype has no native equivalent. *)
let of_aten t : (Tensor.packed, string) result =
  if not (Aten_tensor.is_contiguous t) then
    Error "non-contiguous tensor: make contiguous before converting"
  else
    let shape_arr = Aten_tensor.shape t in
    match Aten_shape.of_aten shape_arr with
    | Error { Core.Error.kind = e; _ } ->
        Error (Format.asprintf "shape error: %a" Aten_shape.pp_error e)
    | Ok shape -> (
        let n = Aten_tensor.numel t in
        match Aten_tensor.scalar_type t with
        | Aten_scalar_type.Float -> (
            match Aten_tensor.data Aten_dtype.float32 t with
            | None -> Error "float32 data_ptr returned null"
            | Some src ->
                let data = Array1.create float32 c_layout n in
                Array1.blit src data;
                Ok
                  (Tensor.Tensor
                     {
                       Tensor.shape;
                       payload =
                         {
                           Payload.fmt = Payload.F32;
                           quant = Payload.No_quant;
                           data;
                         };
                     }))
        | Aten_scalar_type.Long -> (
            match Aten_tensor.data Aten_dtype.int64 t with
            | None -> Error "int64 data_ptr returned null"
            | Some src ->
                let data = Array1.create int64 c_layout n in
                Array1.blit src data;
                Ok
                  (Tensor.Tensor
                     {
                       Tensor.shape;
                       payload =
                         {
                           Payload.fmt = Payload.I64;
                           quant = Payload.No_quant;
                           data;
                         };
                     }))
        | other ->
            let code = Aten_scalar_type.to_int other in
            Error (Printf.sprintf "unsupported ATen dtype (code %d)" code))

(* Convert a Native packed float32 or int64 tensor to a 1-D ATen tensor.
   The ATen tensor has a flat 1D shape [numel] to avoid rank-alignment
   ambiguity.  Callers compare element-wise, not shape-wise. *)
let to_aten_flat (native : Tensor.packed) =
  let (Tensor.Tensor r) = native in
  match r.payload.fmt with
  | Payload.F32 ->
      let n = (Vec6.numel r.shape :> int) in
      Ok (Aten_tensor.of_bigarray Aten_dtype.float32 r.payload.data [ n ])
  | Payload.I64 ->
      let n = (Vec6.numel r.shape :> int) in
      Ok (Aten_tensor.of_bigarray Aten_dtype.int64 r.payload.data [ n ])
  | fmt ->
      Error
        (Printf.sprintf "to_aten_flat: unsupported native fmt %s"
           (Payload.fmt_name fmt))
