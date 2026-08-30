(* Bidirectional conversion between ATen opaque handles and Native 6D tensors.

   ATen layout is strided row-major; native layout is 6D NHWC dense.  For
   contiguous tensors the two layouts share the same element order after
   right-alignment (Aten_shape.of_aten), so [of_aten] is a flat copy with no
   index remapping.  A non-contiguous ATen tensor is first materialized into a
   contiguous copy (via ATen's [contiguous]), so callers need not pre-flatten.

   Only dtypes with a clean ctypes ↔ Bigarray element correspondence are
   supported (Float32, Float64 and Int64); others return an Error.

   The ATen tensor type [Interp_decode.tensor = [`atc_tensor_opaque] structure ptr]
   is the same phantom type ctypes assigns to the opaque C struct, identical to
   the type all Aten_tensor.* functions accept. *)

open Bigarray

(* Convert an ATen tensor to a Native packed tensor.  Copies the data into a new
   Bigarray so the result's lifetime is independent of the ATen handle (a
   non-contiguous input is materialized contiguous first).  Returns Error if the
   rank exceeds 6 or the dtype has no native equivalent. *)
(* Three leaves, and a shared base domain flat-included.

   [Aten_shape.error] arrives whole rather than as [Fmt.str "%a"] of it: this
   used to destructure an [Err.t] only to re-render its payload, which threw
   away the wrapper and left the caller a sentence it could not act on. *)
type error =
  [ Aten_shape.error
  | `Null_data_ptr of Aten_scalar_type.t
  | `Numel_over_limit of Vec6.Numel_bound.t
  | `Unsupported_dtype of Aten_scalar_type.t ]

let pp_error ppf : [< error ] -> unit = function
  | #Aten_shape.error as e -> Aten_shape.pp_error ppf e
  | `Null_data_ptr t ->
      Fmt.pf ppf "data_ptr returned null for ATen dtype (code %d)"
        (Aten_scalar_type.to_int t)
  | `Numel_over_limit e -> Vec6.Numel_bound.pp ppf e
  | `Unsupported_dtype t ->
      Fmt.pf ppf "unsupported ATen dtype (code %d)" (Aten_scalar_type.to_int t)

let of_aten t : (Tensor.packed, [> error ]) Err.t =
  (* Read the shape off the ORIGINAL handle and bound its element count
     BEFORE materializing: [materialize_for_raw_read] below clones a
     non-contiguous tensor contiguous, and an oversized [self] must be
     refused before that clone, not after it and the [Array1.create] that
     follows -- see op3-impl.md F1's bridge-ordering half. Sound because
     materializing never changes the shape (it clones OR returns the same
     tensor -- either way the shape array is unaffected). *)
  let shape_arr = Aten_tensor.shape t in
  match Aten_shape.of_aten shape_arr with
  | Error _ as e ->
      (* [Err.map_error] rather than a rebuild: the row only WIDENS here, and
         map_error is what preserves Aten_shape's own detection origin. *)
      Err.map_error (fun e -> (e :> error)) e
  | Ok shape -> (
      match
        Err.map_error
          (fun e -> (e :> error))
          (Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel shape)
      with
      | Error _ as e -> e
      | Ok _ -> (
          (* An ATen view doesn't share the native flat order: a permute/view
             output (the transposed fc weight feeding [addmm]) is strided, and
             a select/slice output sits at a non-zero storage offset while
             reporting itself contiguous. [materialize_for_raw_read] covers
             both — guarding on [is_contiguous] alone let the offset case
             through with wrong values. *)
          let t = Aten_tensor.materialize_for_raw_read t in
          let n = Aten_tensor.numel t in
          match Aten_tensor.scalar_type t with
          | Aten_scalar_type.Float -> (
              match Aten_tensor.data Aten_dtype.float32 t with
              | None -> Err.fail (`Null_data_ptr Aten_scalar_type.Float)
              | Some src ->
                  let data = Array1.create float32 c_layout n in
                  Array1.blit src data;
                  Err.return
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
          | Aten_scalar_type.Double -> (
              match Aten_tensor.data Aten_dtype.float64 t with
              | None -> Err.fail (`Null_data_ptr Aten_scalar_type.Double)
              | Some src ->
                  let data = Array1.create float64 c_layout n in
                  Array1.blit src data;
                  Err.return
                    (Tensor.Tensor
                       {
                         Tensor.shape;
                         payload =
                           {
                             Payload.fmt = Payload.F64;
                             quant = Payload.No_quant;
                             data;
                           };
                       }))
          | Aten_scalar_type.Long -> (
              match Aten_tensor.data Aten_dtype.int64 t with
              | None -> Err.fail (`Null_data_ptr Aten_scalar_type.Long)
              | Some src ->
                  let data = Array1.create int64 c_layout n in
                  Array1.blit src data;
                  Err.return
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
          | other -> Err.fail (`Unsupported_dtype other)))

(* Convert a Native packed float32, float64 or int64 tensor to a 1-D ATen tensor.
   The ATen tensor has a flat 1D shape [numel] to avoid rank-alignment
   ambiguity.  Callers compare element-wise, not shape-wise. *)
type to_aten_error = [ `Unsupported_native_fmt of Payload.packed_fmt ]

let pp_to_aten_error ppf : [< to_aten_error ] -> unit = function
  | `Unsupported_native_fmt (Payload.Fmt fmt) ->
      Fmt.pf ppf "unsupported native fmt %s" (Payload.fmt_name fmt)

let to_aten_flat (native : Tensor.packed) =
  let (Tensor.Tensor r) = native in
  match r.payload.fmt with
  | Payload.F32 ->
      let n = (Vec6.numel r.shape :> int) in
      Err.return
        (Aten_tensor.of_bigarray Aten_dtype.float32 r.payload.data [ n ])
  | Payload.F64 ->
      let n = (Vec6.numel r.shape :> int) in
      Err.return
        (Aten_tensor.of_bigarray Aten_dtype.float64 r.payload.data [ n ])
  | Payload.I64 ->
      let n = (Vec6.numel r.shape :> int) in
      Err.return (Aten_tensor.of_bigarray Aten_dtype.int64 r.payload.data [ n ])
  | fmt -> Err.fail (`Unsupported_native_fmt (Payload.Fmt fmt))
