(* Conversions between the coordinator's logical values (row-major bytes of a
   declared spec) and the two engines' own tensors. Exact: bytes are moved, never
   re-rounded, and every conversion checks the spec it is handed. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value

let ( let* ) = Result.bind
let codec_name (s : Tensor_spec.t) = Id.Codec.to_string s.dtype.codec
let shape_ints (s : Tensor_spec.t) = List.map Int64.to_int s.logical_shape

(* A live-byte limit is reported as such ("resource"), not as a conversion
   failure, so the runner can classify it. *)
let resource = "resource"

let codec_message e =
  match Err.Error.kind e with
  | Codecs.Codec_error.Resource_exceeded _ -> resource
  | k -> Format.asprintf "%a" Codecs.Codec_error.pp k

(* ---- ATen ---------------------------------------------------------------- *)

let pt2_dtype (s : Tensor_spec.t) : (Pt2_dtype.t, string) result =
  match codec_name s with
  | "f32" -> Ok Pt2_dtype.Float32
  | "i64" -> Ok Pt2_dtype.Int64
  | "bool" -> Ok Pt2_dtype.Bool
  | c -> Error ("the ATen bridge does not carry " ^ c)

let aten_of_logical (v : L.t) : (Interp_decode.tensor, string) result =
  let spec = L.spec v in
  let* dtype = pt2_dtype spec in
  let sizes = shape_ints spec in
  let t =
    {
      Pt2_tensor.dtype;
      sizes;
      strides = Pt2_tensor.contiguous_strides sizes;
      storage_offset = 0;
      data = Pt2_storage.of_string (L.contents v);
    }
  in
  try Ok (Pt2_aten.to_tensor t)
  with Failure m | Aten_tensor.Error m -> Error m

(* ATen's actual shape and dtype are read back, not assumed. *)
let logical_of_aten ctx (spec : Tensor_spec.t) (t : Interp_decode.tensor) :
    (L.t, string) result =
  try
    let t = Aten_tensor.materialize_for_raw_read t in
    let shape = Array.to_list (Aten_tensor.shape t) |> List.map Int64.of_int in
    let n = Aten_tensor.numel t in
    let read tag width put =
      let bytes = Bytes.create (n * width) in
      (match Aten_tensor.data tag t with
      | None when n <> 0 ->
          raise (Aten_tensor.Error "ATen tensor data is unavailable")
      | None -> ()
      | Some data ->
          for i = 0 to n - 1 do
            put bytes (i * width) data.{i}
          done);
      Bytes.to_string bytes
    in
    let* codec, bytes =
      match Aten_tensor.scalar_type t with
      | Aten_scalar_type.Float ->
          Ok
            ( "f32",
              read Aten_dtype.float32 4 (fun b i v ->
                  Bytes.set_int32_le b i (Int32.bits_of_float v)) )
      | Aten_scalar_type.Long ->
          Ok ("i64", read Aten_dtype.int64 8 Bytes.set_int64_le)
      | Aten_scalar_type.Bool ->
          Ok
            ( "bool",
              read Aten_dtype.bool 1 (fun b i v ->
                  Bytes.set b i (if v = 0 then '\000' else '\001')) )
      | Aten_scalar_type.Double ->
          Ok
            ( "f64",
              read Aten_dtype.float64 8 (fun b i v ->
                  Bytes.set_int64_le b i (Int64.bits_of_float v)) )
      | Aten_scalar_type.Int ->
          Ok ("i32", read Aten_dtype.int32 4 Bytes.set_int32_le)
      | _ -> Error "unsupported ATen result element type"
    in
    if
      codec <> codec_name spec
      || not (List.equal Int64.equal shape spec.logical_shape)
    then
      Error
        (Printf.sprintf "ATen result is %s%s, the graph declares %s%s" codec
           (String.concat "x" (List.map Int64.to_string shape))
           (codec_name spec)
           (String.concat "x" (List.map Int64.to_string spec.logical_shape)))
    else L.make ctx spec bytes |> Result.map_error codec_message
  with Aten_tensor.Error m -> Error m

(* ---- Native -------------------------------------------------------------- *)

let native_shape (s : Tensor_spec.t) : (Vec6.shape, string) result =
  match shape_ints s with
  | [ n; t; d; h; w; c ] -> Ok (Vec6.shape ~n ~t ~d ~h ~w ~c)
  | _ -> Error "a Native tensor has six extents"

let make_tensor : type e b.
    shape:Vec6.shape ->
    (e, b, [ `Real ]) Payload.fmt ->
    int ->
    (int -> e) ->
    Tensor.packed =
 fun ~shape fmt n get ->
  let kind : (e, b) Bigarray.kind =
    match fmt with
    | Payload.F32 -> Bigarray.float32
    | Payload.F64 -> Bigarray.float64
    | Payload.I32 -> Bigarray.int32
    | Payload.I64 -> Bigarray.int64
    | Payload.Bool -> Bigarray.int8_unsigned
    | Payload.BF16 -> Bigarray.int16_unsigned
    | Payload.F16 -> Bigarray.int16_unsigned
  in
  let data = Bigarray.Array1.create kind Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.unsafe_set data i (get i)
  done;
  Tensor.Tensor
    { Tensor.shape; payload = { Payload.fmt; quant = Payload.No_quant; data } }

let native_of_logical (v : L.t) : (Tensor.packed, string) result =
  let spec = L.spec v in
  let* shape = native_shape spec in
  let b = L.contents v in
  let n = L.element_count v in
  match codec_name spec with
  | "f32" ->
      Ok
        (make_tensor ~shape Payload.F32 n (fun i ->
             Int32.float_of_bits (String.get_int32_le b (i * 4))))
  | "f64" ->
      Ok
        (make_tensor ~shape Payload.F64 n (fun i ->
             Int64.float_of_bits (String.get_int64_le b (i * 8))))
  | "i64" ->
      Ok
        (make_tensor ~shape Payload.I64 n (fun i ->
             String.get_int64_le b (i * 8)))
  | "i32" ->
      Ok
        (make_tensor ~shape Payload.I32 n (fun i ->
             String.get_int32_le b (i * 4)))
  | "bool" ->
      Ok
        (make_tensor ~shape Payload.Bool n (fun i ->
             if b.[i] = '\000' then 0 else 1))
  | c -> Error ("the Native evaluator does not carry " ^ c)

let logical_of_native ctx (spec : Tensor_spec.t)
    (Tensor.Tensor t : Tensor.packed) : (L.t, string) result =
  let shape =
    List.map (fun a -> Int64.of_int (Vec6.get t.shape a :> int)) Axis.all
  in
  let n = Bigarray.Array1.dim t.payload.data in
  let put width f =
    let bytes = Bytes.create (n * width) in
    for i = 0 to n - 1 do
      f bytes (i * width) i
    done;
    Bytes.to_string bytes
  in
  let* codec, bytes =
    match t.payload.fmt with
    | Payload.F32 ->
        Ok
          ( "f32",
            put 4 (fun b o i ->
                Bytes.set_int32_le b o (Int32.bits_of_float t.payload.data.{i}))
          )
    | Payload.F64 ->
        Ok
          ( "f64",
            put 8 (fun b o i ->
                Bytes.set_int64_le b o (Int64.bits_of_float t.payload.data.{i}))
          )
    | Payload.I64 ->
        Ok
          ("i64", put 8 (fun b o i -> Bytes.set_int64_le b o t.payload.data.{i}))
    | Payload.I32 ->
        Ok
          ("i32", put 4 (fun b o i -> Bytes.set_int32_le b o t.payload.data.{i}))
    | Payload.Bool ->
        Ok
          ( "bool",
            put 1 (fun b o i ->
                Bytes.set b o
                  (if t.payload.data.{i} = 0 then '\000' else '\001')) )
    | _ -> Error "unsupported Native result element type"
  in
  if
    codec <> codec_name spec
    || not (List.equal Int64.equal shape spec.logical_shape)
  then
    Error
      (Printf.sprintf "Native result is %s[%s], the graph declares %s[%s]" codec
         (String.concat ";" (List.map Int64.to_string shape))
         (codec_name spec)
         (String.concat ";" (List.map Int64.to_string spec.logical_shape)))
  else L.make ctx spec bytes |> Result.map_error codec_message
