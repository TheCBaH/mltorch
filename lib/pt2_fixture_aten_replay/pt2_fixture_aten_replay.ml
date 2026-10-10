module F = Pt2_fixture
module Dtype = Pt2_checkpoint_map.Dtype

let backend = "aten-cpp"

(* The flat boundary must account for both strides and storage offset. Preserve
   ATen's actual shape and dtype, rather than imposing the reference metadata. *)
let to_logical tensor =
  let tensor = Aten_tensor.materialize_for_raw_read tensor in
  let shape =
    Array.to_list (Aten_tensor.shape tensor) |> List.map Int64.of_int
  in
  let n = Aten_tensor.numel tensor in
  let read tag dtype width put =
    let bytes = Bytes.create (n * width) in
    match Aten_tensor.data tag tensor with
    | None when n <> 0 -> Error "ATen tensor data is unavailable"
    | data ->
        Option.iter
          (fun data ->
            for i = 0 to n - 1 do
              put bytes (i * width) data.{i}
            done)
          data;
        Ok
          {
            F.Logical.data = Pt2_storage.of_string (Bytes.to_string bytes);
            dtype;
            shape;
          }
  in
  match Aten_tensor.scalar_type tensor with
  | Aten_scalar_type.Bool ->
      read Aten_dtype.bool Dtype.BOOL 1 (fun b i v ->
          Bytes.set b i (if v = 0 then '\000' else '\001'))
  | Aten_scalar_type.Double ->
      read Aten_dtype.float64 Dtype.F64 8 (fun b i v ->
          Bytes.set_int64_le b i (Int64.bits_of_float v))
  | Aten_scalar_type.Float ->
      read Aten_dtype.float32 Dtype.F32 4 (fun b i v ->
          Bytes.set_int32_le b i (Int32.bits_of_float v))
  | Aten_scalar_type.Int -> read Aten_dtype.int32 Dtype.I32 4 Bytes.set_int32_le
  | Aten_scalar_type.Long ->
      read Aten_dtype.int64 Dtype.I64 8 Bytes.set_int64_le
  | Aten_scalar_type.Byte | Aten_scalar_type.Char | Aten_scalar_type.Half
  | Aten_scalar_type.Short ->
      Error "unsupported ATen output dtype"

let execute archive ~inputs ~outputs:_ =
  try
    match Err.payload (Interp.run_named archive ~inputs) with
    | Error error -> Error (Fmt.str "%a" Interp.pp_error error)
    | Ok tensors -> Ok (List.map to_logical tensors)
  with Aten_tensor.Error error -> Error error

let replay ~consumer fixture =
  Pt2_fixture_replay.replay_with ~backend
    ~normalizations:(fun () -> [])
    ~execute ~consumer fixture
