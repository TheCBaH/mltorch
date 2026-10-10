open Err.Syntax
open Transformers_metadata.Json_util
module L = Pt2_fixture.Logical
module D = Pt2_checkpoint_map.Dtype

let tensor (l : L.t) =
  let* () = Spec.require (List.length l.shape <= 6) "tensor rank exceeds six" in
  let* sizes =
    Err.List.map
      (fun n ->
        if n > 0L && n <= 1_000_000L then Ok (Int64.to_int n)
        else invalid "tensor extent exceeds supported bound")
      l.shape
  in
  let* elements =
    Err.List.fold_left
      (fun count n ->
        let* () =
          Spec.require
            (n = 0L || count <= Int64.div 0x200_0000L n)
            "tensor element bound"
        in
        Ok (Int64.mul count n))
      1L l.shape
  in
  let* () =
    Spec.require
      (Int64.of_int (Bigarray.Array1.dim l.data)
      = Int64.mul elements (Int64.of_int (D.byte_width l.dtype)))
      "tensor byte length"
  in
  let dtype =
    match l.dtype with
    | D.F32 -> Some Pt2_dtype.Float32
    | D.I64 -> Some Pt2_dtype.Int64
    | D.U8 -> Some Pt2_dtype.UInt8
    | D.BOOL -> Some Pt2_dtype.Bool
    | _ -> None
  in
  let* dtype =
    Err.of_option (`Metadata_invalid "unsupported input dtype") dtype
  in
  let rec strides = function
    | [] -> []
    | _ :: rest -> List.fold_left ( * ) 1 rest :: strides rest
  in
  Ok
    {
      Pt2_tensor.dtype;
      sizes;
      strides = strides sizes;
      storage_offset = 0;
      data = l.data;
    }

let logical dtype shape bytes =
  let* value =
    L.of_bytes ~dtype ~shape bytes
    |> Err.map_error (fun e -> `Metadata_invalid (Fmt.str "%a" L.pp_error e))
  in
  let+ _ = tensor value in
  value

let i64 shape values =
  let bytes = Bytes.create (8 * List.length values) in
  List.iteri (fun i v -> Bytes.set_int64_le bytes (8 * i) v) values;
  logical D.I64 shape (Bytes.to_string bytes)

let integers shape values = i64 shape (List.map Int64.of_int values)

let f32 shape values =
  let bytes = Bytes.create (4 * Array.length values) in
  Array.iteri
    (fun i v -> Bytes.set_int32_le bytes (4 * i) (Int32.bits_of_float v))
    values;
  logical D.F32 shape (Bytes.to_string bytes)

let validate (contract : Pt2_fixture.Contract.t) inputs =
  let specs = contract.inputs in
  let* () =
    Spec.require
      (List.map fst inputs
      = List.map (fun (s : Pt2_fixture.Contract.Tensor_spec.t) -> s.name) specs
      )
      "model input names/order"
  in
  Err.List.map
    (fun ((name, value), (spec : Pt2_fixture.Contract.Tensor_spec.t)) ->
      let* () =
        Spec.require
          (value.L.dtype = spec.dtype && value.shape = spec.shape)
          ("model input dtype/shape: " ^ name)
      in
      let+ tensor = tensor value in
      (name, tensor))
    (List.combine inputs specs)

let normalizations (r : Native_interp.Empty_cache_report.t) =
  List.concat_map
    (fun (s : Native_interp.Empty_cache_report.source) ->
      ("empty source: " ^ s.ssa)
      :: List.map (fun name -> "removed clone: " ^ name) s.clones
      @ List.map
          (fun (c : Native_interp.Empty_cache_report.cat) ->
            "removed cat operands: " ^ c.cat ^ " ["
            ^ String.concat ","
                (List.map
                   (fun n ->
                     string_of_int
                       (Native_interp.Empty_cache_report.Operand.to_int n))
                   c.removed)
            ^ "]")
          s.cats)
    r.sources

let run ?(on_empty_caches = ignore) archive (contract : Pt2_fixture.Contract.t)
    inputs =
  let* inputs = validate contract inputs in
  let* outputs =
    try
      Native_interp.run_named ~empty_caches:on_empty_caches
        ~dot_accumulation:Direct.Binary64 ~float_to_int:Direct.Checked archive
        ~inputs
      |> Err.map_error (fun e ->
          `Metadata_invalid (Fmt.str "%a" Native_interp.pp_error e))
    with Err.Exn.E e ->
      Err.fail (`Metadata_invalid (Fmt.str "raised: %a" Err.Exn.pp_kind e))
  in
  let* () =
    Spec.require
      (List.length outputs = List.length contract.outputs)
      "model output count"
  in
  Err.List.map
    (fun ((spec : Pt2_fixture.Contract.Tensor_spec.t), output) ->
      let* value =
        Err.import
          (function `Native_layout e -> `Metadata_invalid e)
          (Pt2_fixture_replay.to_logical output ~rank:(List.length spec.shape))
      in
      let+ () =
        Spec.require
          (value.dtype = spec.dtype && value.shape = spec.shape)
          ("model output dtype/shape: " ^ spec.name)
      in
      (spec.name, value))
    (List.combine contract.outputs outputs)
