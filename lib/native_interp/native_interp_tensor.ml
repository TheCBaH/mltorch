(* Runtime materialisation of a captured PT2 tensor.  It is separate from
   graph lowering because it has no access to graph-local provenance state. *)

open Pytorch_types
open Native_interp_error
open Native_interp_decode

(* Shared by every dtype arm below: shape validation, the overflow-safe
   storage-range proof ([Pt2_tensor.Cells]: both ends of the reachable range,
   checked in int64, strides may be negative), and the final per-coordinate
   materialization, factored so the dtypes cannot drift on the tricky overflow
   arithmetic. [~read_cell] decodes one already-in-range element by its cell;
   [~materialize] builds the destination tensor (F32 or I64) from the
   per-coordinate function. *)
let load_dense (tensor : Pt2_tensor.t) ~element_bytes ~read_cell ~materialize =
  let open Err.Syntax in
  match
    List.map (fun s -> SymInt.Int (Pt2_tensor.Size.to_int s)) tensor.sizes
  with
  | exception Invalid_argument m ->
      Err.fail (`Tensor_bridge (`Materialize_failed m))
  | sym_sizes -> (
      (* [shape_of_sizes] is written for graph metadata, so it throws into the
         lowering row; this is the only caller outside [lower], so it owns the
         frame and re-labels what comes out. Only the [malformed] rows can
         arrive, and re-labelling them keeps which row it was. *)
      let* shape =
        Err.Escape.with_escape (fun esc ->
            shape_of_sizes esc "tensor" sym_sizes)
        |> Err.map_error ~pos:__POS__ (function
          | #malformed as e -> `Tensor_bridge (e :> tensor_bridge)
          | e -> e)
      in
      let* cells =
        Pt2_tensor.Cells.of_tensor tensor ~element_bytes
        |> Err.map_error ~pos:__POS__ (function
          | `Rank_mismatch { Pt2_tensor.Cells.Rank_mismatch.sizes; strides } ->
              `Tensor_bridge (`Rank_mismatch { Rank_mismatch.sizes; strides })
          | `Range_overflow _ -> `Tensor_bridge `Storage_index_overflow
          | `Outside_storage
              { Pt2_tensor.Cells.Outside_storage.first; last; storage } ->
              `Tensor_bridge
                (`Storage_out_of_range
                   { Storage_range.lo = first; hi = last; data_bytes = storage }))
      in
      let rank = List.length tensor.sizes in
      let axes = List.init rank (fun i -> List.nth Axis.all (6 - rank + i)) in
      try
        Err.return
          (materialize shape (fun coord ->
               read_cell tensor.data
                 ~cell:
                   (Pt2_tensor.Cells.cell_at cells
                      (List.map (Vec6.get coord) axes))))
      with Invalid_argument m ->
        Err.fail (`Tensor_bridge (`Materialize_failed m)))

let element_bytes n =
  match Core.Storage_units.Element_bytes.of_int64 n with
  | Ok b -> b
  | Error _ -> invalid_arg "element_bytes"

let bytes_1 = element_bytes 1L
let bytes_4 = element_bytes 4L
let bytes_8 = element_bytes 8L

let tensor_of_pt2 (tensor : Pt2_tensor.t) =
  match tensor.dtype with
  | Pt2_dtype.Float32 ->
      load_dense tensor ~element_bytes:bytes_4
        ~read_cell:(fun data ~cell ->
          Int32.float_of_bits (Pt2_storage.cell_int32_le data ~cell))
        ~materialize:Tensor.materialize
  | Pt2_dtype.Float64 ->
      load_dense tensor ~element_bytes:bytes_8
        ~read_cell:(fun data ~cell ->
          Int64.float_of_bits (Pt2_storage.cell_int64_le data ~cell))
        ~materialize:(Tensor.materialize_fmt (Payload.Fmt Payload.F64))
  | Pt2_dtype.Bool ->
      (* A nonzero byte reads as true; the stored payload is canonical 0/1. *)
      load_dense tensor ~element_bytes:bytes_1
        ~read_cell:(fun data ~cell -> Pt2_storage.cell_uint8 data ~cell <> 0)
        ~materialize:Tensor.materialize_bool
  | Pt2_dtype.Int64 ->
      load_dense tensor ~element_bytes:bytes_8
        ~read_cell:(fun data ~cell -> Pt2_storage.cell_int64_le data ~cell)
        ~materialize:Tensor.materialize_i64
  | dtype -> Err.fail (`Tensor_bridge (`Unsupported_dtype dtype))
