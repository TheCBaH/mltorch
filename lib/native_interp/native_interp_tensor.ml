(* Runtime materialisation of a captured PT2 tensor.  It is separate from
   graph lowering because it has no access to graph-local provenance state. *)

open Pytorch_types
open Native_interp_error
open Native_interp_decode

(* Shared by every dtype arm below: shape/rank validation, the overflow-safe
   storage-range check, and the final per-coordinate materialization, factored
   so the two dtypes cannot drift on the tricky overflow arithmetic. [~read_cell]
   decodes one already-in-range element at a byte offset; [~materialize] builds
   the destination tensor (F32 or I64) from the per-coordinate function. *)
let load_dense (tensor : Pt2_tensor.t) ~element_size ~read_cell ~materialize =
  let open Err.Syntax in
  (* [shape_of_sizes] is written for graph metadata, so it throws into the
     lowering row; this is the only caller outside [lower], so it owns the
     frame and re-labels what comes out. Only the [malformed] rows can
     arrive, and re-labelling them keeps which row it was. *)
  let* shape =
    Err.Escape.with_escape (fun esc ->
        shape_of_sizes esc "tensor"
          (List.map (fun x -> SymInt.Int x) tensor.sizes))
    |> Err.map_error ~pos:__POS__ (function
      | #malformed as e -> `Tensor_bridge (e :> tensor_bridge)
      | e -> e)
  in
  if List.compare_lengths tensor.sizes tensor.strides <> 0 then
    Err.fail
      (`Tensor_bridge
         (`Rank_mismatch
            {
              Rank_mismatch.sizes = List.length tensor.sizes;
              strides = List.length tensor.strides;
            }))
  else
    let storage_index coord =
      List.fold_left ( + ) tensor.storage_offset
        (List.mapi
           (fun i stride ->
             let axis = List.nth Axis.all (6 - List.length tensor.sizes + i) in
             Dim.to_int (Vec6.get coord axis) * stride)
           tensor.strides)
    in
    let data_len = Bytes.length tensor.data in
    (* Bound the WHOLE reachable index range once, in int64, before
       materializing. [storage_index] itself runs per element in plain int,
       where a product or sum of in-range factors can still overflow -- and
       js_of_ocaml's int is 32 bits, so that wrap would be silent. Checking
       the extremes up front proves every coordinate in between is in range,
       so the inner loop stays int and stays fast. Strides may be negative
       (PyTorch stores channels-last inputs non-contiguously), hence both
       ends rather than just the maximum. See [[js_backends_design]]. *)
    (* The arithmetic that computes the bound must itself be checked, or the
       bound is decorative: [sizes=[2]] with [strides=[1 lsl 61]] makes
       [hi * 4 + 4] wrap negative, sail past a naive comparison, and then the
       per-element offset wraps to 0 and silently re-reads element 0. So each
       multiply and add is overflow-tested, and the byte bound is checked by
       DIVIDING the capacity rather than scaling [hi] up. *)
    let exception Range_overflow in
    let mul64 a b =
      if Int64.equal b 0L then 0L
      else
        let r = Int64.mul a b in
        if Int64.equal (Int64.div r b) a then r else raise Range_overflow
    in
    let add64 a b =
      let r = Int64.add a b in
      (* Overflow iff the operands agree in sign and the result does not. *)
      if
        Int64.compare (Int64.logxor a b) 0L >= 0
        && Int64.compare (Int64.logxor a r) 0L < 0
      then raise Range_overflow
      else r
    in
    let range =
      try
        let o = Int64.of_int tensor.storage_offset in
        Ok
          (List.fold_left2
             (fun (lo, hi) size stride ->
               let span =
                 mul64 (Int64.of_int (max 0 (size - 1))) (Int64.of_int stride)
               in
               if Int64.compare span 0L < 0 then (add64 lo span, hi)
               else (lo, add64 hi span))
             (o, o) tensor.sizes tensor.strides)
      with Range_overflow -> Error ()
    in
    let out_of_range (lo, hi) =
      (* Every shape has extent >= 1, so at least one element is always read
         and fewer than [element_size] bytes is always an error.
         [(data_len - element_size) / element_size] is the largest admissible
         index, computed without scaling [hi]. *)
      data_len < element_size
      || Int64.compare lo 0L < 0
      || Int64.compare hi
           (Int64.of_int ((data_len - element_size) / element_size))
         > 0
    in
    match range with
    | Error () -> Err.fail (`Tensor_bridge `Storage_index_overflow)
    | Ok (lo, hi) when out_of_range (lo, hi) ->
        Err.fail
          (`Tensor_bridge
             (`Storage_out_of_range
                { Storage_range.lo; hi; data_bytes = data_len }))
    | Ok _ -> (
        try
          Err.return
            (materialize shape (fun coord ->
                 read_cell tensor.data (storage_index coord * element_size)))
        with Invalid_argument m ->
          Err.fail (`Tensor_bridge (`Materialize_failed m)))

let tensor_of_pt2 (tensor : Pt2_tensor.t) =
  match tensor.dtype with
  | Pt2_dtype.Float32 ->
      load_dense tensor ~element_size:4
        ~read_cell:(fun data offset ->
          Int32.float_of_bits (Bytes.get_int32_le data offset))
        ~materialize:Tensor.materialize
  | Pt2_dtype.Float64 ->
      load_dense tensor ~element_size:8
        ~read_cell:(fun data offset ->
          Int64.float_of_bits (Bytes.get_int64_le data offset))
        ~materialize:(Tensor.materialize_fmt (Payload.Fmt Payload.F64))
  | Pt2_dtype.Int64 ->
      load_dense tensor ~element_size:8
        ~read_cell:(fun data offset -> Bytes.get_int64_le data offset)
        ~materialize:Tensor.materialize_i64
  | dtype -> Err.fail (`Tensor_bridge (`Unsupported_dtype dtype))
