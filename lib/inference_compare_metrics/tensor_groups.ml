open Inference_compare_schema

type t = {
  groups : int;
  group_size : int;
  comp_dims : int array;
  comp_strides : int array;  (** Element strides of the complementary axes. *)
  reduced_offsets : int array;
      (** Offsets of a group's elements from its base. *)
}

let fail e = Err.fail ~pp_error:Metric_config_error.pp e
let ( let* ) = Err.bind

let make ~shape (partition : Tensor_metric.partition) =
  let rank = List.length shape in
  let* elements =
    match Inference_compare_codecs.Shape_math.byte_count ~width:1 shape with
    | Ok n -> Ok n
    | Error _ -> fail Metric_config_error.Payload_too_large
  in
  ignore elements;
  let dims =
    Array.of_list (Inference_compare_codecs.Shape_math.to_ints shape)
  in
  let strides =
    Array.of_list
      (Inference_compare_codecs.Shape_math.strides (Array.to_list dims))
  in
  let reduced =
    match partition with
    | Whole_tensor -> Ok (List.init rank Fun.id)
    | Reduce_axes [] -> fail Metric_config_error.Empty_axes
    | Reduce_axes axes ->
        let axes = List.map Tensor_axis.to_index axes in
        let rec check seen = function
          | [] -> Ok (List.sort Int.compare seen)
          | a :: rest ->
              if a >= rank then
                fail (Metric_config_error.Axis_out_of_range { axis = a; rank })
              else if List.mem a seen then
                fail (Metric_config_error.Duplicate_axis a)
              else check (a :: seen) rest
        in
        check [] axes
  in
  let* reduced = reduced in
  let comp =
    List.filter (fun a -> not (List.mem a reduced)) (List.init rank Fun.id)
  in
  let prod axes = List.fold_left (fun acc a -> acc * dims.(a)) 1 axes in
  let groups = prod comp and group_size = prod reduced in
  let reduced_a = Array.of_list reduced in
  let offsets =
    Array.init group_size (fun r ->
        (* Unravel [r] over the reduced dims, last axis fastest. *)
        let rest = ref r and off = ref 0 in
        for k = Array.length reduced_a - 1 downto 0 do
          let d = dims.(reduced_a.(k)) in
          off := !off + (!rest mod d * strides.(reduced_a.(k)));
          rest := !rest / d
        done;
        !off)
  in
  Ok
    {
      groups;
      group_size;
      comp_dims = Array.of_list (List.map (fun a -> dims.(a)) comp);
      comp_strides = Array.of_list (List.map (fun a -> strides.(a)) comp);
      reduced_offsets = offsets;
    }

let groups t = t.groups
let group_size t = t.group_size

let comp_index t g =
  let n = Array.length t.comp_dims in
  let idx = Array.make n 0 in
  let rest = ref g in
  for k = n - 1 downto 0 do
    idx.(k) <- !rest mod t.comp_dims.(k);
    rest := !rest / t.comp_dims.(k)
  done;
  idx

let coords t g = Array.to_list (Array.map Int64.of_int (comp_index t g))

let indices t g =
  let idx = comp_index t g in
  let base = ref 0 in
  Array.iteri (fun k i -> base := !base + (i * t.comp_strides.(k))) idx;
  Array.map (fun o -> !base + o) t.reduced_offsets
