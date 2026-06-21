(* A raw tensor parsed out of a `.pt2` / `.pt` archive: dtype + logical shape +
   strides (in elements) + storage offset, over the full storage blob. Keeping
   the whole blob (rather than a pre-sliced contiguous copy) lets us honour the
   non-contiguous layouts PyTorch emits — the sample input images are stored
   channels-last, so their strides are not row-major. This type is pure OCaml;
   lib/pt2_aten bridges it to a runnable [Aten_tensor.t]. *)

type t = {
  dtype : Pt2_dtype.t;
  sizes : int list;
  strides : int list;
  storage_offset : int;
  data : bytes; (* the full storage, [numel storage] elements of [dtype] *)
}

let numel t = List.fold_left ( * ) 1 t.sizes

(* Row-major (C-contiguous) strides for a shape. *)
let contiguous_strides sizes =
  let rec go acc = function
    | [] -> acc
    | _ :: rest ->
        let s = List.fold_left ( * ) 1 rest in
        go (s :: acc) rest
  in
  List.rev (go [] sizes)

let is_contiguous t =
  t.storage_offset = 0 && t.strides = contiguous_strides t.sizes

(* SymInt -> int; every shape in a stored tensor is static (never symbolic). *)
let int_of_symint = function
  | Pytorch_types.SymInt.Int i -> i
  | Pytorch_types.SymInt.Expr _ ->
      failwith "Pt2_tensor: dynamic (symbolic) shape in a stored tensor"

let of_meta (m : Pytorch_types.TensorMeta.t) ~data =
  {
    dtype = Pt2_dtype.of_scalar_type m.dtype;
    sizes = List.map int_of_symint m.sizes;
    strides = List.map int_of_symint m.strides;
    storage_offset = int_of_symint m.storage_offset;
    data;
  }

let pp_shape ppf t =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    t.sizes

let pp ppf t =
  Format.fprintf ppf "%s%a" (Pt2_dtype.to_string t.dtype) pp_shape t
