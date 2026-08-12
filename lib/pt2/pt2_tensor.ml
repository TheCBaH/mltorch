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

type error = [ Pt2_dtype.error | `Symbolic_value of string ]

(* Folded in int64, not int. Every size is individually bounded when it is
   decoded ([Pt2_pickle.int_of]), but a product of in-range factors need not be
   in range itself -- and js_of_ocaml's int is 32 bits, so the wrap would be
   silent. Narrow only after proving the product fits; see [[js_backends_design]].

   Checked at EVERY step, not only on the result. [Int64.mul] wraps silently
   too, so validating the final value alone accepts a product that already
   overflowed: sizes [max_int; 4] fold to -4, which passes any range test and is
   nonsense. The division identity is the check -- if [r / d] does not recover
   [acc], the multiply wrapped. [acc] can never be [Int64.min_int] (it is range
   checked each step, so it stays within the narrower [int] bounds), which is the
   one input that would make the division itself overflow.

   [invalid_arg] rather than a result: every caller uses the answer to index
   [data], so a product this large describes a tensor no [bytes] could hold.
   The bound is here to make that unrepresentable rather than lucky. *)
let product what dims =
  let step acc d =
    let d = Int64.of_int d in
    if Int64.equal d 0L then 0L
    else
      let r = Int64.mul acc d in
      if not (Int64.equal (Int64.div r d) acc) then
        invalid_arg (Fmt.str "%s overflows a 64-bit intermediate" what)
      else if
        Int64.compare r (Int64.of_int min_int) < 0
        || Int64.compare r (Int64.of_int max_int) > 0
      then
        invalid_arg (Fmt.str "%s is %Ld, which does not fit in an int" what r)
      else r
  in
  Int64.to_int (List.fold_left step 1L dims)

let numel t = product "tensor numel" t.sizes

(* Row-major (C-contiguous) strides for a shape. *)
let contiguous_strides sizes =
  let rec go acc = function
    | [] -> acc
    | _ :: rest ->
        let s = product "tensor stride" rest in
        go (s :: acc) rest
  in
  List.rev (go [] sizes)

let is_contiguous t =
  t.storage_offset = 0 && t.strides = contiguous_strides t.sizes

(* SymInt -> int; every shape in a stored tensor is static (never symbolic). *)
let pp_error ppf : error -> unit = function
  | #Pt2_dtype.error as e -> Pt2_dtype.pp_error ppf e
  | `Symbolic_value field ->
      Fmt.pf ppf "symbolic tensor metadata is unsupported for %s" field

let int_of_symint ?(field = "tensor metadata") = function
  | Pytorch_types.SymInt.Int i -> Err.return i
  | Pytorch_types.SymInt.Expr _ -> Err.fail (`Symbolic_value field)

let of_meta (m : Pytorch_types.TensorMeta.t) ~data =
  let open Err.Syntax in
  let* dtype =
    Pt2_dtype.of_scalar_type m.dtype |> Err.map_error (fun e -> (e :> error))
  in
  let* sizes = Err.List.map (int_of_symint ~field:"sizes") m.sizes in
  let* strides = Err.List.map (int_of_symint ~field:"strides") m.strides in
  let* storage_offset =
    int_of_symint ~field:"storage_offset" m.storage_offset
  in
  Err.return { dtype; sizes; strides; storage_offset; data }

let pp_shape ppf t = Fmt.brackets (Fmt.list ~sep:Fmt.semi Fmt.int) ppf t.sizes
let pp ppf t = Fmt.pf ppf "%s%a" (Pt2_dtype.to_string t.dtype) pp_shape t
