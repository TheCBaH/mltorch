open Err.Syntax
module Dtype = Pt2_checkpoint_map.Dtype

type t = { data : Pt2_storage.t; dtype : Dtype.t; shape : int64 list }

type error =
  [ `Logical_numel_overflow
  | `Logical_over_limit of int64
  | `Logical_stride_range of int
  | `Logical_unsupported_dtype of Pt2_dtype.t ]

let pp_error ppf : error -> unit = function
  | `Logical_numel_overflow -> Fmt.string ppf "tensor element count overflows"
  | `Logical_over_limit n ->
      Fmt.pf ppf "tensor of %Ld bytes is over the limit" n
  | `Logical_stride_range dim ->
      Fmt.pf ppf "strides and offset of dimension %d reach outside the storage"
        dim
  | `Logical_unsupported_dtype d ->
      Fmt.pf ppf "unsupported dtype %s" (Pt2_dtype.to_string d)

let checked_mul a b =
  if Int64.equal a 0L || Int64.equal b 0L then Some 0L
  else
    let r = Int64.mul a b in
    if Int64.compare r 0L < 0 || not (Int64.equal (Int64.div r b) a) then None
    else Some r

let numel_of_shape shape =
  List.fold_left
    (fun acc d -> Option.bind acc (fun a -> checked_mul a d))
    (Some 1L) shape

let numel t = Option.get (numel_of_shape t.shape)
let byte_count t = Int64.mul (numel t) (Int64.of_int (Dtype.byte_width t.dtype))

let alloc n : Pt2_storage.t =
  Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

let default_max = 0x4000_0000

let sized ~max_bytes dtype shape =
  match numel_of_shape shape with
  | None -> Err.fail `Logical_numel_overflow
  | Some n -> (
      match checked_mul n (Int64.of_int (Dtype.byte_width dtype)) with
      | Some b when Int64.compare b (Int64.of_int max_bytes) <= 0 ->
          Err.return (Int64.to_int b)
      | Some b -> Err.fail (`Logical_over_limit b)
      | None -> Err.fail `Logical_numel_overflow)

let of_bytes ~dtype ~shape s =
  let* bytes = sized ~max_bytes:default_max dtype shape in
  if String.length s <> bytes then
    Err.fail (`Logical_over_limit (Int64.of_int (String.length s)))
  else Err.return { data = Pt2_storage.of_string s; dtype; shape }

let of_pt2 ?(max_bytes = default_max) (p : Pt2_tensor.t) =
  let dtype = Dtype.of_pt2 p.dtype in
  let width = Dtype.byte_width dtype in
  let shape = List.map Int64.of_int p.sizes in
  let* bytes = sized ~max_bytes dtype shape in
  (* The reachable storage range of the whole tensor, in int64, before any
     element is read: strides may be negative, and an unchecked product of
     in-range factors can wrap (js_of_ocaml's int is 32-bit). *)
  let storage_elements = Pt2_storage.length p.data / width in
  let lo = ref (Int64.of_int p.storage_offset)
  and hi = ref (Int64.of_int p.storage_offset) in
  let* () =
    if List.compare_lengths p.sizes p.strides <> 0 then
      Err.fail (`Logical_stride_range 0)
    else
      Err.List.iter
        (fun (dim, (size, stride)) ->
          if size = 0 then Err.return ()
          else
            match
              checked_mul (Int64.of_int (size - 1)) (Int64.of_int (abs stride))
            with
            | None -> Err.fail (`Logical_stride_range dim)
            | Some span ->
                if stride < 0 then lo := Int64.sub !lo span
                else hi := Int64.add !hi span;
                Err.return ())
        (List.mapi (fun i x -> (i, x)) (List.combine p.sizes p.strides))
  in
  let* () =
    if
      bytes > 0
      && (Int64.compare !lo 0L < 0
         || Int64.compare !hi (Int64.of_int storage_elements) >= 0)
    then Err.fail (`Logical_stride_range 0)
    else Err.return ()
  in
  let out = alloc bytes in
  if bytes > 0 then begin
    let rank = List.length p.sizes in
    let sizes = Array.of_list p.sizes and strides = Array.of_list p.strides in
    let index = Array.make rank 0 in
    let n = bytes / width in
    for i = 0 to n - 1 do
      let off = ref p.storage_offset in
      for d = 0 to rank - 1 do
        off := !off + (index.(d) * strides.(d))
      done;
      for b = 0 to width - 1 do
        Bigarray.Array1.unsafe_set out
          ((i * width) + b)
          (Bigarray.Array1.unsafe_get p.data ((!off * width) + b))
      done;
      (* Odometer: last dimension fastest. *)
      let d = ref (rank - 1) in
      let carry = ref true in
      while !carry && !d >= 0 do
        index.(!d) <- index.(!d) + 1;
        if index.(!d) = sizes.(!d) then begin
          index.(!d) <- 0;
          decr d
        end
        else carry := false
      done
    done
  end;
  Err.return { data = out; dtype; shape }

let get_float t i =
  match t.dtype with
  | Dtype.F32 -> Int32.float_of_bits (Pt2_storage.get_int32_le t.data (4 * i))
  | Dtype.F64 -> Int64.float_of_bits (Pt2_storage.get_int64_le t.data (8 * i))
  | _ -> invalid_arg "Logical.get_float: not a float dtype"

let get_int64 t i =
  let b = Pt2_storage.get_uint8 t.data in
  match t.dtype with
  | Dtype.BOOL | Dtype.U8 -> Int64.of_int (b i)
  | Dtype.I8 -> Int64.of_int (if b i >= 128 then b i - 256 else b i)
  | Dtype.I16 ->
      let v = b (2 * i) lor (b ((2 * i) + 1) lsl 8) in
      Int64.of_int (if v >= 32768 then v - 65536 else v)
  | Dtype.I32 -> Int64.of_int32 (Pt2_storage.get_int32_le t.data (4 * i))
  | Dtype.I64 -> Pt2_storage.get_int64_le t.data (8 * i)
  | _ -> invalid_arg "Logical.get_int64: not an integer dtype"
