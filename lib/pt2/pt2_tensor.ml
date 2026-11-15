(* A raw tensor parsed out of a `.pt2` / `.pt` archive: dtype + logical shape +
   strides (in elements) + storage offset, over the full storage blob. Keeping
   the whole blob (rather than a pre-sliced contiguous copy) lets us honour the
   non-contiguous layouts PyTorch emits — the sample input images are stored
   channels-last, so their strides are not row-major. This type is pure OCaml;
   lib/pt2_aten bridges it to a runnable [Aten_tensor.t].

   Sizes, strides and the offset are [int64]-backed domain types, not [int]:
   js_of_ocaml's [int] is 32 bits, so a checkpoint's size or stride must not be
   narrowed until a bound has been proven ([Cells]). *)

module Storage_units = Core.Storage_units

(* One axis extent: zero or more. *)
module Size : sig
  type t

  val of_int64 : int64 -> (t, [> `Negative_value of string * int64 ]) Err.t

  val of_int : int -> t
  (** For a literal; raises [Invalid_argument] on a negative. *)

  val to_int64 : t -> int64

  val to_int : t -> int
  (** Raises [Invalid_argument] unless the value fits this backend's [int]. *)

  val equal : t -> t -> bool
  val is_zero : t -> bool
  val pp : Format.formatter -> t -> unit
end = struct
  type t = int64

  let of_int64 v =
    if Int64.compare v 0L < 0 then Err.fail (`Negative_value ("size", v))
    else Err.return v

  let of_int i =
    if i < 0 then invalid_arg "Pt2_tensor.Size.of_int: negative"
    else Int64.of_int i

  let to_int64 t = t

  let to_int t =
    if Int64.compare t (Int64.of_int max_int) > 0 then
      invalid_arg "Pt2_tensor.Size.to_int: does not fit an int"
    else Int64.to_int t

  let equal = Int64.equal
  let is_zero t = Int64.equal t 0L
  let pp ppf t = Fmt.pf ppf "%Ld" t
end

(* A stride, in elements: signed (a flipped axis has a negative one). *)
module Stride : sig
  type t

  val of_int64 : int64 -> t
  val of_int : int -> t
  val of_count : Storage_units.Element_count.t -> t
  val to_int64 : t -> int64

  val to_int : t -> int
  (** Raises [Invalid_argument] unless the value fits this backend's [int]. *)

  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end = struct
  type t = int64

  let of_int64 v = v
  let of_int = Int64.of_int
  let of_count = Storage_units.Element_count.to_int64
  let to_int64 t = t

  let to_int t =
    if
      Int64.compare t (Int64.of_int max_int) > 0
      || Int64.compare t (Int64.of_int min_int) < 0
    then invalid_arg "Pt2_tensor.Stride.to_int: does not fit an int"
    else Int64.to_int t

  let equal = Int64.equal
  let pp ppf t = Fmt.pf ppf "%Ld" t
end

type t = {
  dtype : Pt2_dtype.t;
  sizes : Size.t list;
  strides : Stride.t list;
  storage_offset : Storage_units.Element_offset.t;
  data : Pt2_storage.t;
      (* the full storage, [numel storage] elements of [dtype]; read-only *)
}

type error =
  [ Pt2_dtype.error
  | `Negative_value of string * int64
  | `Symbolic_value of string ]

(* Folded in int64 with every step checked. [Int64.mul] wraps silently, so
   validating the final value alone accepts a product that already overflowed:
   sizes [max; 4] fold to a negative. The division identity is the check -- if
   [r / d] does not recover [acc], the multiply wrapped. [acc] is never
   [Int64.min_int] (it stays nonnegative), the one input that would make the
   division itself overflow.

   [invalid_arg] rather than a result: every caller uses the answer to index
   [data], so a product this large describes a tensor no [bytes] could hold. *)
let product what dims =
  let step acc d =
    let d = Size.to_int64 d in
    if Int64.equal d 0L then 0L
    else
      let r = Int64.mul acc d in
      if not (Int64.equal (Int64.div r d) acc) then
        invalid_arg (Fmt.str "%s overflows a 64-bit intermediate" what)
      else r
  in
  match Storage_units.Element_count.of_int64 (List.fold_left step 1L dims) with
  | Ok n -> n
  | Error _ -> invalid_arg (Fmt.str "%s overflows a 64-bit intermediate" what)

let numel t = product "tensor numel" t.sizes

(* [numel] for a caller that loops over the elements: raises [Invalid_argument]
   unless the count fits this backend's [int]. *)
let numel_int t =
  let n = Storage_units.Element_count.to_int64 (numel t) in
  if Int64.compare n (Int64.of_int max_int) > 0 then
    invalid_arg "Pt2_tensor.numel_int: does not fit an int"
  else Int64.to_int n

(* Row-major (C-contiguous) strides for a shape. *)
let contiguous_strides sizes =
  let rec go acc = function
    | [] -> acc
    | _ :: rest ->
        go (Stride.of_count (product "tensor stride" rest) :: acc) rest
  in
  List.rev (go [] sizes)

let is_contiguous t =
  Storage_units.Element_offset.equal t.storage_offset
    Storage_units.Element_offset.zero
  && List.equal Stride.equal t.strides (contiguous_strides t.sizes)

(* A tensor from literal [int] metadata, for a fixture or a producer that has
   the numbers in hand; raises [Invalid_argument] on a negative size or offset.
   A decoded checkpoint goes through [of_meta] or the pickle, which are total. *)
let of_ints dtype ~sizes ~strides ~storage_offset data =
  let storage_offset =
    match
      Storage_units.Element_offset.of_int64 (Int64.of_int storage_offset)
    with
    | Ok o -> o
    | Error _ -> invalid_arg "Pt2_tensor.of_ints: negative storage offset"
  in
  {
    dtype;
    sizes = List.map Size.of_int sizes;
    strides = List.map Stride.of_int strides;
    storage_offset;
    data;
  }

(* A row-major tensor over [data], offset zero. *)
let dense dtype sizes data =
  {
    dtype;
    sizes;
    strides = contiguous_strides sizes;
    storage_offset = Storage_units.Element_offset.zero;
    data;
  }

(* A leading batch axis of one over a stored tensor, as the release inputs are
   fed to a model. *)
let batched t =
  {
    t with
    sizes = Size.of_int 1 :: t.sizes;
    strides = Stride.of_count (numel t) :: t.strides;
  }

type tensor = t

(* The storage cells a tensor's layout reads, proven to lie inside its storage.

   The proof is one pass in int64 over the extremes of the reachable range
   (strides may be negative, so both ends), before any element is touched:
   every partial sum of per-axis terms lies between the lowest and highest
   cell, so once those two are inside, the per-element arithmetic below runs in
   plain [int] without wrapping, even on a 32-bit backend. The arithmetic that
   computes the bound is itself checked -- a bound computed with a wrapping
   multiply is decorative. *)
module Cells : sig
  type t

  module Rank_mismatch : sig
    type t = { sizes : int; strides : int }
  end

  module Outside_storage : sig
    type t = {
      first : int64;
      last : int64;
      storage : Storage_units.Byte_size.t;
    }
  end

  type error =
    [ `Outside_storage of Outside_storage.t
    | `Range_overflow of Core.Dim.index Core.Dim.t
    | `Rank_mismatch of Rank_mismatch.t ]

  val of_tensor :
    tensor -> element_bytes:Storage_units.Element_bytes.t -> (t, error) Err.t
  (** [Range_overflow axis]: the reach along [axis] does not fit in 63 bits, or
      the axis is longer than this backend's [int]. *)

  val iter : t -> (int -> unit) -> unit
  (** Every cell in row-major order. *)

  val cell_at : t -> Core.Dim.index Core.Dim.t list -> int
  (** The cell of one coordinate, one position per axis. Raises
      [Invalid_argument] on a wrong arity or a position past its axis. *)
end = struct
  module Rank_mismatch = struct
    type t = { sizes : int; strides : int }
  end

  module Outside_storage = struct
    type t = {
      first : int64;
      last : int64;
      storage : Storage_units.Byte_size.t;
    }
  end

  type error =
    [ `Outside_storage of Outside_storage.t
    | `Range_overflow of Core.Dim.index Core.Dim.t
    | `Rank_mismatch of Rank_mismatch.t ]

  type t = {
    empty : bool;
    offset : int;
    sizes : int array;
    strides : int array;
  }

  (* Both operands nonnegative. *)
  let mul a b =
    if Int64.equal b 0L then Some 0L
    else
      let r = Int64.mul a b in
      if Int64.equal (Int64.div r b) a then Some r else None

  let add a b =
    let r = Int64.add a b in
    if Int64.compare r 0L < 0 then None else Some r

  let of_tensor (tensor : tensor) ~element_bytes =
    let n_sizes = List.length tensor.sizes
    and n_strides = List.length tensor.strides in
    if n_sizes <> n_strides then
      Err.fail
        (`Rank_mismatch { Rank_mismatch.sizes = n_sizes; strides = n_strides })
    else
      let offset =
        Storage_units.Element_offset.to_int64 tensor.storage_offset
      in
      let empty = List.exists Size.is_zero tensor.sizes in
      if empty then
        Err.return { empty; offset = 0; sizes = [||]; strides = [||] }
      else
        let exception Overflow of int in
        try
          let neg = ref 0L and pos = ref 0L in
          List.iteri
            (fun axis (size, stride) ->
              let stride = Stride.to_int64 stride in
              let magnitude = Int64.abs stride in
              if Int64.compare magnitude 0L < 0 then raise (Overflow axis);
              let reach = Int64.sub (Size.to_int64 size) 1L in
              match mul reach magnitude with
              | None -> raise (Overflow axis)
              | Some span -> (
                  let acc = if Int64.compare stride 0L < 0 then neg else pos in
                  match add !acc span with
                  | None -> raise (Overflow axis)
                  | Some s -> acc := s))
            (List.combine tensor.sizes tensor.strides);
          let first = Int64.sub offset !neg in
          let last =
            match add offset !pos with
            | Some l -> l
            | None -> raise (Overflow 0)
          in
          let width = Storage_units.Element_bytes.to_int64 element_bytes in
          let storage = Pt2_storage.length tensor.data in
          let cells =
            Int64.div (Storage_units.Byte_size.to_int64 storage) width
          in
          if Int64.compare first 0L < 0 || Int64.compare last cells >= 0 then
            Err.fail (`Outside_storage { Outside_storage.first; last; storage })
          else
            let sizes =
              Array.of_list
                (List.mapi
                   (fun axis s ->
                     try Size.to_int s
                     with Invalid_argument _ -> raise (Overflow axis))
                   tensor.sizes)
            in
            (* [last < cells] and a stride of an axis longer than one cell is
               at most the reach, so both fit an [int]; a stride of an axis of
               one cell is never multiplied by a nonzero position. *)
            let strides =
              Array.of_list
                (List.map2
                   (fun s st ->
                     if Size.equal s (Size.of_int 1) then 0
                     else Int64.to_int (Stride.to_int64 st))
                   tensor.sizes tensor.strides)
            in
            Err.return { empty; offset = Int64.to_int offset; sizes; strides }
        with Overflow axis -> Err.fail (`Range_overflow (Core.Dim.index axis))

  let iter t f =
    if not t.empty then begin
      let rank = Array.length t.sizes in
      let index = Array.make rank 0 in
      let cell = ref t.offset in
      let running = ref true in
      while !running do
        f !cell;
        (* Odometer: last axis fastest. *)
        let d = ref (rank - 1) in
        let carry = ref true in
        while !carry && !d >= 0 do
          index.(!d) <- index.(!d) + 1;
          cell := !cell + t.strides.(!d);
          if index.(!d) = t.sizes.(!d) then begin
            cell := !cell - (t.sizes.(!d) * t.strides.(!d));
            index.(!d) <- 0;
            decr d
          end
          else carry := false
        done;
        if !carry then running := false
      done
    end

  let cell_at t indices =
    if List.compare_length_with indices (Array.length t.sizes) <> 0 then
      invalid_arg "Pt2_tensor.Cells.cell_at: wrong arity";
    let cell = ref t.offset in
    List.iteri
      (fun axis i ->
        let i = Core.Dim.to_int i in
        if i >= t.sizes.(axis) then
          invalid_arg "Pt2_tensor.Cells.cell_at: position past its axis";
        cell := !cell + (i * t.strides.(axis)))
      indices;
    !cell
end

let pp_error ppf : error -> unit = function
  | #Pt2_dtype.error as e -> Pt2_dtype.pp_error ppf e
  | `Negative_value (field, v) ->
      Fmt.pf ppf "tensor %s is %Ld, which is negative" field v
  | `Symbolic_value field ->
      Fmt.pf ppf "symbolic tensor metadata is unsupported for %s" field

(* SymInt -> int; every shape in a stored tensor is static (never symbolic). *)
let int_of_symint ?(field = "tensor metadata") = function
  | Pytorch_types.SymInt.Int i -> Err.return i
  | Pytorch_types.SymInt.Expr _ -> Err.fail (`Symbolic_value field)

let of_meta (m : Pytorch_types.TensorMeta.t) ~data =
  let open Err.Syntax in
  let* dtype =
    Pt2_dtype.of_scalar_type m.dtype |> Err.map_error (fun e -> (e :> error))
  in
  let* sizes =
    Err.List.map
      (fun s ->
        let* i = int_of_symint ~field:"sizes" s in
        Size.of_int64 (Int64.of_int i))
      m.sizes
  in
  let* strides =
    Err.List.map
      (fun s ->
        let+ i = int_of_symint ~field:"strides" s in
        Stride.of_int i)
      m.strides
  in
  let* off = int_of_symint ~field:"storage_offset" m.storage_offset in
  let* storage_offset =
    match Storage_units.Element_offset.of_int64 (Int64.of_int off) with
    | Ok o -> Ok o
    | Error _ -> Err.fail (`Negative_value ("storage_offset", Int64.of_int off))
  in
  Err.return { dtype; sizes; strides; storage_offset; data }

let pp_shape ppf t = Fmt.brackets (Fmt.list ~sep:Fmt.semi Size.pp) ppf t.sizes
let pp ppf t = Fmt.pf ppf "%s%a" (Pt2_dtype.to_string t.dtype) pp_shape t
