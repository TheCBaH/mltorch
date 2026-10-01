(* See storage_units.mli. *)

module Quantity = struct
  type t =
    | Byte_alignment
    | Byte_offset
    | Byte_size
    | Element_bytes
    | Element_count
    | Element_offset

  let pp ppf t =
    Format.pp_print_string ppf
      (match t with
      | Byte_alignment -> "byte alignment"
      | Byte_offset -> "byte offset"
      | Byte_size -> "byte size"
      | Element_bytes -> "element width"
      | Element_count -> "element count"
      | Element_offset -> "element offset")
end

module Invalid = struct
  module Reason = struct
    type t = Negative | Not_power_of_two | Zero
  end

  type t = { quantity : Quantity.t; value : int64; reason : Reason.t }
end

module Operation = struct
  type t = Add | Advance | Align_up | Distance | Scale | Sub | To_elements

  let pp ppf t =
    Format.pp_print_string ppf
      (match t with
      | Add -> "add"
      | Advance -> "advance"
      | Align_up -> "align up"
      | Distance -> "distance"
      | Scale -> "scale"
      | Sub -> "subtract"
      | To_elements -> "convert to elements")
end

module Operands = struct
  type t = { operation : Operation.t; left : int64; right : int64 }
end

type error =
  [ `Inexact_conversion of Operands.t
  | `Invalid_quantity of Invalid.t
  | `Quantity_overflow of Operands.t
  | `Quantity_underflow of Operands.t ]

let pp_operands what ppf { Operands.operation; left; right } =
  Format.fprintf ppf "%s: %a %Ld, %Ld" what Operation.pp operation left right

let pp_error ppf : [< error ] -> unit = function
  | `Inexact_conversion o -> pp_operands "not a whole number of elements" ppf o
  | `Invalid_quantity { Invalid.quantity; value; reason } ->
      Format.fprintf ppf "invalid %a %Ld: %s" Quantity.pp quantity value
        (match reason with
        | Invalid.Reason.Negative -> "negative"
        | Not_power_of_two -> "not a power of two"
        | Zero -> "zero")
  | `Quantity_overflow o -> pp_operands "overflow" ppf o
  | `Quantity_underflow o -> pp_operands "underflow" ppf o

let invalid quantity value reason =
  Err.fail ~pos:__POS__ (`Invalid_quantity { Invalid.quantity; value; reason })

let operands operation left right = { Operands.operation; left; right }

(* A quantity that admits zero. *)
let nonneg quantity v =
  if Int64.compare v 0L < 0 then invalid quantity v Invalid.Reason.Negative
  else Err.return v

(* A quantity that does not. *)
let positive quantity v =
  if Int64.compare v 0L < 0 then invalid quantity v Invalid.Reason.Negative
  else if v = 0L then invalid quantity v Invalid.Reason.Zero
  else Err.return v

(* Both operands are nonnegative, so only overflow is possible. *)
let add operation a b =
  if Int64.compare a (Int64.sub Int64.max_int b) > 0 then
    Err.fail ~pos:__POS__ (`Quantity_overflow (operands operation a b))
  else Err.return (Int64.add a b)

let sub operation a b =
  if Int64.compare b a > 0 then
    Err.fail ~pos:__POS__ (`Quantity_underflow (operands operation a b))
  else Err.return (Int64.sub a b)

(* [a] nonnegative, [w] positive. *)
let scale a w =
  if Int64.compare a (Int64.div Int64.max_int w) > 0 then
    Err.fail ~pos:__POS__ (`Quantity_overflow (operands Operation.Scale a w))
  else Err.return (Int64.mul a w)

let exact_div a w =
  if Int64.rem a w <> 0L then
    Err.fail ~pos:__POS__
      (`Inexact_conversion (operands Operation.To_elements a w))
  else Err.return (Int64.div a w)

(* The least multiple of [alignment] at or above [t]. [alignment] is a
   positive power of two, so the remainder is exact and nonnegative; the
   padding is below [alignment] and its addition checked. *)
let round_up t alignment =
  let r = Int64.rem t alignment in
  if r = 0L then Err.return t
  else
    let pad = Int64.sub alignment r in
    if Int64.compare t (Int64.sub Int64.max_int pad) > 0 then
      Err.fail ~pos:__POS__
        (`Quantity_overflow (operands Operation.Align_up t alignment))
    else Err.return (Int64.add t pad)

let pp_int64 ppf v = Format.fprintf ppf "%Ld" v

module Element_bytes = struct
  type t = int64

  let of_int64 v = positive Quantity.Element_bytes v
  let to_int64 t = t
  let equal = Int64.equal
  let compare = Int64.compare
  let pp = pp_int64
end

module Byte_size = struct
  type t = int64

  let zero = 0L
  let of_int64 v = nonneg Quantity.Byte_size v
  let to_int64 t = t
  let equal = Int64.equal
  let compare = Int64.compare
  let max = Int64.max
  let pp = pp_int64
  let add a b = add Operation.Add a b
  let sub a b = sub Operation.Sub a b
  let succ t = add t 1L
  let pred t = sub t 1L

  let midpoint a b =
    let lo = Int64.min a b and hi = Int64.max a b in
    Int64.add lo (Int64.div (Int64.sub hi lo) 2L)

  module Nonzero = struct
    type t = int64

    let of_size v =
      if v = 0L then invalid Quantity.Byte_size v Invalid.Reason.Zero
      else Err.return v

    let to_size t = t
  end
end

module Byte_alignment = struct
  type t = int64

  let of_power quantity v =
    let open Err.Syntax in
    let* v = positive quantity v in
    if Int64.logand v (Int64.pred v) <> 0L then
      invalid quantity v Invalid.Reason.Not_power_of_two
    else Err.return v

  let of_int64 v = of_power Quantity.Byte_alignment v
  let to_int64 t = t
  let equal = Int64.equal
  let compare = Int64.compare
  let max = Int64.max
  let pp = pp_int64
  let to_size t = t
  let of_element_bytes w = of_power Quantity.Element_bytes w
  let pad size t = round_up size t
end

module Byte_offset = struct
  type t = int64

  let zero = 0L
  let of_int64 v = nonneg Quantity.Byte_offset v
  let to_int64 t = t
  let equal = Int64.equal
  let compare = Int64.compare
  let max = Int64.max
  let pp = pp_int64
  let of_size size = size
  let to_size t = t
  let advance t size = add Operation.Advance t size
  let align_up t alignment = round_up t alignment
  let is_aligned t alignment = Int64.rem t alignment = 0L
  let distance ~from t = sub Operation.Distance t from
end

module Element_count = struct
  type t = int64

  let zero = 0L
  let of_int64 v = nonneg Quantity.Element_count v
  let to_int64 t = t
  let equal = Int64.equal
  let compare = Int64.compare
  let pp = pp_int64
  let to_bytes t w = scale t w
  let of_bytes size w = exact_div size w

  module Nonzero = struct
    type t = int64

    let of_count v =
      if v = 0L then invalid Quantity.Element_count v Invalid.Reason.Zero
      else Err.return v

    let to_count t = t
  end
end

module Element_offset = struct
  type t = int64

  let zero = 0L
  let of_int64 v = nonneg Quantity.Element_offset v
  let to_int64 t = t
  let equal = Int64.equal
  let compare = Int64.compare
  let pp = pp_int64
  let advance t count = add Operation.Advance t count
  let distance ~from t = sub Operation.Distance t from
  let to_bytes t w = scale t w
  let of_bytes offset w = exact_div offset w
end
