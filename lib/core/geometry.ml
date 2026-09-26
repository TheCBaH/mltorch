(* See geometry.mli. *)

module Nonneg = struct
  type t = int

  let of_int n =
    if n < 0 then invalid_arg "Op_config.Nonneg.of_int: negative" else n

  let to_int (x : t) = x
  let to_int64 (x : t) = Int64.of_int x
  let pp fmt (x : t) = Fmt.int fmt x
end

module Pos = struct
  type t = int

  let of_int n =
    if n < 1 then invalid_arg "Op_config.Pos.of_int: not positive" else n

  let to_int (x : t) = x
  let to_int64 (x : t) = Int64.of_int x
  let pp fmt (x : t) = Fmt.int fmt x
end

module Hw = struct
  type 'a t = { h : 'a; w : 'a }

  let pp pp_elt fmt { h; w } =
    Fmt.pf fmt "@[<hv>{h=%a;@ w=%a}@]" pp_elt h pp_elt w
end
