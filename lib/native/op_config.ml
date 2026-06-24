(* See op_config.mli. *)

module Nonneg = struct
  type t = int

  let of_int n =
    if n < 0 then invalid_arg "Op_config.Nonneg.of_int: negative" else n

  let to_int (x : t) = x
end

module Pos = struct
  type t = int

  let of_int n =
    if n < 1 then invalid_arg "Op_config.Pos.of_int: not positive" else n

  let to_int (x : t) = x
end

module Hw = struct
  type 'a t = { h : 'a; w : 'a }
end
