(* See op_config.mli. *)

module Nonneg = struct
  type t = int

  let of_int n =
    if n < 0 then invalid_arg "Op_config.Nonneg.of_int: negative" else n

  let to_int (x : t) = x

  let jsont : t Jsont.t =
    Jsont.map ~kind:"nonneg"
      ~dec:(fun n ->
        if n < 0 then
          Jsont.Error.msgf Jsont.Meta.none "nonneg: must be >= 0, got %d" n
        else of_int n)
      ~enc:to_int Jsont.int
end

module Pos = struct
  type t = int

  let of_int n =
    if n < 1 then invalid_arg "Op_config.Pos.of_int: not positive" else n

  let to_int (x : t) = x

  let jsont : t Jsont.t =
    Jsont.map ~kind:"pos"
      ~dec:(fun n ->
        if n < 1 then
          Jsont.Error.msgf Jsont.Meta.none "pos: must be >= 1, got %d" n
        else of_int n)
      ~enc:to_int Jsont.int
end

module Hw = struct
  type 'a t = { h : 'a; w : 'a }

  let jsont elt =
    Jsont.Object.map ~kind:"hw" (fun h w -> { h; w })
    |> Jsont.Object.mem "h" elt ~enc:(fun hw -> hw.h)
    |> Jsont.Object.mem "w" elt ~enc:(fun hw -> hw.w)
    |> Jsont.Object.finish
end
