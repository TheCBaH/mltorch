(* See op_config.mli. *)

(* The guarded scalars and the H/W pair are defined in [Core.Geometry], below
   the expression language; this module adds their wire codecs. *)

module Nonneg = struct
  include Core.Geometry.Nonneg

  let jsont : t Jsont.t =
    Jsont.map ~kind:"nonneg"
      ~dec:(fun n ->
        if n < 0 then
          Jsont.Error.msgf Jsont.Meta.none "nonneg: must be >= 0, got %d" n
        else of_int n)
      ~enc:to_int Jsont.int
end

module Pos = struct
  include Core.Geometry.Pos

  let jsont : t Jsont.t =
    Jsont.map ~kind:"pos"
      ~dec:(fun n ->
        if n < 1 then
          Jsont.Error.msgf Jsont.Meta.none "pos: must be >= 1, got %d" n
        else of_int n)
      ~enc:to_int Jsont.int
end

module Bad = struct
  type param =
    [ `Dilation
    | `Groups
    | `Kernel_size
    | `Output_padding
    | `Output_size
    | `Padding
    | `Repeats
    | `Split_size
    | `Stride ]

  type fault = [ `Negative of int | `Not_positive of int ]
  type t = { op : string; param : param; fault : fault }

  let pp_param ppf : param -> unit =
   fun p ->
    Fmt.string ppf
      (match p with
      | `Dilation -> "dilation"
      | `Groups -> "groups"
      | `Kernel_size -> "kernel_size"
      | `Output_padding -> "output_padding"
      | `Output_size -> "output_size"
      | `Padding -> "padding"
      | `Repeats -> "repeats"
      | `Split_size -> "split_size"
      | `Stride -> "stride")

  let pp ppf { op; param; fault } =
    match fault with
    | `Negative n ->
        Fmt.pf ppf "%s: %a must not be negative, got %d" op pp_param param n
    | `Not_positive n ->
        Fmt.pf ppf "%s: %a must be positive, got %d" op pp_param param n

  let pos ~op ~param n =
    if n < 1 then Error { op; param; fault = `Not_positive n }
    else Ok (Pos.of_int n)

  let nonneg ~op ~param n =
    if n < 0 then Error { op; param; fault = `Negative n }
    else Ok (Nonneg.of_int n)
end

module Hw = struct
  include Core.Geometry.Hw

  let jsont elt =
    Jsont.Object.map ~kind:"hw" (fun h w -> { h; w })
    |> Jsont.Object.mem "h" elt ~enc:(fun hw -> hw.h)
    |> Jsont.Object.mem "w" elt ~enc:(fun hw -> hw.w)
    |> Jsont.Object.finish
end
