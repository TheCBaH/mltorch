(* arange overloads need monotonic, like-typed scalar endpoints. *)
module Sv = Aten_spec.Scalar_value

type t = { start : Sv.t; end_ : Sv.t }

let cascade c = c
let start c = c.start
let end_ c = c.end_

let candidates =
  [
    { start = Sv.Int 0; end_ = Sv.Int 5 };
    { start = Sv.Int (-3); end_ = Sv.Int 2 };
    { start = Sv.Float 0.5; end_ = Sv.Float 3.5 };
    { start = Sv.Float (-1.25); end_ = Sv.Float 1.75 };
  ]

let axes ~range = Walk.[ field_axis "range" range (fun _ v -> v) ]

let pp_scalar ppf = function
  | Sv.Int i -> Fmt.pf ppf "int:%d" i
  | Sv.Float x -> Fmt.pf ppf "float:%g" x
  | Sv.Bool b -> Fmt.pf ppf "bool:%b" b

let pp ppf c =
  Format.fprintf ppf "{start=%a end=%a}" pp_scalar c.start pp_scalar c.end_
