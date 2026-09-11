(* eye.m has two independent positive output extents. *)
type t = { n : int; m : int }

let cascade c = c
let n c = c.n
let m c = c.m

let axes ~n ~m =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "m" m (fun c v -> { c with m = v });
    ]

let pp ppf c = Format.fprintf ppf "{n=%d m=%d}" c.n c.m
