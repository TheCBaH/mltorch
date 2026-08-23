(* Self shape plus one required Scalar (mul.Scalar's "other"; no existing
   recipe carries a single required scalar -- [Recipe_bounds] carries an
   optional PAIR, for clamp/hardtanh). *)

module Sv = Aten_spec.Scalar_value

type t = { n : int; c : int; h : int; w : int; value : Sv.t }

let cascade c = c
let self_shape c = [ c.n; c.c; c.h; c.w ]
let value c = c.value

(* 0.1 is not f32-exact (unlike 0.5/2.5, both exact dyadic fractions) --
   included so the walk actually exercises the scalar's f32 narrowing, not
   just its Int/Float tagging. *)
let candidates = [ Sv.Int 2; Sv.Int (-3); Sv.Float 0.1; Sv.Float 2.5 ]

let axes ~n ~c ~h ~w ~value =
  Walk.
    [
      field_axis "n" n (fun (t : t) v -> { t with n = v });
      field_axis "c" c (fun (t : t) v -> { t with c = v });
      field_axis "h" h (fun (t : t) v -> { t with h = v });
      field_axis "w" w (fun (t : t) v -> { t with w = v });
      field_axis "value" value (fun (t : t) v -> { t with value = v });
    ]

let pp_value ppf = function
  | Sv.Int i -> Fmt.pf ppf "int:%d" i
  | Sv.Float f -> Fmt.pf ppf "float:%g" f
  | Sv.Bool b -> Fmt.pf ppf "bool:%b" b

let pp ppf c =
  Fmt.pf ppf "{shape=[%d,%d,%d,%d] value=%a}" c.n c.c c.h c.w pp_value c.value
