(* Bound-pair walk recipe, shared by clamp.default and hardtanh.default: a
   rank-4 tensor plus a lower/upper scalar bound pair.

   The bound pair is ONE axis rather than two independent ones. clamp's bounds
   are both optional and ATen rejects the both-absent pair outright ("At least
   one of 'min' or 'max' must not be None"), so an axis per bound could step to
   an invalid configuration that the walk would then report as a spurious
   mismatch. Enumerating whole valid pairs makes that state unreachable instead
   of relying on [cascade] to repair it after the fact.

   The candidate pairs deliberately include the cases the native op has to get
   right and which the schema defaults alone never reach: one-sided bounds,
   MobileNet's relu6 window, reversed bounds (ATen permits min > max and yields
   max everywhere), and NaN bounds (ATen fills the whole result with NaN). *)

module Sv = Aten_spec.Scalar_value

type bounds = { min : Sv.t option; max : Sv.t option }
type t = { n : int; c : int; h : int; w : int; bounds : bounds }

let cascade c = c
let self_shape c = [ c.n; c.c; c.h; c.w ]
let min c = c.bounds.min
let max c = c.bounds.max

(* Both bounds present, for hardtanh, whose schema makes them mandatory. *)
let required c =
  ( Option.value c.bounds.min ~default:(Sv.Int (-1)),
    Option.value c.bounds.max ~default:(Sv.Int 1) )

(* Valid clamp configurations: never (None, None). *)
let optional_pairs =
  [
    { min = Some (Sv.Int 0); max = Some (Sv.Int 6) };
    { min = Some (Sv.Int 0); max = None };
    { min = None; max = Some (Sv.Int 6) };
    { min = Some (Sv.Float (-1.5)); max = Some (Sv.Float 1.5) };
    (* reversed: ATen's min(max(a, lo), hi) gives hi everywhere *)
    { min = Some (Sv.Int 1); max = Some (Sv.Int (-1)) };
    { min = Some (Sv.Float Float.nan); max = Some (Sv.Int 1) };
    { min = Some (Sv.Int 0); max = Some (Sv.Float Float.nan) };
  ]

(* Both-present configurations, for hardtanh. *)
let required_pairs =
  [
    { min = Some (Sv.Int (-1)); max = Some (Sv.Int 1) };
    { min = Some (Sv.Int 0); max = Some (Sv.Int 6) };
    { min = Some (Sv.Float (-2.5)); max = Some (Sv.Float 2.5) };
    { min = Some (Sv.Int 1); max = Some (Sv.Int (-1)) };
  ]

let axes ~n ~c ~h ~w ~bounds =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "c" c (fun cf v -> { cf with c = v });
      field_axis "h" h (fun (c : t) v -> { c with h = v });
      field_axis "w" w (fun (c : t) v -> { c with w = v });
      field_axis "bounds" bounds (fun c v -> { c with bounds = v });
    ]

let pp_bound ppf = function
  | None -> Format.pp_print_string ppf "none"
  | Some (Sv.Int i) -> Format.fprintf ppf "%d" i
  | Some (Sv.Float f) -> Format.fprintf ppf "%g" f
  | Some (Sv.Bool b) -> Format.fprintf ppf "%b" b

let pp ppf c =
  Format.fprintf ppf "{shape=[%d,%d,%d,%d] min=%a max=%a}" c.n c.c c.h c.w
    pp_bound c.bounds.min pp_bound c.bounds.max
