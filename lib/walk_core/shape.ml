(* A walk-config tensor shape over the six logical axes (N T D H W C,
   channels-last — same order as the native engine's Vec6, but neutral: no
   native/aten dependency). Carried as a single compound config entry; backends
   convert it to their own shape type. Mutations go through [resize], which
   clamps to the per-axis cap and the total-numel budget from [Limits], so a
   shape is valid by construction: every axis is an extent, so none is below 1. *)

type ext = Core.Dim.extent Core.Dim.t
type t = { n : ext; t : ext; d : ext; h : ext; w : ext; c : ext }

type axis = [ `C | `H | `N | `W ]
(** The mutable axes for the 4D ops (T/D stay 1). *)

let numel s = Core.Dim.(one_count *@ s.n *@ s.t *@ s.d *@ s.h *@ s.w *@ s.c)

let cap (l : Limits.t) : axis -> ext = function
  | `C -> Core.Dim.extent l.max_channels
  | `H | `W -> Core.Dim.extent l.max_extent
  | `N -> Core.Dim.extent l.max_batch

let get s : axis -> ext = function
  | `C -> s.c
  | `H -> s.h
  | `N -> s.n
  | `W -> s.w

let set s (axis : axis) (v : ext) =
  match axis with
  | `C -> { s with c = v }
  | `H -> { s with h = v }
  | `N -> { s with n = v }
  | `W -> { s with w = v }

let valid (l : Limits.t) s =
  s.n <= cap l `N
  && s.h <= cap l `H
  && s.w <= cap l `W
  && s.c <= cap l `C
  && (numel s :> int) <= l.max_numel

(* Set [axis] to [v], clamped to [1, cap]; if that breaks the numel budget,
   shrink the axis further so the total fits. *)
let resize (l : Limits.t) s ~(axis : axis) (v : ext) =
  let v = min v (cap l axis) in
  let s' = set s axis v in
  if (numel s' :> int) <= l.max_numel then s'
  else
    let others = (numel (set s' axis Core.Dim.one) :> int) in
    let max_v = max 1 (l.max_numel / max 1 others) in
    set s axis (min v (Core.Dim.extent max_v))

let pp fmt s =
  Format.fprintf fmt "[n=%a c=%a h=%a w=%a]" Core.Dim.pp s.n Core.Dim.pp s.c
    Core.Dim.pp s.h Core.Dim.pp s.w
