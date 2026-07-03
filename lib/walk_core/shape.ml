(* A walk-config tensor shape over the six logical axes (N T D H W C,
   channels-last — same order as the native engine's Vec6, but neutral: no
   native/aten dependency). Carried as a single compound config entry; backends
   convert it to their own shape type. Mutations go through [resize], which
   clamps to the per-axis cap and the total-numel budget from [Limits], so a
   shape is valid by construction. *)

type t = { n : int; t : int; d : int; h : int; w : int; c : int }

type axis = [ `N | `H | `W | `C ]
(** The mutable axes for the 4D ops (T/D stay 1). *)

let numel s = s.n * s.t * s.d * s.h * s.w * s.c

let cap (l : Limits.t) = function
  | `N -> l.max_batch
  | `H | `W -> l.max_extent
  | `C -> l.max_channels

let get s = function `N -> s.n | `H -> s.h | `W -> s.w | `C -> s.c

let set s axis v =
  match axis with
  | `N -> { s with n = v }
  | `H -> { s with h = v }
  | `W -> { s with w = v }
  | `C -> { s with c = v }

let valid (l : Limits.t) s =
  s.n >= 1 && s.t >= 1 && s.d >= 1 && s.h >= 1 && s.w >= 1 && s.c >= 1
  && s.n <= l.max_batch && s.h <= l.max_extent && s.w <= l.max_extent
  && s.c <= l.max_channels
  && numel s <= l.max_numel

(* Set [axis] to [v], clamped to [1, cap]; if that breaks the numel budget,
   shrink the axis further so the total fits. *)
let resize (l : Limits.t) s ~axis v =
  let v = max 1 (min v (cap l axis)) in
  let s' = set s axis v in
  if numel s' <= l.max_numel then s'
  else
    let others = numel s' / get s' axis in
    let max_v = max 1 (l.max_numel / max 1 others) in
    set s axis (max 1 (min v max_v))

let pp fmt s = Format.fprintf fmt "[n=%d c=%d h=%d w=%d]" s.n s.c s.h s.w
