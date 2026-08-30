(* expand.default walk recipe: a rank-4 [n,c,h,w] shape plus a broadcast
   PATTERN naming which axis (if any) [self] holds at 1 while [size] asks for
   the full extent there -- the shape ATen's [inferExpandGeometryImpl] (real
   stride vs. a broadcast stride 0) actually discriminates, not an
   independently-drawn [size] (which would usually be broadcast-incompatible
   and degrade to a skip, the same failure mode [Recipe_view]'s header
   warns about). [cascade] recomputes both [self] and [size] from
   [n,c,h,w,pattern] after every mutation, whichever axis a step touched, the
   same discipline [Recipe_view] uses for its own [target].

   [Add_leading] exercises the one shape [expand] alone can produce and
   [view] cannot: [size] naming one more entry than [self]'s own rank (a
   leading axis expand ADDS, self is only rank 3 here). *)

type pattern =
  | No_broadcast  (** self already equals size: the degenerate expand *)
  | Broadcast_n
  | Broadcast_c
  | Broadcast_h
  | Broadcast_w
  | Add_leading  (** self is rank 3; size adds a leading [n] *)

type t = {
  n : int;
  c : int;
  h : int;
  w : int;
  pattern : pattern;
  self : int list;
  size : int list;
}

let self_of pattern (n, c, h, w) =
  match pattern with
  | No_broadcast -> [ n; c; h; w ]
  | Broadcast_n -> [ 1; c; h; w ]
  | Broadcast_c -> [ n; 1; h; w ]
  | Broadcast_h -> [ n; c; 1; w ]
  | Broadcast_w -> [ n; c; h; 1 ]
  | Add_leading -> [ c; h; w ]

let cascade t =
  {
    t with
    self = self_of t.pattern (t.n, t.c, t.h, t.w);
    size = [ t.n; t.c; t.h; t.w ];
  }

(* Stable scenario order; intentionally differs from [pattern]'s declaration
   order, the same convention [Recipe_view.all_patterns] follows. *)
let all_patterns =
  [
    No_broadcast;
    Broadcast_w;
    Broadcast_h;
    Add_leading;
    Broadcast_c;
    Broadcast_n;
  ]

let self_shape t = t.self
let size t = t.size

(* [h]/[w] need the explicit [(c : t)] annotation -- [Walk.[ ... ]] locally
   opens [Walk], which re-exports [Walk_core.Walk.hw] (a compound {h; w}
   record), so an unpinned record update on those two field names would
   otherwise disambiguate against THAT type rather than this recipe's own,
   the same annotation [Recipe_view.axes] needs for the identical reason. *)
let axes ~n ~c ~h ~w ~pattern =
  Walk.
    [
      field_axis "n" n (fun t v -> { t with n = v });
      field_axis "c" c (fun t v -> { t with c = v });
      field_axis "h" h (fun (t : t) v -> { t with h = v });
      field_axis "w" w (fun (t : t) v -> { t with w = v });
      field_axis "pattern" pattern (fun t v -> { t with pattern = v });
    ]

let pp_pattern ppf = function
  | No_broadcast -> Format.fprintf ppf "none"
  | Broadcast_n -> Format.fprintf ppf "n"
  | Broadcast_c -> Format.fprintf ppf "c"
  | Broadcast_h -> Format.fprintf ppf "h"
  | Broadcast_w -> Format.fprintf ppf "w"
  | Add_leading -> Format.fprintf ppf "add_leading"

let pp_ints ppf l =
  Format.fprintf ppf "[%s]" (String.concat "," (List.map string_of_int l))

let pp ppf t =
  Format.fprintf ppf "{self=%a broadcast=%a size=%a}" pp_ints t.self pp_pattern
    t.pattern pp_ints t.size
