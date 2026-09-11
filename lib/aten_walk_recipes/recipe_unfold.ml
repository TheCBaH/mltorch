(* unfold.default: dimension and window mode are correlated.  Window values are
   derived from the selected extent so every shape mutation remains valid. *)
type mode = Overlap | Stepped
type config = { dimension : int; mode : mode }
type t = { n : int; c : int; h : int; w : int; config : config }

let cascade c = c
let self_shape c = [ c.n; c.c; c.h; c.w ]
let dimension c = c.config.dimension

let extent c =
  let d =
    if c.config.dimension < 0 then c.config.dimension + 4
    else c.config.dimension
  in
  List.nth (self_shape c) d

let size c = min 3 (extent c)
let step c = match c.config.mode with Overlap -> 1 | Stepped -> min 2 (size c)

let all_configs =
  [
    { dimension = 0; mode = Overlap };
    { dimension = 1; mode = Stepped };
    { dimension = 2; mode = Overlap };
    { dimension = 3; mode = Stepped };
    { dimension = -1; mode = Overlap };
    { dimension = -2; mode = Stepped };
    { dimension = -3; mode = Overlap };
    { dimension = -4; mode = Stepped };
  ]

let axes ~n ~c ~h ~w ~config =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "c" c (fun cf v -> { cf with c = v });
      field_axis "h" h (fun (c : t) v -> { c with h = v });
      field_axis "w" w (fun (c : t) v -> { c with w = v });
      field_axis "config" config (fun c v -> { c with config = v });
    ]

let pp_mode = function Overlap -> "overlap" | Stepped -> "stepped"

let pp ppf c =
  Format.fprintf ppf "{shape=[%d,%d,%d,%d] dim=%d size=%d step=%d %s}" c.n c.c
    c.h c.w (dimension c) (size c) (step c) (pp_mode c.config.mode)
