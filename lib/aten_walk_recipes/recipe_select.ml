(* select.int needs a rank-aware dim and an index valid for that dim's extent.
   Keep rank/dim/index scenario as one candidate, deriving the index from the
   current extent so shape mutations cannot make a previously legal case fail. *)

type index = First | Last | Negative_last
type config = { rank : int; dim : int; index : index }
type t = { n : int; c : int; h : int; w : int; config : config }

let cascade c = c

let self_shape c =
  let full = [ c.n; c.c; c.h; c.w ] in
  let drop = List.length full - c.config.rank in
  List.filteri (fun i _ -> i >= drop) full

let dim c = c.config.dim

let extent c =
  let d =
    if c.config.dim < 0 then c.config.dim + c.config.rank else c.config.dim
  in
  List.nth (self_shape c) d

let index c =
  match c.config.index with
  | First -> 0
  | Last -> extent c - 1
  | Negative_last -> -1

let all_configs =
  [
    { rank = 1; dim = 0; index = First };
    { rank = 1; dim = -1; index = Negative_last };
    { rank = 2; dim = 0; index = Last };
    { rank = 2; dim = -1; index = First };
    { rank = 3; dim = 1; index = Negative_last };
    { rank = 3; dim = -3; index = Last };
    { rank = 4; dim = 0; index = Last };
    { rank = 4; dim = 2; index = First };
    { rank = 4; dim = -1; index = Negative_last };
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

let pp_index = function
  | First -> "first"
  | Last -> "last"
  | Negative_last -> "neg_last"

let pp ppf c =
  Format.fprintf ppf "{shape=[%s] rank=%d dim=%d index=%d (%s)}"
    (String.concat "," (List.map string_of_int (self_shape c)))
    c.config.rank c.config.dim (index c) (pp_index c.config.index)
