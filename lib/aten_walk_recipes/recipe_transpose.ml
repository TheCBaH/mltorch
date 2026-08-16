(* transpose.int walk recipe: rank and the dim pair drawn as ONE correlated
   axis, never independently -- a dim pair is only valid for a specific rank
   (e.g. (2,3) needs rank >= 4), so mutating rank and dims separately would
   generate invalid combinations ATen rejects and degrade to skips that read
   as coverage. Candidates cover positive, negative, equal, leading and
   trailing pairs, across ranks 2, 3 and 4. [cascade] is the identity: every
   candidate is a whole valid (rank, dim0, dim1) triple by construction. *)

type config = { rank : int; dim0 : int; dim1 : int }
type t = { n : int; c : int; h : int; w : int; config : config }

let cascade c = c

(* The trailing [rank] extents of [n;c;h;w] -- rank 2 is [h;w], rank 3 is
   [c;h;w], rank 4 is the whole shape. *)
let self_shape c =
  let full = [ c.n; c.c; c.h; c.w ] in
  let drop = List.length full - c.config.rank in
  List.filteri (fun i _ -> i >= drop) full

let dim0 c = c.config.dim0
let dim1 c = c.config.dim1

let all_configs =
  [
    (* rank 2: positive, negative, equal *)
    { rank = 2; dim0 = 0; dim1 = 1 };
    { rank = 2; dim0 = -1; dim1 = -2 };
    { rank = 2; dim0 = 0; dim1 = 0 };
    (* rank 3: leading pair, trailing pair, mixed sign *)
    { rank = 3; dim0 = 0; dim1 = 1 };
    { rank = 3; dim0 = 1; dim1 = 2 };
    { rank = 3; dim0 = -1; dim1 = -3 };
    (* rank 4: leading/trailing extremes, negative, equal *)
    { rank = 4; dim0 = 0; dim1 = 3 };
    { rank = 4; dim0 = -2; dim1 = -3 };
    { rank = 4; dim0 = 2; dim1 = 2 };
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

let pp ppf c =
  let shape_str =
    "[" ^ String.concat "," (List.map string_of_int (self_shape c)) ^ "]"
  in
  Format.fprintf ppf "{shape=%s rank=%d dims=(%d,%d)}" shape_str c.config.rank
    c.config.dim0 c.config.dim1
