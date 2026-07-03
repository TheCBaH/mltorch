(* Reduction-family walk recipe (mean.dim and friends): a rank-4 tensor reduced
   over a non-empty subset of its dims, with an optional keepdim flag. Any
   non-empty dim subset of a rank-4 tensor is valid, so [cascade] is identity. *)

type t = { n : int; c : int; h : int; w : int; dims : int list; keepdim : bool }

(* All 15 non-empty subsets of the rank-4 dims [0;1;2;3], a ready candidate set
   for the dims axis. *)
let all_dim_subsets =
  let rec subs = function
    | [] -> [ [] ]
    | x :: xs ->
        let rest = subs xs in
        rest @ List.map (fun s -> x :: s) rest
  in
  List.filter (fun s -> s <> []) (subs [ 0; 1; 2; 3 ])

let cascade c = c
let self_shape c = [ c.n; c.c; c.h; c.w ]
let dims c = c.dims
let keepdim c = c.keepdim

let axes ~n ~c ~h ~w ~dims ~keepdim =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "c" c (fun cf v -> { cf with c = v });
      field_axis "h" h (fun c v -> { c with h = v });
      field_axis "w" w (fun c v -> { c with w = v });
      field_axis "dims" dims (fun c v -> { c with dims = v });
      field_axis "keepdim" keepdim (fun c v -> { c with keepdim = v });
    ]

let pp ppf c =
  let dims_str =
    "[" ^ String.concat "," (List.map string_of_int c.dims) ^ "]"
  in
  Format.fprintf ppf "{shape=[%d,%d,%d,%d] dims=%s keepdim=%b}" c.n c.c c.h c.w
    dims_str c.keepdim
