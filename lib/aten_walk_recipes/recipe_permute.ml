(* permute.default walk recipe: rank and the permutation drawn as ONE
   correlated axis, like [Recipe_transpose] -- a permutation is only valid for
   the rank it permutes, so mutating rank and dims independently would
   generate invalid combinations ATen rejects. Candidates are curated full
   permutations per rank (identity, full reverse, and a couple of structurally
   distinct partial swaps), not an exhaustive enumeration -- the walk's job is
   to exercise the bridge's [native_perm_of_aten] on genuinely different
   permutation SHAPES, not every permutation of small ranks. [cascade] is the
   identity: every candidate is a whole valid (rank, dims) pair by
   construction. *)

type config = { rank : int; dims : int list }
type t = { n : int; c : int; h : int; w : int; config : config }

let cascade c = c

(* The trailing [rank] extents of [n;c;h;w], same convention as
   [Recipe_transpose.self_shape]. *)
let self_shape c =
  let full = [ c.n; c.c; c.h; c.w ] in
  let drop = List.length full - c.config.rank in
  List.filteri (fun i _ -> i >= drop) full

let dims c = c.config.dims

let all_configs =
  [
    (* rank 2: identity, swap *)
    { rank = 2; dims = [ 0; 1 ] };
    { rank = 2; dims = [ 1; 0 ] };
    (* rank 3: identity, full reverse, rotate *)
    { rank = 3; dims = [ 0; 1; 2 ] };
    { rank = 3; dims = [ 2; 1; 0 ] };
    { rank = 3; dims = [ 1; 2; 0 ] };
    (* rank 4: identity, full reverse, NCHW->NHWC, swap trailing pair *)
    { rank = 4; dims = [ 0; 1; 2; 3 ] };
    { rank = 4; dims = [ 3; 2; 1; 0 ] };
    { rank = 4; dims = [ 0; 2; 3; 1 ] };
    { rank = 4; dims = [ 0; 1; 3; 2 ] };
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
  let dims_str =
    "[" ^ String.concat "," (List.map string_of_int c.config.dims) ^ "]"
  in
  Format.fprintf ppf "{shape=%s rank=%d dims=%s}" shape_str c.config.rank
    dims_str
