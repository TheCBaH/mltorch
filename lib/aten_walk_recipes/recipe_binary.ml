(* Binary elementwise (sub.Tensor, and any future add/mul/div.Tensor walk that
   reuses it): a rank-4 shape plus a broadcast PATTERN drawn as a single
   correlated axis -- lhs and rhs equal, or one operand with a single axis
   forced to extent 1. Candidates are whole valid (lhs, rhs) shape PAIRS,
   never two independently mutated shape fields: mutating each operand's shape
   on its own axis would generate invalid pairs (a genuine broadcast
   mismatch, which ATen rejects) and degrade to skips that read as coverage
   rather than exercise it. [cascade] is therefore the identity -- every
   [pattern] is valid for every base shape, so no combination needs
   filtering. *)

type pattern = Equal | Lhs_axis of int | Rhs_axis of int
type t = { n : int; c : int; h : int; w : int; pattern : pattern }

let cascade c = c

(* Every 1-axis pattern, once per rank-4 axis, on each side. *)
let all_patterns =
  Equal :: List.concat_map (fun i -> [ Lhs_axis i; Rhs_axis i ]) [ 0; 1; 2; 3 ]

let base c = [ c.n; c.c; c.h; c.w ]
let with_axis_one shape i = List.mapi (fun j v -> if i = j then 1 else v) shape

let lhs_shape c =
  match c.pattern with
  | Equal | Rhs_axis _ -> base c
  | Lhs_axis i -> with_axis_one (base c) i

let rhs_shape c =
  match c.pattern with
  | Equal | Lhs_axis _ -> base c
  | Rhs_axis i -> with_axis_one (base c) i

let axes ~n ~c ~h ~w ~pattern =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "c" c (fun cf v -> { cf with c = v });
      field_axis "h" h (fun (c : t) v -> { c with h = v });
      field_axis "w" w (fun (c : t) v -> { c with w = v });
      field_axis "pattern" pattern (fun c v -> { c with pattern = v });
    ]

let pp_pattern ppf = function
  | Equal -> Format.fprintf ppf "equal"
  | Lhs_axis i -> Format.fprintf ppf "lhs[%d]=1" i
  | Rhs_axis i -> Format.fprintf ppf "rhs[%d]=1" i

let pp ppf c =
  Format.fprintf ppf "{shape=[%d,%d,%d,%d] pattern=%a}" c.n c.c c.h c.w
    pp_pattern c.pattern
