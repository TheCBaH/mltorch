type t = { n : int; c : int; h : int; w : int; dim : int }

let cascade c = c
let self_shape c = [ c.n; c.c; c.h; c.w ]
let dim c = c.dim

let extent c =
  let d = if c.dim < 0 then c.dim + 4 else c.dim in
  List.nth (self_shape c) d

let split_size c = min 3 (extent c)

let axes ~n ~c ~h ~w ~dim =
  Walk.
    [
      field_axis "n" n (fun (x : t) v -> { x with n = v });
      field_axis "c" c (fun (x : t) v -> { x with c = v });
      field_axis "h" h (fun (x : t) v -> { x with h = v });
      field_axis "w" w (fun (x : t) v -> { x with w = v });
      field_axis "dim" dim (fun (x : t) v -> { x with dim = v });
    ]

let pp ppf c =
  Format.fprintf ppf "{shape=[%d,%d,%d,%d] dim=%d split_size=%d}" c.n c.c c.h
    c.w c.dim (split_size c)

let split_sizes c =
  let e = extent c in
  let left = max 1 (e / 2) in
  [ left; e - left ]
