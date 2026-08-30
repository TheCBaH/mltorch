(* matmul.default walk recipe, batch-less shape family only
   (`.ai/matmul_softmax_design.md` §4): both operands rank-2, self=[n,m],
   other=[m,p] -- ATen's own 2D matmul contract, and the only shape family
   the bridge arm accepts. The batched/multi-head shape family is a
   distinct, larger unit of work with no walk here yet. *)

type t = { n : int; m : int; p : int }

let cascade c = c
let self_shape c = [ c.n; c.m ]
let other_shape c = [ c.m; c.p ]

let axes ~n ~m ~p =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "m" m (fun c v -> { c with m = v });
      field_axis "p" p (fun c v -> { c with p = v });
    ]

let pp ppf c = Format.fprintf ppf "{self=[%d,%d] other=[%d,%d]}" c.n c.m c.m c.p
