(* matmul.default walk recipe, covering BOTH shape families the bridge
   accepts (`.ai/matmul_softmax_design.md` §4-5): self=[d,h,n,m],
   other=[d,h,m,p], with [d]/[h] shared between both operands so every
   combination is a VALID accepted graph -- there is nothing to "cascade" out
   of the config space. [d]=[h]=1 exercises the batch-less family (binds to
   the existing [Bmm] node); [d]>1 or [h]>1 exercises the batched/multi-head
   family (binds to the new [Batched_matmul] node) -- both through the SAME
   walk, since the bridge's OWN dispatch already distinguishes them by the
   raw ATen rank/extents, not by anything this recipe chooses on its behalf. *)

type t = { d : int; h : int; n : int; m : int; p : int }

let cascade c = c
let self_shape c = [ c.d; c.h; c.n; c.m ]
let other_shape c = [ c.d; c.h; c.m; c.p ]

let axes ~d ~h ~n ~m ~p =
  Walk.
    [
      field_axis "d" d (fun (c : t) v -> { c with d = v });
      field_axis "h" h (fun (c : t) v -> { c with h = v });
      field_axis "n" n (fun (c : t) v -> { c with n = v });
      field_axis "m" m (fun (c : t) v -> { c with m = v });
      field_axis "p" p (fun (c : t) v -> { c with p = v });
    ]

let pp ppf c =
  Format.fprintf ppf "{self=[%d,%d,%d,%d] other=[%d,%d,%d,%d]}" c.d c.h c.n c.m
    c.d c.h c.m c.p
