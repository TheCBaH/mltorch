(* matmul.default walk recipe, covering BOTH shape families the bridge
   accepts (`.ai/matmul_softmax_design.md` §4-5): self=[d,h,n,m],
   other=[d,h,m,p], with [d]/[h] shared between both operands so every
   combination is a VALID accepted graph -- there is nothing to "cascade" out
   of the config space. [d]=[h]=1 exercises the batch-less family (binds to
   the existing [Bmm] node); [d]>1 or [h]>1 exercises the batched/multi-head
   family (binds to the new [Batched_matmul] node) -- both through the SAME
   walk, since the bridge's OWN dispatch already distinguishes them by the
   raw ATen rank/extents, not by anything this recipe chooses on its behalf.

   [d_bc]/[h_bc] add a genuine BROADCAST configuration on each of [d]/[h]
   independently -- one operand's own axis pinned to 1 while the other keeps
   the shared value -- "flash-admissible by construction" the same way
   [Recipe_sdpa]'s [mask_kind] booleans are: every representable value is a
   legal ATen broadcast (equal, or one side 1), so there is nothing for
   [cascade] to repair here either.

   [self_rank]/[other_rank] add a genuine UNEQUAL-RANK configuration by
   dropping [self_shape]/[other_shape]'s own LEADING axes independently
   (rank 2 keeps only [n,m]/[m,p]; rank 3 keeps [h,n,m]/[h,m,p]; rank 4 keeps
   everything). This is valid by construction too: real ATen right-aligns a
   lower-rank operand's batch prefix against the higher-rank one's own
   TRAILING batch axes with an implicit leading 1, and since both operands'
   [d]/[h] values are the SAME shared fields, a dropped axis is exactly
   equivalent to that axis already being 1 there -- never a genuine
   mismatch, at any combination of [self_rank]/[other_rank]/[d]/[h]/[d_bc]/
   [h_bc]. The importer's own floor is rank 2 on each operand
   (`rank_a >= 2 && rank_b >= 2`), so neither field ever goes below that. *)

type broadcast_side = Neither | Self_one | Other_one

type t = {
  d : int;
  h : int;
  d_bc : broadcast_side;
  h_bc : broadcast_side;
  self_rank : int;
  other_rank : int;
  n : int;
  m : int;
  p : int;
}

let cascade c = c

(* [Neither] keeps the shared value on both sides; [Self_one]/[Other_one]
   pin exactly one side to 1, per ATen's own "equal, or one side 1" rule. *)
let sides bc shared =
  match bc with
  | Neither -> (shared, shared)
  | Self_one -> (1, shared)
  | Other_one -> (shared, 1)

(* The innermost [rank] axes of a rank-4 [full] list, the same right-
   alignment [Aten_shape.used_axes] applies to a Native six-axis frame --
   here over the recipe's own four ATen-facing positions. *)
let innermost rank full =
  let drop = 4 - rank in
  List.filteri (fun i _ -> i >= drop) full

let self_shape c =
  let d_self, _ = sides c.d_bc c.d in
  let h_self, _ = sides c.h_bc c.h in
  innermost c.self_rank [ d_self; h_self; c.n; c.m ]

let other_shape c =
  let _, d_other = sides c.d_bc c.d in
  let _, h_other = sides c.h_bc c.h in
  innermost c.other_rank [ d_other; h_other; c.m; c.p ]

let axes ~d ~h ~d_bc ~h_bc ~self_rank ~other_rank ~n ~m ~p =
  Walk.
    [
      field_axis "d" d (fun (c : t) v -> { c with d = v });
      field_axis "h" h (fun (c : t) v -> { c with h = v });
      field_axis "d_bc" d_bc (fun (c : t) v -> { c with d_bc = v });
      field_axis "h_bc" h_bc (fun (c : t) v -> { c with h_bc = v });
      field_axis "self_rank" self_rank (fun (c : t) v ->
          { c with self_rank = v });
      field_axis "other_rank" other_rank (fun (c : t) v ->
          { c with other_rank = v });
      field_axis "n" n (fun (c : t) v -> { c with n = v });
      field_axis "m" m (fun (c : t) v -> { c with m = v });
      field_axis "p" p (fun (c : t) v -> { c with p = v });
    ]

let pp_shape ppf l =
  Format.fprintf ppf "[%s]" (String.concat "," (List.map string_of_int l))

let pp_bc ppf = function
  | Neither -> Format.pp_print_string ppf "neither"
  | Self_one -> Format.pp_print_string ppf "self"
  | Other_one -> Format.pp_print_string ppf "other"

let pp ppf c =
  Format.fprintf ppf "{self=%a other=%a d_bc=%a h_bc=%a}" pp_shape
    (self_shape c) pp_shape (other_shape c) pp_bc c.d_bc pp_bc c.h_bc
