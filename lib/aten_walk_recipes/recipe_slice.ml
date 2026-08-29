(* slice.Tensor walk recipe. Both bounds are [SymInt?], and
   [Aten_walk_gen.fill_non_tensor] declines an optional TYPE before it looks at
   the default, so this op has no Default tier and this entry is the only walk
   it can have.

   That is worth stating because the alternative would have been actively
   harmful. slice.Tensor has a default for every non-tensor argument
   (dim=0, start=None, end=None, step=1) and those defaults are jointly the
   IDENTITY slice, so a Default-tier walk would vary only the input shape, hold
   every parameter at the identity, and report [matched] for any implementation
   that returns its input. The generator's conservatism is what prevents it; see
   .ai/native_walk_design.md.

   ONE correlated axis of whole (rank, dim, pattern) configurations. A dim is
   only valid for a specific rank, and a (start, end) pair is only valid for a
   specific EXTENT -- so rank and dim travel together as one candidate, and the
   bounds are DERIVED from the extent currently drawn rather than stored. That
   second correlation is why [cascade] is the identity: there is no stale value
   for a repair to fix.

   [Whole] is deliberately in the candidate list. It is the configuration the
   Default tier would have produced, kept so the golden shows it beside the
   configurations that actually move data rather than pretending it is not a
   valid slice. *)

type pattern =
  | Both_negative (* [-3, -1) *)
  | Clamp_high (* end past the extent: ATen clamps, it does not reject *)
  | Clamp_low (* start below -extent: clamps to 0 *)
  | Head (* [0, n/2) *)
  | Negative_end (* [0, -1) *)
  | Negative_start (* [-2, n), so the normalization is exercised *)
  | Stride2 (* whole axis, step 2 *)
  | Stride3_offset (* [1, n) step 3 -- span and step share no factor *)
  | Tail (* [n/2, n) *)
  | Whole (* every bound absent: the identity, and a real ATen answer *)

type config = { rank : int; dim : int; pattern : pattern }
type t = { n : int; c : int; h : int; w : int; config : config }

let cascade c = c

(* The trailing [rank] extents of [n;c;h;w], as [Recipe_transpose.self_shape]
   does -- rank 1 is [w], rank 4 the whole shape. *)
let self_shape c =
  let full = [ c.n; c.c; c.h; c.w ] in
  let drop = List.length full - c.config.rank in
  List.filteri (fun i _ -> i >= drop) full

let dim c = c.config.dim

(* The extent the bounds are resolved against. [dim] may be negative, which is
   half the point of the negative candidates below. *)
let extent c =
  let shape = self_shape c in
  let d = c.config.dim in
  let d = if d < 0 then d + List.length shape else d in
  List.nth shape d

(* Every result is NON-EMPTY for any extent >= 2, which the axis lists below
   guarantee. An empty slice is legal in ATen and has no Native form, so a walk
   that drew one would report a skip and read as coverage. *)
let bounds c =
  let n = extent c in
  match c.config.pattern with
  | Both_negative -> (Some (-min n 3), Some (-1), 1)
  | Clamp_high -> (Some 0, Some (n + 5), 1)
  | Clamp_low -> (Some (-(n + 5)), None, 1)
  | Head -> (Some 0, Some (max 1 (n / 2)), 1)
  | Negative_end -> (Some 0, Some (-1), 1)
  | Negative_start -> (Some (-min n 2), None, 1)
  | Stride2 -> (None, None, 2)
  | Stride3_offset -> (Some 1, None, 3)
  | Tail -> (Some (min (n - 1) (n / 2)), None, 1)
  | Whole -> (None, None, 1)

let start c =
  let s, _, _ = bounds c in
  s

let stop c =
  let _, e, _ = bounds c in
  e

let step c =
  let _, _, s = bounds c in
  s

(* Whole (rank, dim) pairs, each spelled both ways where the rank allows it --
   [Recipe_unbind.all_dims]' rule. A dim is checked against its rank here, once,
   rather than by a filter that would silently shrink the space. *)
let all_configs =
  let dims_for rank =
    List.init rank Fun.id @ List.init rank (fun i -> i - rank)
  in
  List.concat_map
    (fun rank ->
      List.concat_map
        (fun dim ->
          List.map
            (fun pattern -> { rank; dim; pattern })
            (* Stable scenario-family order; this sequence determines the walk
               trace and intentionally differs from [pattern]'s declaration
               order. *)
            [
              Whole;
              Head;
              Tail;
              Negative_start;
              Negative_end;
              Both_negative;
              Clamp_high;
              Clamp_low;
              Stride2;
              Stride3_offset;
            ])
        (dims_for rank))
    [ 1; 2; 3; 4 ]

let axes ~n ~c ~h ~w ~config =
  Walk.
    [
      field_axis "n" n (fun c v -> { c with n = v });
      field_axis "c" c (fun cf v -> { cf with c = v });
      field_axis "h" h (fun (c : t) v -> { c with h = v });
      field_axis "w" w (fun (c : t) v -> { c with w = v });
      field_axis "config" config (fun c v -> { c with config = v });
    ]

let pp_pattern = function
  | Both_negative -> "both_neg"
  | Clamp_high -> "clamp_high"
  | Clamp_low -> "clamp_low"
  | Head -> "head"
  | Negative_end -> "neg_end"
  | Negative_start -> "neg_start"
  | Stride2 -> "stride2"
  | Stride3_offset -> "stride3_offset"
  | Tail -> "tail"
  | Whole -> "whole"

let pp ppf c =
  let ints l = String.concat "," (List.map string_of_int l) in
  let b = function None -> "none" | Some i -> string_of_int i in
  let s, e, st = bounds c in
  Format.fprintf ppf "{shape=[%s] dim=%d pattern=%s [%s,%s) step=%d}"
    (ints (self_shape c))
    c.config.dim
    (pp_pattern c.config.pattern)
    (b s) (b e) st
