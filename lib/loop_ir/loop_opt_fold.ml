let in_domain n =
  Loop_range.within ~inner:(Loop_range.point n) ~outer:Loop_range.domain

(* [Scale (0, a) -> Const 0] is deliberately NOT here: it would drop [a]
   outright, which can silently defeat an overflow guard the lowering placed
   on [a] BECAUSE [a] itself isn't proven ([Loop_range.proven] requires a
   [Scale]'s operand to be proven regardless of the factor) -- a
   [Fail_if (Index_overflows (Scale (0, a)))] elsewhere in the same
   expression would fold to [Index_overflows (Const 0)], which can never
   fire. See the mutation test in loop_opt_differential_test.ml. Every rule
   below instead either keeps its non-constant operand verbatim, or folds two
   already-known constants, so no [Loop_range.proven] check is needed --
   only a saturating bound on the new constant itself (invariant 3: an
   aggregate is bounded by [Loop_range]'s saturating arithmetic, never by
   inspecting a value already computed with the host's own, possibly
   32-bit-under-jsoo, [int]). *)
let fold_index : Loop_index.t -> Loop_index.t = function
  | Loop_index.Add (Loop_index.Const 0, a)
  | Loop_index.Add (a, Loop_index.Const 0) ->
      a
  | Loop_index.Add (Loop_index.Const a, Loop_index.Const b) as idx ->
      let sum = Loop_range.saturating_add (Int64.of_int a) (Int64.of_int b) in
      if in_domain sum then Loop_index.Const (Int64.to_int sum) else idx
  | Loop_index.Scale (1, a) -> a
  (* [k <> 0]: the fold removes the intermediate [m * a]. With [|k| >= 1] the
     combined product is at least as large, so it still reports that
     overflow; with [k = 0] it is [0] and would not. *)
  | Loop_index.Scale (k, Loop_index.Scale (m, a)) as idx when k <> 0 ->
      let combined = Loop_range.saturating_mul k (Int64.of_int m) in
      if in_domain combined then Loop_index.Scale (Int64.to_int combined, a)
      else idx
  | Loop_index.Scale (k, Loop_index.Const a) as idx ->
      let product = Loop_range.saturating_mul k (Int64.of_int a) in
      if in_domain product then Loop_index.Const (Int64.to_int product) else idx
  | idx -> idx

let run program =
  {
    program with
    Loop_program.body =
      Loop_index_map.stmts ~f:fold_index program.Loop_program.body;
  }
