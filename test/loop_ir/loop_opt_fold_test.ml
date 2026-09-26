open Loop_ir

(* Stage 2 of the Loop IR optimization plan: constant folding must stay inside
   [Loop_range.domain] ([-2^31, 2^31 - 1]), the same bound the rest of the
   Loop IR guarantees so a value is exact in both an OCaml [int] and a
   js_of_ocaml [int] (32 bits, [.ai/loop_ir_optimization_design.md] invariant
   3). Folding the sum/product with the host's own (possibly 32-bit-under-jsoo)
   [int] arithmetic, rather than [Loop_range]'s saturating [int64], would
   silently wrap under jsoo where the unfolded expression -- evaluated as a
   plain JavaScript [Number] at runtime -- would not: exactly the divergence
   these cases check for. Runs under node too via this library's
   [(modes best js)]. *)

let show label idx =
  match Loop_opt_fold.fold_index idx with
  | Loop_index.Const n -> Fmt.pr "%s: folded to Const %d@." label n
  | Loop_index.Scale (k, Loop_index.Var _) ->
      Fmt.pr "%s: folded to Scale (%d, _)@." label k
  | _ -> Fmt.pr "%s: not folded@." label

let%expect_test "constant folding stays in the proven index domain near 2^31" =
  (* Deep inside the domain: folds. *)
  show "small sum" (Loop_index.Add (Loop_index.Const 100, Loop_index.Const 200));
  (* Exactly at the domain's upper edge (2^31 - 1): folds. *)
  show "sum at the edge"
    (Loop_index.Add
       (Loop_index.Const 2_000_000_000, Loop_index.Const 147_483_647));
  (* One past the edge: the fold is declined, not wrapped. *)
  show "sum past the edge"
    (Loop_index.Add
       (Loop_index.Const 2_000_000_000, Loop_index.Const 147_483_648));
  (* Exactly at the domain's lower edge (-2^31): folds. *)
  show "negative sum at the edge"
    (Loop_index.Add
       (Loop_index.Const (-2_000_000_000), Loop_index.Const (-147_483_648)));
  (* One past the lower edge: declined. *)
  show "negative sum past the edge"
    (Loop_index.Add
       (Loop_index.Const (-2_000_000_000), Loop_index.Const (-147_483_649)));
  (* Nested Scale, combined factor still small: folds to one Scale. *)
  show "nested scale in range"
    (Loop_index.Scale
       (1000, Loop_index.Scale (2000, Loop_index.Var (Loop_var.of_int 0))));
  (* Nested Scale whose combined factor alone overflows the domain, though
     neither factor does on its own: declined. *)
  show "nested scale overflowing only combined"
    (Loop_index.Scale
       (100_000, Loop_index.Scale (100_000, Loop_index.Var (Loop_var.of_int 0))));
  [%expect
    {|
    small sum: folded to Const 300
    sum at the edge: folded to Const 2147483647
    sum past the edge: not folded
    negative sum at the edge: folded to Const -2147483648
    negative sum past the edge: not folded
    nested scale in range: folded to Scale (2000000, _)
    nested scale overflowing only combined: not folded
    |}]

(* Review regression: [Scale (k, Scale (m, a)) -> Scale (k * m, a)] removes the
   intermediate [m * a], and with [k = 0] that intermediate can overflow while
   the combined [0 * a] cannot -- the same dropped-overflow shape as
   [Scale (0, a) -> Const 0]. [Scale (2, Const 2e9)] is left unfolded (its
   product is out of the domain), so the outer rule is what sees it. *)
let%expect_test "a zero outer scale keeps the inner scale's overflow" =
  let inner = Loop_index.Scale (2, Loop_index.Const 2_000_000_000) in
  let idx = Loop_index.Scale (0, inner) in
  let out =
    Loop_fixtures.buffer 0 (Loop_fixtures.shape_w 1) Loop_fixtures.f32
      Loop_buffer.Output
  in
  let program =
    Loop_fixtures.program ~buffers:[ out ]
      [
        Loop_stmt.Fail_if
          ( Loop_bool.Index_overflows idx,
            Loop_failure.Index_overflow { index = idx } );
      ]
  in
  let bind = Loop_fixtures.bind_none in
  let reference =
    Err.map_error
      (fun (e : Loop_interp.error) -> (e :> Kernel_eval.error))
      (Loop_interp.run program ~bind)
  in
  let folded = Loop_interp.run (Loop_opt_fold.run program) ~bind in
  Fmt.pr "%a@." Loop_check.pp_verdict
    (Loop_check.compare ~reference ~loop:folded);
  [%expect {| agree on failure: index_overflow |}]
