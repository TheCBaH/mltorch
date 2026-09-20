(* [Check.value_i64]: the int64-rooted twin of [Check.value], for a bare
   [int64 Value.t] (not one reached only through a [Float_to_i64] wrapper --
   the shape a standalone typed pixel value has, e.g. an exact int64 Arange:
   [start + i*step] with no locals, sources or reducers of its own). Mirrors
   [value_test.ml]'s own "Check rejects what composition can still break"
   cases, at int64. *)

open Expr

let ok = Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Check.pp_error

(* [start + i*step] over the C-axis output coordinate, cast from the exact
   position index -- the closed-form shape an exact int64 Arange needs (no
   [I64_load], no locals, no reducers). *)
let arange_shaped ~start ~step =
  Value.i64_add (Value.i64_const start)
    (Value.i64_mul (Value.i64_const step)
       (Value.float_to_i64
          (Value.value_of_index (Index.of_position (Index.output Axis.C)))))

let%expect_test "Check.value_i64: a closed int64 pixel expression is ok" =
  Fmt.pr "%a@." ok (Check.value_i64 (arange_shaped ~start:2L ~step:3L));
  [%expect {| ok |}]

let%expect_test "Check.value_i64: an unbound local is caught" =
  let local, _ = Builder.run_from Builder.initial Builder.fresh_local in
  Fmt.pr "%a@." ok (Check.value_i64 (Value.i64_local local));
  [%expect {| unbound local #0 |}]

let%expect_test "Check.value_i64: a free reducer is caught" =
  let stray, _ = Builder.run_from Builder.initial Builder.fresh_reduce in
  Fmt.pr "%a@." ok
    (Check.value_i64
       (Value.float_to_i64
          (Value.value_of_index (Index.of_position (Index.reduce stray)))));
  [%expect {| free reducer #0 |}]

(* [duplicate_binder_i64] has no binding construct to trip on its own (no
   [Reduce]/[Scan_at] exists at int64); a
   duplicate binder can only be REACHED through an embedded float subtree,
   via [Float_to_i64]. Two independently built (both from [Builder.initial])
   reduction fragments composed without freshening, exactly
   [value_test.ml]'s own real-world capture shape, wrapped in [Float_to_i64]
   so [Check.value_i64] is what actually catches it. *)
let%expect_test
    "Check.value_i64: a duplicate binder reached through Float_to_i64 is caught"
    =
  let frag () =
    Builder.run_from Builder.initial
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 2)
         (fun r -> Builder.return (Value.value_of_index (Index.of_position r))))
  in
  let inner, _ = frag () in
  let captured, _ =
    Builder.run_from Builder.initial
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 2)
         (fun _ -> Builder.return inner))
  in
  Fmt.pr "%a@." ok (Check.value_i64 (Value.float_to_i64 captured));
  [%expect {| reducer #0 is bound twice on one path |}]

let%expect_test "Check.value_i64: size/depth limits are exact, like Check.value"
    =
  let e = arange_shaped ~start:2L ~step:3L in
  Fmt.pr "size %d depth %d@." (Fold.size_i64 e) (Fold.depth_i64 e);
  [%expect {| size 8 depth 6 |}];
  Fmt.pr "%a  %a@." ok
    (Check.value_i64 ~max_size:7 e)
    ok
    (Check.value_i64 ~max_depth:5 e);
  [%expect {| size exceeds limit 7  depth exceeds limit 5 |}];
  Fmt.pr "at the limit: %a %a@." ok
    (Check.value_i64 ~max_size:8 e)
    ok
    (Check.value_i64 ~max_depth:6 e);
  [%expect {| at the limit: ok ok |}]
