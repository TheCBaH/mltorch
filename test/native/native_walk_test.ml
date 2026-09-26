(* Random-walk equivalence for native ops: walk each op's own config space and
   assert the Direct and Symbolic backends agree at every step. Pure OCaml (no
   libtorch) — the oracle is Direct vs Symbolic. The full-coverage sweep over
   [Native_op_walk.all_walks] lives in native_walk_coverage_test.ml, split out
   once it pushed this file over the 1000-line cap; these are the pinned,
   deeper-than-5-step fixtures for individual walks whose default coverage
   step count leaves a real blind spot. *)

module Pcg = Walk_core.Pcg

let capture f = print_string (Core.Pretty.capture_to_string f)

(* The coverage sweep runs every walk at 5 steps with seed = index, and on
   linear's seed the [bias] axis is never drawn -- so the sweep's golden shows
   `bias=true` throughout and says nothing about the other state. That is the
   shape of evidence this file exists to avoid: an axis that exists, is never
   exercised, and reads as coverage.

   [Graph_ir]'s [Linear] carries the bias as an OPTION and [Eval_op] synthesizes
   a zero one when it is absent, so the two states are different graphs through
   different code and both need the Direct-vs-Symbolic check. rms_norm's own
   optional operand happens to be drawn by the sweep; this one is not. *)
let%expect_test "native walk: linear reaches both bias states" =
  capture (fun ppf ->
      match Native_op_walk.find "linear" with
      | None -> Format.fprintf ppf "no linear walk registered@."
      | Some m ->
          assert (
            Native_op_walk.run m ~ppf ~pcg:(Pcg.seed ~seed:3L ~seq:1L) ~steps:8));
  [%expect
    {|
    step 0: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    [native] linear: direct==symbolic
    step 1 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    [native] linear: direct==symbolic
    step 2 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    [native] linear: direct==symbolic
    step 3 [out_features]: {shape=[n=1 c=8 h=1 w=4] out_features=13 bias=true}
    [native] linear: direct==symbolic
    step 4 [out_features]: {shape=[n=1 c=8 h=1 w=4] out_features=19 bias=true}
    [native] linear: direct==symbolic
    step 5 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=19 bias=false}
    [native] linear: direct==symbolic
    step 6 [input]: {shape=[n=1 c=27 h=1 w=4] out_features=19 bias=false}
    [native] linear: direct==symbolic
    step 7 [out_features]: {shape=[n=1 c=27 h=1 w=4] out_features=16 bias=false}
    [native] linear: direct==symbolic
    step 8 [input]: {shape=[n=1 c=27 h=5 w=4] out_features=16 bias=false}
    [native] linear: direct==symbolic |}]

(* The coverage sweep runs pad at 5 steps on its own index as the seed, and that
   walk never draws an extent of 1 -- so [Pad.Walk]'s config space could emit a
   configuration the op REFUSES, and the sweep stayed green. [Pad_nwalk] builds
   through [Err.or_raise], so an invalid config is not a mismatch, it is an
   uncaught exception that aborts the whole sweep.

   Seed 9 at 12 steps draws H=1 and pattern=reflect_hw, which is the pair that
   used to raise ("reflect pad of axis H by (1, 1) needs each side below the
   extent 1"). Pinned here rather than left to the sweep, because the sweep's
   seed is its INDEX in [all_walks] -- so inserting any walk ahead of pad
   silently re-rolls its dice, and that is how this surfaced. *)
let%expect_test "native walk: pad stays inside its own domain at extent 1" =
  capture (fun ppf ->
      match Native_op_walk.find "pad" with
      | None -> Format.fprintf ppf "no pad walk registered@."
      | Some m ->
          assert (
            Native_op_walk.run m ~ppf ~pcg:(Pcg.seed ~seed:9L ~seq:1L) ~steps:12));
  [%expect
    {|
    step 0: {shape=[n=1 c=3 h=6 w=6] pattern=pad_hw}
    [native] pad: direct==symbolic
    step 1 [pattern]: {shape=[n=1 c=3 h=6 w=6] pattern=pad_asym_w}
    [native] pad: direct==symbolic
    step 2 [input]: {shape=[n=1 c=3 h=6 w=1] pattern=pad_asym_w}
    [native] pad: direct==symbolic
    step 3 [pattern]: {shape=[n=1 c=3 h=6 w=1] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 4 [input]: {shape=[n=1 c=3 h=6 w=10] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 5 [pattern]: {shape=[n=1 c=3 h=6 w=10] pattern=reflect_asym_w}
    [native] pad: direct==symbolic
    step 6 [pattern]: {shape=[n=1 c=3 h=6 w=10] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 7 [input]: {shape=[n=1 c=22 h=6 w=10] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 8 [pattern]: {shape=[n=1 c=22 h=6 w=10] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 9 [input]: {shape=[n=1 c=22 h=6 w=10] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 10 [pattern]: {shape=[n=1 c=22 h=6 w=10] pattern=reflect_asym_w}
    [native] pad: direct==symbolic
    step 11 [input]: {shape=[n=1 c=22 h=6 w=9] pattern=reflect_asym_w}
    [native] pad: direct==symbolic
    step 12 [pattern]: {shape=[n=1 c=22 h=6 w=9] pattern=pad_hw}
    [native] pad: direct==symbolic |}]

(* The coverage sweep (native_walk_coverage_test.ml) only runs lstm for 5
   steps on its position's index as seed, which never draws
   [bidirectional]/[batch_first]/[num_layers]>1 together -- the same blind
   spot the [linear] comment above warns about. Seed 0 at 12 steps reaches a
   config with all three AND [bias=false] at once (step 12), so this pins
   the walk actually exercising the stacked + bidirectional + batch-first +
   biasless combination, not merely each axis in isolation. *)
let%expect_test
    "native walk: lstm reaches stacked/bidirectional/batch-first together" =
  capture (fun ppf ->
      match Native_op_walk.find "lstm" with
      | None -> Format.fprintf ppf "no lstm walk registered@."
      | Some m ->
          assert (
            Native_op_walk.run m ~ppf ~pcg:(Pcg.seed ~seed:0L ~seq:1L) ~steps:12));
  [%expect
    {|
    step 0: {hidden_size=3 input_size=2 seq=4 batch=2 num_layers=1 bidirectional=false bias=true batch_first=false}
    [native] lstm: direct==symbolic
    step 1 [bias]: {hidden_size=3 input_size=2 seq=4 batch=2 num_layers=1 bidirectional=false bias=false batch_first=false}
    [native] lstm: direct==symbolic
    step 2 [bidirectional]: {hidden_size=3 input_size=2 seq=4 batch=2 num_layers=1 bidirectional=true bias=false batch_first=false}
    [native] lstm: direct==symbolic
    step 3 [batch_first]: {hidden_size=3 input_size=2 seq=4 batch=2 num_layers=1 bidirectional=true bias=false batch_first=false}
    [native] lstm: direct==symbolic
    step 4 [bidirectional]: {hidden_size=3 input_size=2 seq=4 batch=2 num_layers=1 bidirectional=false bias=false batch_first=false}
    [native] lstm: direct==symbolic
    step 5 [seq]: {hidden_size=3 input_size=2 seq=1 batch=2 num_layers=1 bidirectional=false bias=false batch_first=false}
    [native] lstm: direct==symbolic
    step 6 [bias]: {hidden_size=3 input_size=2 seq=1 batch=2 num_layers=1 bidirectional=false bias=false batch_first=false}
    [native] lstm: direct==symbolic
    step 7 [batch_first]: {hidden_size=3 input_size=2 seq=1 batch=2 num_layers=1 bidirectional=false bias=false batch_first=true}
    [native] lstm: direct==symbolic
    step 8 [batch_first]: {hidden_size=3 input_size=2 seq=1 batch=2 num_layers=1 bidirectional=false bias=false batch_first=false}
    [native] lstm: direct==symbolic
    step 9 [hidden_size]: {hidden_size=2 input_size=2 seq=1 batch=2 num_layers=1 bidirectional=false bias=false batch_first=false}
    [native] lstm: direct==symbolic
    step 10 [batch_first]: {hidden_size=2 input_size=2 seq=1 batch=2 num_layers=1 bidirectional=false bias=false batch_first=true}
    [native] lstm: direct==symbolic
    step 11 [num_layers]: {hidden_size=2 input_size=2 seq=1 batch=2 num_layers=3 bidirectional=false bias=false batch_first=true}
    [native] lstm: direct==symbolic
    step 12 [bidirectional]: {hidden_size=2 input_size=2 seq=1 batch=2 num_layers=1 bidirectional=true bias=false batch_first=true}
    [native] lstm: direct==symbolic |}]
