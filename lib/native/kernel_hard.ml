(* [eval_depth] alone, split from kernel_hard_shared.ml (tail-call plan,
   Stage 7; see .ai/): native's own recursive [Eval.value] is bound by node's
   stack even off the JS backends (this repo's dev/CI toolchain runs under
   node), so the ceiling below is native's measured value. A mirrored build
   overrides only this file -- never kernel_hard_shared.ml -- once its own
   backend's measured frontier is known; see js/jsoo/native_js/kernel_hard.ml
   once that override exists. Private module: reach this only through
   [Kernel.Limits.Hard] (see kernel.ml). *)

include Kernel_hard_shared

(* From test/native/depth_probe.ml, re-measured after the evaluator's
   [go]/[guard]/[eval_i64] split was unified into one polymorphic-recursive
   [eval] over the whole carrier-indexed grammar (see .ai/): the bigger match
   (now also covering [I64_const]/[I64_binary]/[Float_to_i64]) costs more
   stack per level under node, moving the measured frontier from ~1536 down
   to ~1472-1504 (unstable in that band across repeated runs; 1408 was the
   last value that survived every run). 1280 is the accepted ceiling pinned
   below that with real margin, keeping roughly 1.66x headroom over
   resnet18's ~770 combined depth requirement -- less than the previous
   ceiling's 2x, since the new frontier itself is lower, but still
   comfortably clear of it. The exact failure frontier is deliberately not a
   contract: it changes with whole-program linking and V8 optimization. *)
let eval_depth = 1280
