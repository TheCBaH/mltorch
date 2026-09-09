(* [eval_depth] alone, split from kernel_hard_shared.ml (tail-call plan,
   Stage 7; see .ai/): native's own recursive [Eval.value] is bound by node's
   stack even off the JS backends (this repo's dev/CI toolchain runs under
   node), so the ceiling below is native's measured value. A mirrored build
   overrides only this file -- never kernel_hard_shared.ml -- once its own
   backend's measured frontier is known; see js/jsoo/native_js/kernel_hard.ml
   once that override exists. Private module: reach this only through
   [Kernel.Limits.Hard] (see kernel.ml). *)

include Kernel_hard_shared

(* From test/native/depth_probe.ml, re-measured after the scan primitive
   widened [Value.t] and [Eval.value] (two more constructors, plus the
   inline [Scan_at] recurrence): 1536 is the accepted ceiling pinned under
   node. The exact failure frontier is deliberately not a contract: it
   changes with whole-program linking and V8 optimization. 1536 keeps
   roughly 2x headroom over resnet18's ~770 combined depth requirement,
   matching the margin the original ceiling had. *)
let eval_depth = 1536
