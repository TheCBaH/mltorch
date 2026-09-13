(* Every [Kernel.Limits.Hard] constant except [eval_depth] -- see
   kernel_hard.ml for why that one alone is split out. Private module: reach
   these only through [Kernel.Limits.Hard], which re-exports this file (see
   kernel.ml). Mirrored builds copy this file unchanged; only [eval_depth]
   is ever overridden per backend. *)

(* [depth] comes from test/native/depth_probe.ml, measured under node: every
   traversal survives 1024 there and the first failures are at 2048
   (Pp.value, Value.compare, Value.hash) -- CHECK.VALUE STILL SURVIVES 2048,
   which is why the ceiling follows the minimum over all traversals rather
   than the checker's own figure. [eval_depth]'s own measurement (in
   kernel_hard.ml) follows the same probe but is a materially higher ceiling,
   since [Eval.value] recurses through whole-program depth that [depth]'s
   per-expression traversals never see. *)
let depth = 256

(* Measured under node with [Kernel_eval.value_at] over a real producer
   chain (test/native/depth_probe.ml). The frontier there is both lower and
   less stable than for a flat expression -- a 1024-transition chain
   overflowed on three runs out of four at the previous ceiling, and a
   384-transition chain of depth-4 bodies overflows while a 192-transition
   chain of depth-16 bodies does not, so transition count dominates and the
   limit does not fit a tidy cost model. Region execution classification
   adds a small fixed frame cost, so 96 preserves headroom for the mixed
   producer/body frontier on both native and JavaScript backends. *)
let eval_recursion = 96

(* Memory and time, not stack. *)
let size = 65536
let values = 65536
let dep_depth = 4096
let inputs = 4096
let outputs = 4096

(* The JS-reachable runtime domain: extents, coordinates and reachable
   storage offsets stay below 2^31. *)
let extent = 0x8000_0000L
let numel = 0x8000_0000L

(* A policy ceiling, not an empirically discovered frontier (same category as
   [max_local_slots]/[max_scan_state] below): nothing bounded actual
   allocation BYTE size before this existed -- [numel] above bounds
   JS-reachable coordinate addressability, not memory, and is deliberately
   the same for every format. Sized to the worst case [numel] already
   implicitly allowed at F32's 4-byte cell (2^31 * 4 = 8 GiB), so admitting
   this ceiling does not retroactively shrink what an existing F32 kernel
   could already request -- it only stops a WIDER-celled format (I64's 8
   bytes) from using more total bytes than that at the same cell count. *)
let max_bytes = 0x2_0000_0000L

(* Memory- and array-length-bound, not stack-bound, per the scan design
   record's array-capacity probe -- policy ceilings with deliberate
   headroom, not empirically discovered frontiers like [depth]/[eval_depth].
   [max_local_slots]/[max_scan_state] share one ceiling with
   [Expr.Scan_limits.hard_max_state], since both bound a count of resident
   [float] slots. *)
let max_local_slots = 1_048_576
let max_scan_state = 1_048_576
let max_scan_updates_per_key = 1_048_576L
let max_scan_updates_total = 100_000_000L
