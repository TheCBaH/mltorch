(* [torch.ops.aten.arange]: the bridge's own arm (op_bridge_factory.ml)
   decodes start/end/step through [Op_bridge_decode.scalar_arg_exact], which
   keeps the real [Aten_scalar.Int] alongside the float every other caller
   still reads -- so a `dtype=Long` arange with integer arguments carries
   [Factory.Arange.params.exact], and [Eval_direct]'s Arange arm generates
   its values with no float round trip at all. See the implementation
   tracker's D02/D09 notes. *)

open Helpers

let%expect_test "verify: arange.default with a small Long end" =
  verify_print ~target:"torch.ops.aten.arange.default" ~bindings:[]
    ~inputs:[ in_int "end" 5 ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: arange.start with a Float start (legacy path)" =
  verify_print ~target:"torch.ops.aten.arange.start" ~bindings:[]
    ~inputs:[ in_float "start" 0.5; in_float "end" 3.5 ];
  [%expect {| aten and native agree |}]

(* A start well past [int32] range (2^31), proving [Aten_scalar.Int]'s real
   [int64] survives the whole bridge -> [Factory.Arange.Exact.t] ->
   [Eval_direct] path intact, end to end, not merely in the standalone unit
   test in factory_test.ml (which exercises [value_i64_exact] directly). *)
let%expect_test "verify: arange.start with an int64 start past int32 range" =
  verify_print ~target:"torch.ops.aten.arange.start" ~bindings:[]
    ~inputs:[ in_int "start" 5_000_000_000; in_int "end" 5_000_000_003 ];
  [%expect {| aten and native agree |}]

(* [dispatch_print], not [verify_print]: past 2^53, [Factory.Arange.length]'s
   own element COUNT still comes from the float [start]/[stop]/[step] (a
   separate, still-open D02 gap -- confirmed empirically, see below), which
   makes real ATen and native disagree on SHAPE if compared directly. Going
   through the native-only path instead isolates exactly what changed here:
   the VALUES [Op_bridge]'s real decode produces, past the point a legacy
   float round trip would have silently corrupted them (compare this
   dispatch's own count and values to factory_test.ml's identical hand-built
   case, which documents the same count coming from the same float
   computation). *)
let%expect_test "dispatch: arange.start with a real exact int64 start past 2^53"
    =
  dispatch_print ~target:"torch.ops.aten.arange.start" ~bindings:[]
    ~inputs:
      [
        in_int "start" 9_007_199_254_740_993; in_int "end" 9_007_199_254_740_996;
      ]
    ~noutputs:1;
  [%expect
    {| tensor i64 [C=4] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996} |}]
