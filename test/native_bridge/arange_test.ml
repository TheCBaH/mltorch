(* [torch.ops.aten.arange]: the bridge's own arm (op_bridge_factory.ml)
   decodes start/end/step through [Op_bridge_decode.scalar_arg_exact], which
   keeps the real [Aten_scalar.Int] alongside the float every other caller
   still reads -- so a `dtype=Long` arange with integer arguments carries
   [Factory.Arange.params.exact], and [Eval_direct]'s Arange arm generates
   its values with no float round trip at all. *)

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

(* Past 2^53, an odd exact bound and its even neighbor can round to the same
   float, so a float-based element COUNT disagreed with real ATen's integer
   count here (native computed [C=4], ATen [C=3]) even though the VALUES
   [Op_bridge]'s real decode produced were already exact.
   [Factory.Arange.length_exact] computes the count in checked [int64]
   arithmetic
   instead, so this is now a real [verify_print] end-to-end agreement check
   rather than the native-only [dispatch_print] isolation the gap used to
   require. *)
let%expect_test "verify: arange.start with a real exact int64 start past 2^53" =
  verify_print ~target:"torch.ops.aten.arange.start" ~bindings:[]
    ~inputs:
      [
        in_int "start" 9_007_199_254_740_993; in_int "end" 9_007_199_254_740_996;
      ];
  [%expect {| aten and native agree |}]
