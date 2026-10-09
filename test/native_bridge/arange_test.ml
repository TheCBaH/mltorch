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

(* [arange.start_step]: the overload whose step is required. The Transformers
   SmolVLM vision tower serializes it with all three bounds as floats and no
   dtype: start = step = 1/32, end = 1, i.e. the 31 positions 1/32 ... 31/32. *)
let%expect_test
    "verify: arange.start_step as the SmolVLM vision tower writes it" =
  verify_print ~target:"torch.ops.aten.arange.start_step" ~bindings:[]
    ~inputs:
      [ in_float "start" 0.03125; in_float "end" 1.0; in_float "step" 0.03125 ];
  [%expect {| aten and native agree |}]

(* A descending range is refused, as it is for the other overloads: the engine
   admits a positive step only. The refusal is the restriction, not a gap in
   this overload. *)
let%expect_test
    "verify: arange.start_step integer, uneven steps, descending refused" =
  verify_print ~target:"torch.ops.aten.arange.start_step" ~bindings:[]
    ~inputs:[ in_int "start" 2; in_int "end" 11; in_int "step" 3 ];
  verify_print ~target:"torch.ops.aten.arange.start_step" ~bindings:[]
    ~inputs:[ in_int "start" 10; in_int "end" (-3); in_int "step" (-4) ];
  verify_print ~target:"torch.ops.aten.arange.start_step" ~bindings:[]
    ~inputs:[ in_float "start" 0.0; in_float "end" 1.0; in_float "step" 0.3 ];
  [%expect
    {|
    aten and native agree
    [verify] torch.ops.aten.arange.start_step: bridge error: arange(10, -3, -4): only a positive step is supported
    aten and native agree
    aten and native agree |}]

let%expect_test "dispatch: arange.start_step keeps its own dtype rule" =
  dispatch_print ~target:"torch.ops.aten.arange.start_step" ~bindings:[]
    ~inputs:[ in_int "start" 2; in_int "end" 11; in_int "step" 3 ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.arange.start_step" ~bindings:[]
    ~inputs:[ in_float "start" 0.5; in_float "end" 2.0; in_float "step" 0.5 ]
    ~noutputs:1;
  [%expect
    {|
    tensor i64 [C=3] {2, 5, 8}
    tensor f32 [C=3] {0.5, 1, 1.5} |}]

(* The schema makes [step] required; an absent one must not read as 1. *)
let%expect_test "dispatch: arange.start_step without a step is refused" =
  dispatch_print ~target:"torch.ops.aten.arange.start_step" ~bindings:[]
    ~inputs:[ in_int "start" 2; in_int "end" 11 ]
    ~noutputs:1;
  [%expect {| error: missing required argument "step" |}]
