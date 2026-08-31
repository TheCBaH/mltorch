(* _to_copy.default, through the ATen-linked bridge. Restricted to the
   three-way [Pointwise.To_copy.target] domain the corpus actually uses
   (bool/float/long) -- see op_bridge_shape.ml's own comment.
   Promote with [dune promote test/native_bridge/to_copy_test.ml]. *)

open Helpers

let dispatch_to_copy ?dtype ?(non_blocking = false) self =
  dispatch_print ~target:"torch.ops.aten._to_copy.default"
    ~bindings:[ ("self", self) ]
    ~inputs:
      ([ in_tensor "self" ]
      @ (match dtype with
        | None -> []
        | Some dt -> [ in_scalar_type "dtype" dt ])
      @ [ in_bool "non_blocking" non_blocking ])
    ~noutputs:1

(* dtype=None keeps self's own (float) values unchanged, the same as an
   explicit FLOAT target -- Native has no per-tensor dtype tag to fall back
   on (see op_bridge_shape.ml's own comment), so both read as the identity. *)
let%expect_test "dispatch: _to_copy with no dtype is the identity" =
  let self = float_tensor [ 3 ] [ 1.5; -2.7; 0. ] in
  dispatch_to_copy self;
  [%expect {| tensor f32 [C=3] {1.5, -2.7, 0} |}]

let%expect_test "dispatch: _to_copy dtype=FLOAT is the identity" =
  let self = float_tensor [ 3 ] [ 1.5; -2.7; 0. ] in
  dispatch_to_copy ~dtype:PT.ScalarType.FLOAT self;
  [%expect {| tensor f32 [C=3] {1.5, -2.7, 0} |}]

(* Truncates TOWARD ZERO, matching real ATen's [static_cast<int64_t>] --
   confirmed against real ATen directly in
   test/aten_tensor_test.ml's "to.dtype casts float -> int64 (truncating)",
   and cross-checked against real ATen again below via [verify_print]. *)
let%expect_test "dispatch: _to_copy dtype=LONG truncates toward zero" =
  let self = float_tensor [ 5 ] [ -1.9; -0.5; 0.; 2.4; 3.9 ] in
  dispatch_to_copy ~dtype:PT.ScalarType.LONG self;
  [%expect {| tensor f32 [C=5] {-1, -0, 0, 2, 3} |}]

(* A genuine nonzero test, not an overfit to the corpus's own all-zero
   operand. *)
let%expect_test "dispatch: _to_copy dtype=BOOL is a nonzero test" =
  let self = float_tensor [ 5 ] [ -1.9; -0.5; 0.; 2.4; 3.9 ] in
  dispatch_to_copy ~dtype:PT.ScalarType.BOOL self;
  [%expect {| tensor f32 [C=5] {1, 1, 0, 1, 1} |}]

(* [non_blocking] carries no computational effect -- both spellings must
   build the identical result, the same [implicit] proof [expand_test.ml]
   makes for [expand.default]. *)
let%expect_test "dispatch: _to_copy ignores non_blocking" =
  let self = float_tensor [ 3 ] [ 1.5; -2.7; 0. ] in
  dispatch_to_copy ~dtype:PT.ScalarType.LONG ~non_blocking:false self;
  dispatch_to_copy ~dtype:PT.ScalarType.LONG ~non_blocking:true self;
  [%expect
    {|
    tensor f32 [C=3] {1, -2, 0}
    tensor f32 [C=3] {1, -2, 0} |}]

(* Outside the three-way corpus-evidenced domain: no Native value
   representation this op would target (see op_bridge_shape.ml's comment). *)
let%expect_test "dispatch: _to_copy rejects an unsupported dtype" =
  let self = float_tensor [ 3 ] [ 1.; 2.; 3. ] in
  dispatch_to_copy ~dtype:PT.ScalarType.INT self;
  [%expect
    {| error: _to_copy: only default/BOOL/FLOAT/LONG dtype are supported |}]

(* [Interp_decode.scalar_type_opt_arg] (the decoder [Interp_dispatch]'s
   generated arm uses, and so the one [verify_print] goes through) refuses
   to decode ANY present [ScalarType] value at all -- "these are always
   omitted... raise loudly on an unexpected present value rather than guess
   the export enum -> c10 enum translation" -- so a real-ATen-oracle walk
   through this node is only reachable with [dtype] OMITTED, the same
   restriction [zeros.default]'s/[arange.start]'s own [verify_print] tests
   above already live with. The LONG/BOOL targets are cross-checked against
   real ATen a different way instead: LONG's truncation is the same fact
   test/aten_tensor_test.ml's "to.dtype casts float -> int64 (truncating)"
   confirms directly against [Aten_tensor.O.to_dtype], bypassing this
   decoder. *)
let%expect_test "verify: _to_copy with no dtype against real ATen" =
  let self = float_tensor [ 3 ] [ 1.5; -2.7; 0. ] in
  verify_print ~target:"torch.ops.aten._to_copy.default"
    ~bindings:[ ("self", self) ]
    ~inputs:[ in_tensor "self"; in_bool "non_blocking" false ];
  [%expect {| aten and native agree |}]
