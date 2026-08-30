(* ATen-vs-native on the serialized-scalar path. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/scalar_verify_test.ml]. *)

open Helpers

(* ---- ATen-vs-native on the serialized-scalar path ------------------------- *)

(* The bridge arms above pin the native compute against hand-derived values, and
   the generated walks (test/native_walk_test.ml) compare whole ops against real
   ATen — but neither reaches THIS case. A compile-time scalar in a Tensor slot
   (`aten.add.Tensor(x, 3)`, which is how MobileNet-v3's hardsigmoid is
   serialised) is skipped by [bin/pt2_spec_gen] when it writes node fixtures,
   because a Tensor-typed param holding an [Argument.Int] is not something an
   op-spec can express; and the walk generator synthesises tensor arguments, so
   it never produces one either.

   [Interp_verify.dispatch ~verify:true] is the dual path: it runs the node
   through [Interp_dispatch] (real ATen, which materialises the scalar with
   [full_like]) AND through [Op_bridge] + [Eval_direct] (native, which routes it
   into an op parameter), then compares element-wise with [Verify.verify_node].
   Silence means the two agree. *)
let%expect_test "verify: add.Tensor with a serialized Int scalar" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  verify_print ~target:"torch.ops.aten.add.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3 ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: add.Tensor with a serialized Float scalar" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  verify_print ~target:"torch.ops.aten.add.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "other" 0.5 ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: div.Tensor with a serialized Int scalar" =
  let a = float_tensor [ 2; 3 ] [ 0.; 1.; 3.; 6.; 9.; 12. ] in
  verify_print ~target:"torch.ops.aten.div.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 6 ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: div.Tensor with a serialized Float scalar" =
  let a = float_tensor [ 2; 3 ] [ 0.; 1.; 3.; 6.; 9.; 12. ] in
  verify_print ~target:"torch.ops.aten.div.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "other" 0.1 ];
  [%expect {| aten and native agree |}]

(* op3-impl.md F6: [sub.Tensor]'s scalar spelling cannot be expressed as an
   [Op_spec] fixture (there is no [Arg_value] constructor for "a Tensor-typed
   slot the exporter wrote as a bare scalar"), so this is the only route to
   real ATen evidence for it. [x - s] legalizes to [x + (-s)] on the native
   side (op_bridge.ml, native_interp.ml) -- if that negation were ever wrong
   (e.g. [add_scalar s] instead of [add_scalar (-.s)]), this would print a
   mismatch rather than agreement. *)
let%expect_test "verify: sub.Tensor with a serialized Int scalar" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  verify_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3 ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: sub.Tensor with a serialized Float scalar" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  verify_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "other" 0.5 ];
  [%expect {| aten and native agree |}]

(* Broadcasting a [1,3] against the [2,3] self, then a [2,1]: two different
   axes carry the extent-1 side, checked against real ATen rather than a
   hand-derived expectation. *)
let%expect_test "verify: sub.Tensor broadcasts [1,3] and [2,1] against [2,3]" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  let row = float_tensor [ 1; 3 ] [ 1.; 2.; 3. ] in
  let col = float_tensor [ 2; 1 ] [ 10.; 20. ] in
  verify_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a); ("other", row) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ];
  verify_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a); ("other", col) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

(* [other] of a shape that cannot broadcast against [self] at all: the
   existing native [`Broadcast] row, at the native boundary -- not a new
   importer check, since [sub.Tensor] adds no shape rule of its own. *)
let%expect_test "dispatch: sub.Tensor rejects an incompatible other shape" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  let b = float_tensor [ 2; 4 ] (List.init 8 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect {| error: incompatible broadcast extents on axis C: 3 vs 4 |}]

(* The whole hardsigmoid chain, each node checked against ATen in turn: the
   scalar add, both one-sided clamps, and the scalar divide. *)
let%expect_test "verify: MobileNet-v3 hardsigmoid chain against ATen" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  verify_print ~target:"torch.ops.aten.add.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3 ];
  verify_print ~target:"torch.ops.aten.clamp.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "min" 0 ];
  verify_print ~target:"torch.ops.aten.clamp.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_none "min"; in_int "max" 6 ];
  verify_print ~target:"torch.ops.aten.div.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 6 ];
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

(* alpha is neither implemented nor silently dropped: [self + alpha * other] is
   not what the arm computes, so a non-default alpha must be refused on both the
   tensor and the scalar path, and for sub as well as add. *)
let%expect_test "dispatch: a non-default alpha is rejected, not ignored" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 2; 3 ] [ 1.; 1.; 1.; 1.; 1.; 1. ] in
  dispatch_print ~target:"torch.ops.aten.add.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other"; in_int "alpha" 2 ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.add.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3; in_float "alpha" 2.5 ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other"; in_int "alpha" 2 ]
    ~noutputs:1;
  (* alpha=1 is the default and must still go through *)
  dispatch_print ~target:"torch.ops.aten.add.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other"; in_int "alpha" 1 ]
    ~noutputs:1;
  [%expect
    {|
    error: alpha=2 is not supported (only 1)
    error: alpha=2.5 is not supported (only 1)
    error: alpha=2 is not supported (only 1)
    tensor f32 [W=2 C=3] {2, 3, 4, 5, 6, 7} |}]

(* Native's engine has exactly one physical layout per shape, so `contiguous`
   and `preserve` are always already true (a no-op, same as omitting the
   argument); `channels_last`/`channels_last_3d` request an actual different
   physical arrangement this engine cannot represent, and are refused. Every
   `clone.default` occurrence in the 100-model sweep requests either no
   format or exactly `contiguous` -- see `.ai/pt2_model_support.md`. *)
let%expect_test
    "dispatch: clone's memory_format is a no-op when it can be, rejected \
     otherwise" =
  let a = float_tensor [ 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  let clone mf =
    dispatch_print ~target:"torch.ops.aten.clone.default"
      ~bindings:[ ("self", a) ]
      ~inputs:
        ([ in_tensor "self" ]
        @
        match mf with
        | None -> []
        | Some m -> [ in_memory_format "memory_format" m ])
  in
  clone None ~noutputs:1;
  clone (Some PT.MemoryFormat.ContiguousFormat) ~noutputs:1;
  clone (Some PT.MemoryFormat.PreserveFormat) ~noutputs:1;
  clone (Some PT.MemoryFormat.ChannelsLast) ~noutputs:1;
  clone (Some PT.MemoryFormat.ChannelsLast3d) ~noutputs:1;
  [%expect
    {|
    tensor f32 [W=2 C=2] {1, 2, 3, 4}
    tensor f32 [W=2 C=2] {1, 2, 3, 4}
    tensor f32 [W=2 C=2] {1, 2, 3, 4}
    error: clone: memory_format=channels_last is not supported
    error: clone: memory_format=channels_last_3d is not supported |}]

(* A wrong-kind argument under this name is still flagged as a decode
   error, not silently accepted or misreported as an unsupported format. *)
let%expect_test
    "dispatch: clone's memory_format rejects the wrong argument kind" =
  let a = float_tensor [ 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.clone.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "memory_format" 0 ]
    ~noutputs:1;
  [%expect
    {| error: argument "memory_format": expected memory_format?, got Int |}]

(* clamp with neither bound is refused where the node is built, mirroring
   ATen's own meta-function check. *)
let%expect_test "dispatch: clamp with no bounds is rejected" =
  let a = float_tensor [ 2; 2 ] [ -1.; 0.; 3.; 9. ] in
  dispatch_print ~target:"torch.ops.aten.clamp.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_none "min"; in_none "max" ]
    ~noutputs:1;
  [%expect {| error: clamp: at least one of 'min' or 'max' must be given |}]

(* Hardtanh's bounds are schema Scalars with defaults: they may be omitted
   entirely, or arrive as Int rather than Float. *)
let%expect_test "dispatch: hardtanh bound spellings" =
  let a = float_tensor [ 2; 3 ] [ -3.; -1.; 0.; 0.5; 2.5; 7. ] in
  (* omitted -> schema defaults (-1, 1) *)
  dispatch_print ~target:"torch.ops.aten.hardtanh.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  (* Int bounds, as MobileNet-v3-style graphs spell them *)
  dispatch_print ~target:"torch.ops.aten.hardtanh.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "min_val" 0; in_int "max_val" 6 ]
    ~noutputs:1;
  (* Float bounds, as MobileNet-v2 spells them *)
  dispatch_print ~target:"torch.ops.aten.hardtanh.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "min_val" 0.; in_float "max_val" 6. ]
    ~noutputs:1;
  [%expect
    {|
    tensor f32 [W=2 C=3] {-1, -1, 0, 0.5, 1, 1}
    tensor f32 [W=2 C=3] {0, 0, 0, 0.5, 2.5, 6}
    tensor f32 [W=2 C=3] {0, 0, 0, 0.5, 2.5, 6} |}]

(* clamp_min(self, min) = clamp(self, min=min, max=None), the same
   [Pointwise.Clamp] node [clamp.default] builds -- checked against real ATen
   with both scalar spellings the schema admits. *)
let%expect_test "verify: clamp_min.default with a serialized Int min" =
  let a = float_tensor [ 2; 2 ] [ -1.; 0.; 3.; 9. ] in
  verify_print ~target:"torch.ops.aten.clamp_min.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "min" 2 ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: clamp_min.default with a serialized Float min" =
  let a = float_tensor [ 2; 2 ] [ -1.; 0.; 3.; 9. ] in
  verify_print ~target:"torch.ops.aten.clamp_min.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "min" 1.5 ];
  [%expect {| aten and native agree |}]

(* rsqrt(x) = x ** -0.5: legalizes onto the existing [Pow] node's own
   reciprocal-of-sqrt special case (see op_bridge_pointwise.ml's arm
   comment) -- checked against real ATen, which has its own dedicated
   [rsqrt_kernel] rather than routing through [pow]. *)
let%expect_test "verify: rsqrt.default against real ATen" =
  let a = float_tensor [ 2; 3 ] [ 1.; 4.; 0.25; 16.; 2.; 100. ] in
  verify_print ~target:"torch.ops.aten.rsqrt.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  [%expect {| aten and native agree |}]
