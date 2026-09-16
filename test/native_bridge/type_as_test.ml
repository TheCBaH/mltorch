(* type_as.default, through the ATen-linked bridge. Legalizes onto the SAME
   [Pointwise.To_copy] node [_to_copy.default] does -- the target comes from
   [other]'s own runtime dtype instead of an explicit scalar argument.
   Promote with [dune promote test/native_bridge/type_as_test.ml]. *)

open Helpers

let dispatch_type_as ~self ~other () =
  dispatch_print ~target:"torch.ops.aten.type_as.default"
    ~bindings:[ ("self", self); ("other", other) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1

(* The corpus's own shape: [other] is float, same as [self] -- an identity,
   matching [To_copy.Float]'s own compute. *)
let%expect_test "dispatch: type_as float-to-float is the identity" =
  let self = float_tensor [ 3 ] [ 1.5; -2.25; 0. ] in
  let other = float_tensor [ 2 ] [ 0.; 0. ] in
  dispatch_type_as ~self ~other ();
  [%expect {| tensor f32 [C=3] {1.5, -2.25, 0} |}]

(* [other] is int64 -- casts to [To_copy.Long] (truncation toward zero),
   proving the target is read from [other]'s dtype, not defaulted. *)
let%expect_test "dispatch: type_as float-to-long truncates toward zero" =
  let self = float_tensor [ 3 ] [ 1.5; -2.25; 0. ] in
  let other = i64_tensor [ 1 ] [ 0L ] in
  dispatch_type_as ~self ~other ();
  [%expect {| tensor i64 [C=3] {1, -2, 0} |}]

(* No [verify_print] differential fixture for this node: confirmed
   [Interp_verify.dispatch] has no generated arm at all for
   [torch.ops.aten.type_as.default] ("unhandled op"), unrelated to the
   [ScalarType]-decoding restriction [to_copy_test.ml]'s own comment
   documents for [_to_copy.default]'s Long/Bool cases -- adding one is
   outside this cast-direction fix's scope. *)
