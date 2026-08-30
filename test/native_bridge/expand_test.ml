(* expand.default and Aten_shape.resolve_expand_size, through the ATen-linked
   bridge. Promote with [dune promote test/native_bridge/expand_test.ml]. *)

open Helpers

let dispatch_expand ?(implicit = false) ~x ~size () =
  dispatch_print ~target:"torch.ops.aten.expand.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [ in_tensor "self"; in_ints "size" size; in_bool "implicit" implicit ]
    ~noutputs:1

(* [implicit] carries no computational effect (ATen's own [at::native::expand]
   never reads it, TensorShape.cpp) -- both spellings must build the identical
   result. *)
let%expect_test "dispatch: expand broadcasts a size-1 axis, implicit ignored" =
  let x = float_tensor [ 1; 3 ] [ 0.; 1.; 2. ] in
  dispatch_expand ~x ~size:[ 2; 3 ] ~implicit:false ();
  dispatch_expand ~x ~size:[ 2; 3 ] ~implicit:true ();
  [%expect
    {|
    tensor f32 [W=2 C=3] {0, 1, 2, 0, 1, 2}
    tensor f32 [W=2 C=3] {0, 1, 2, 0, 1, 2} |}]

(* [-1] stands for "keep self's own extent here" -- a non-broadcast axis, left
   alone. *)
let%expect_test "dispatch: expand's -1 keeps self's own extent" =
  let x = float_tensor [ 1; 3 ] [ 0.; 1.; 2. ] in
  dispatch_expand ~x ~size:[ 2; -1 ] ();
  [%expect {| tensor f32 [W=2 C=3] {0, 1, 2, 0, 1, 2} |}]

(* [size] naming one more entry than [self]'s own rank ADDS a leading axis --
   the one shape [expand] alone can produce and [view] cannot. *)
let%expect_test "dispatch: expand adds a leading axis" =
  let x = float_tensor [ 3 ] [ 0.; 1.; 2. ] in
  dispatch_expand ~x ~size:[ 2; 3 ] ();
  [%expect {| tensor f32 [W=2 C=3] {0, 1, 2, 0, 1, 2} |}]

(* ATen's own [TORCH_CHECK]: [size] must name at least as many entries as
   [self]'s own rank -- expand never REDUCES rank. *)
let%expect_test "dispatch: expand rejects a size shorter than self's rank" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_expand ~x ~size:[ 3 ] ();
  [%expect
    {| error: expand size [3] must have at least as many entries as self's shape [2, 3] (rank 2) |}]

(* ATen's other own [TORCH_CHECK]: a [-1] naming a leading position beyond
   [self]'s rank has nothing to copy. *)
let%expect_test "dispatch: expand rejects a leading -1" =
  let x = float_tensor [ 3 ] [ 0.; 1.; 2. ] in
  dispatch_expand ~x ~size:[ -1; 3 ] ();
  [%expect
    {| error: expand size [-1, 3]: -1 at position 0 is not allowed for a leading dimension self (shape [3]) does not have |}]

(* Not [Aten_shape]'s business -- [self]'s axis is 2, not 1, so it cannot
   broadcast to 5. [Pointwise.Expand.output_shape]'s own axis-wise check
   catches it, the general graph invariant reachable from a raw
   [Graph_builder] call or a JSON-decoded graph too, not just this
   resolver. *)
let%expect_test "dispatch: expand rejects a non-broadcastable target" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_expand ~x ~size:[ 5; 3 ] ();
  [%expect {| error: incompatible broadcast extents on axis W: 2 vs 5 |}]
