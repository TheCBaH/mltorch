(* repeat.default and Aten_shape.resolve_repeat_size, through the ATen-linked
   bridge. Promote with [dune promote test/native_bridge/repeat_test.ml]. *)

open Helpers

let dispatch_repeat ~x ~repeats () =
  dispatch_print ~target:"torch.ops.aten.repeat.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "repeats" repeats ]
    ~noutputs:1

(* Unlike [expand], which can only fan a size-1 axis out, [repeat] tiles an
   axis of ANY extent -- self already has extent 3 on this axis, and the
   result wraps it twice rather than broadcasting a single value. *)
let%expect_test "dispatch: repeat tiles a non-broadcastable axis" =
  let x = float_tensor [ 3 ] [ 0.; 1.; 2. ] in
  dispatch_repeat ~x ~repeats:[ 2 ] ();
  [%expect {| tensor f32 [C=6] {0, 1, 2, 0, 1, 2} |}]

(* Two axes at once, with different multipliers -- catches a bridge that
   mixed up which repeats entry belongs to which axis. *)
let%expect_test "dispatch: repeat, two axes with different multipliers" =
  let x = float_tensor [ 2; 2 ] [ 0.; 1.; 2.; 3. ] in
  dispatch_repeat ~x ~repeats:[ 1; 2 ] ();
  [%expect {| tensor f32 [W=2 C=4] {0, 1, 0, 1, 2, 3, 2, 3} |}]

(* [repeats] naming one more entry than [self]'s own rank ADDS a leading
   axis -- self is implicitly viewed with a size-1 axis prepended, which
   [repeats] then tiles like any other. *)
let%expect_test "dispatch: repeat adds a leading axis" =
  let x = float_tensor [ 3 ] [ 0.; 1.; 2. ] in
  dispatch_repeat ~x ~repeats:[ 2; 1 ] ();
  [%expect {| tensor f32 [W=2 C=3] {0, 1, 2, 0, 1, 2} |}]

(* ATen's own [TORCH_CHECK]: [repeats] must name at least as many entries as
   [self]'s own rank -- repeat never REDUCES rank. *)
let%expect_test "dispatch: repeat rejects repeats shorter than self's rank" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_repeat ~x ~repeats:[ 2 ] ();
  [%expect
    {| error: repeat repeats [2] must have at least as many entries as self's shape [2, 3] (rank 2) |}]
