(* repeat_interleave.self_int, through the ATen-linked bridge. Promote with
   [dune promote test/native_bridge/repeat_interleave_test.ml]. *)

open Helpers

let dispatch_repeat_interleave ?dim ~x ~repeats () =
  dispatch_print ~target:"torch.ops.aten.repeat_interleave.self_int"
    ~bindings:[ ("self", x) ]
    ~inputs:
      (in_tensor "self" :: in_int "repeats" repeats
      :: (match dim with Some d -> [ in_int "dim" d ] | None -> []))
    ~noutputs:1

(* CONTIGUOUS duplication -- each source element repeated in place -- unlike
   [repeat.default]'s wraparound tiling. *)
let%expect_test "dispatch: repeat_interleave duplicates each element in place" =
  let x = float_tensor [ 3 ] [ 0.; 1.; 2. ] in
  dispatch_repeat_interleave ~x ~repeats:2 ~dim:0 ();
  [%expect {| tensor f32 [C=6] {0, 0, 1, 1, 2, 2} |}]

(* Two axes' worth of a rank-2 tensor, so a bridge that swapped [dim] and the
   axis it converts to would show a transposed rather than duplicated
   result. *)
let%expect_test
    "dispatch: repeat_interleave on the outer axis of a rank-2 tensor" =
  let x = float_tensor [ 2; 2 ] [ 0.; 1.; 2.; 3. ] in
  dispatch_repeat_interleave ~x ~repeats:2 ~dim:0 ();
  [%expect {| tensor f32 [W=4 C=2] {0, 1, 0, 1, 2, 3, 2, 3} |}]

let%expect_test "dispatch: repeat_interleave by 1 is the identity" =
  let x = float_tensor [ 3 ] [ 0.; 1.; 2. ] in
  dispatch_repeat_interleave ~x ~repeats:1 ~dim:0 ();
  [%expect {| tensor f32 [C=3] {0, 1, 2} |}]

(* The [dim=None] flatten-first form is out of scope for this landing -- see
   [Repeat.RepeatInterleave]'s own comment. *)
let%expect_test "dispatch: repeat_interleave rejects an absent dim" =
  let x = float_tensor [ 3 ] [ 0.; 1.; 2. ] in
  dispatch_repeat_interleave ~x ~repeats:2 ();
  [%expect
    {| error: repeat_interleave.self_int: dim=None (flatten-first) is not yet supported |}]

(* [Op_config.Bad.pos]'s own refusal, shared with every other op that takes a
   positive-int parameter: [repeats] must be at least 1, the same "the
   engine has no empty tensors" invariant [Dim.extent] enforces elsewhere. *)
let%expect_test "dispatch: repeat_interleave rejects a non-positive repeats" =
  let x = float_tensor [ 3 ] [ 0.; 1.; 2. ] in
  dispatch_repeat_interleave ~x ~repeats:0 ~dim:0 ();
  [%expect
    {| error: repeat_interleave.self_int: repeats must be positive, got 0 |}]
