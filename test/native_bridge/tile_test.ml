(* tile.default and Aten_shape.resolve_tile_size, through the ATen-linked
   bridge. Legalizes onto the SAME [Repeat] node [repeat.default] does; see
   repeat_test.ml for that node's own compute coverage. Promote with
   [dune promote test/native_bridge/tile_test.ml]. *)

open Helpers

let dispatch_tile ~x ~dims () =
  dispatch_print ~target:"torch.ops.aten.tile.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dims" dims ]
    ~noutputs:1

(* [dims] names FEWER entries than [self]'s rank -- the corpus's own shape
   (`vit_small_patch16_dinov3_qkvb`'s [self]=[256,32], [dims]=[2]). Unlike
   [repeat.default], which REJECTS this, [tile] left-pads [dims] with 1s to
   [self]'s rank first: [dims]=[2] becomes [1,2], so only the LAST axis
   tiles. *)
let%expect_test "dispatch: tile left-pads dims shorter than self's rank" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_tile ~x ~dims:[ 2 ] ();
  [%expect {| tensor f32 [W=2 C=6] {0, 1, 2, 0, 1, 2, 3, 4, ...} |}]

(* [dims] already names as many entries as [self]'s rank -- no padding, and
   [resolve_tile_size] hands [dims] to [resolve_repeat_size] unchanged, so
   this is bit-for-bit [repeat.default]'s own "two axes, different
   multipliers" case. *)
let%expect_test
    "dispatch: tile with dims at self's own rank behaves like repeat" =
  let x = float_tensor [ 2; 2 ] [ 0.; 1.; 2.; 3. ] in
  dispatch_tile ~x ~dims:[ 1; 2 ] ();
  [%expect {| tensor f32 [W=2 C=4] {0, 1, 0, 1, 2, 3, 2, 3} |}]

(* [dims] names MORE entries than [self]'s rank -- no padding needed either;
   [resolve_repeat_size]'s own "extra leading axis" case, verbatim. *)
let%expect_test
    "dispatch: tile with dims longer than self's rank adds a leading axis" =
  let x = float_tensor [ 3 ] [ 0.; 1.; 2. ] in
  dispatch_tile ~x ~dims:[ 2; 1 ] ();
  [%expect {| tensor f32 [W=2 C=3] {0, 1, 2, 0, 1, 2} |}]
