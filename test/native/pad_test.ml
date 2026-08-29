(* pad: constant and reflect modes, asymmetric pads, negative pads (crop),
   and the configurations with no Native result. Split from compute_test.ml. *)

open Compute_fixtures

(* ---- Pad: hand-computed, one rule per test ------------------------------- *)

let pad_eval ~x_shape ~x (p : Pad.Pad.params) =
  let module P = Pad.Pad.Compute (Direct) in
  let open Err.Syntax in
  let r =
    let* out_shape = Pad.Pad.output_shape ~x_shape p in
    Err.return (out_shape, Schedule.evaluate out_shape (P.pixel p ~x_shape ~x))
  in
  Format.printf "@[<v>%a@]@."
    (pp_result (fun ppf (shape, t) -> pp_grid shape ppf t))
    r

let%expect_test "Direct: pad constant, one axis, asymmetric" =
  let x_shape, x = coord_hw 2 3 in
  (* W: 1 before, 2 after -> extent 3 + 3 = 6. Row 0 reads 0,1,2 into slots
     1..3; slots 0, 4, 5 are the fill. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads = [ (Axis.W, { Pad.Pad.before = 1; after = 2 }) ];
      mode = Pad.Pad.Constant (-7.);
    };
  [%expect {|
    [H=2 W=6 C=1]
    -7 0 1 2 -7 -7
    -7 10 11 12 -7 -7 |}]

let%expect_test "Direct: pad constant, two axes at once" =
  let x_shape, x = coord_hw 2 2 in
  (* H by (1,0) and W by (0,1): 3 rows of 3. The interior is the 2x2 source in
     the LOWER-LEFT, which distinguishes a before/after swap on either axis. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads =
        [
          (Axis.H, { Pad.Pad.before = 1; after = 0 });
          (Axis.W, { Pad.Pad.before = 0; after = 1 });
        ];
      mode = Pad.Pad.Constant 9.;
    };
  [%expect {|
    [H=3 W=3 C=1]
    9 9 9
    0 1 9
    10 11 9 |}]

let%expect_test "Direct: pad constant fill is used, not zero" =
  let x_shape, x = coord_hw 1 2 in
  (* A dropped [value] would print 0 in the pad slots. The source's own first
     element is 0 too, which is exactly why the fill must not be. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads = [ (Axis.W, { Pad.Pad.before = 1; after = 1 }) ];
      mode = Pad.Pad.Constant 0.5;
    };
  [%expect {|
    [W=4 C=1]
    0.5 0 1 0.5 |}]

let%expect_test "Direct: pad reflect mirrors about the boundary, not through it"
    =
  let x_shape, x = coord_hw 1 4 in
  (* Source 0,1,2,3. Reflect by (2,2) gives 2,1 | 0,1,2,3 | 2,1 — the boundary
     element is NOT repeated, which is what separates reflect from replicate,
     and the left block is the mirror of the LEFT end rather than a copy of the
     right one. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads = [ (Axis.W, { Pad.Pad.before = 2; after = 2 }) ];
      mode = Pad.Pad.Reflect;
    };
  [%expect {|
    [W=8 C=1]
    2 1 0 1 2 3 2 1 |}]

let%expect_test "Direct: pad reflect, two axes, asymmetric" =
  let x_shape, x = coord_hw 3 3 in
  (* H by (1,0), W by (0,2): the top row mirrors row 1 (values 10..12) and each
     row's tail mirrors its own columns 1 and 0. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads =
        [
          (Axis.H, { Pad.Pad.before = 1; after = 0 });
          (Axis.W, { Pad.Pad.before = 0; after = 2 });
        ];
      mode = Pad.Pad.Reflect;
    };
  [%expect
    {|
    [H=4 W=5 C=1]
    10 11 12 11 10
    0 1 2 1 0
    10 11 12 11 10
    20 21 22 21 20 |}]

let%expect_test "Direct: negative pads crop" =
  let x_shape, x = coord_hw 3 4 in
  (* Crop one column from each side of W and the first row of H: a 2x2 window
     starting at (1,1). *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads =
        [
          (Axis.H, { Pad.Pad.before = -1; after = 0 });
          (Axis.W, { Pad.Pad.before = -1; after = -1 });
        ];
      mode = Pad.Pad.Constant 0.;
    };
  [%expect {|
    [H=2 W=2 C=1]
    11 12
    21 22 |}]

let%expect_test "Direct: pad and crop on different axes of one node" =
  let x_shape, x = coord_hw 2 3 in
  (* H padded by (1,0), W cropped by (-1,0): 3 rows of 2. *)
  pad_eval ~x_shape ~x
    {
      Pad.Pad.pads =
        [
          (Axis.H, { Pad.Pad.before = 1; after = 0 });
          (Axis.W, { Pad.Pad.before = -1; after = 0 });
        ];
      mode = Pad.Pad.Constant 5.;
    };
  [%expect {|
    [H=3 W=2 C=1]
    5 5
    1 2
    11 12 |}]

let%expect_test "Direct: pad rejects the configurations with no Native result" =
  let x_shape, _ = coord_hw 3 4 in
  let refuse (p : Pad.Pad.params) =
    Format.printf "%a@." (pp_result Vec6.pp_shape)
      (Pad.Pad.output_shape ~x_shape p)
  in
  (* A crop that consumes the axis: legal in ATen (size-0), unrepresentable
     here. *)
  refuse
    {
      Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = -2; after = -1 }) ];
      mode = Pad.Pad.Constant 0.;
    };
  refuse
    {
      Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = -5; after = 0 }) ];
      mode = Pad.Pad.Constant 0.;
    };
  (* Reflect needs each POSITIVE side below the extent it mirrors: H is 3, so 3
     is already too wide while 2 is fine. *)
  refuse
    {
      Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = 3; after = 0 }) ];
      mode = Pad.Pad.Reflect;
    };
  refuse
    {
      Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = 0; after = 3 }) ];
      mode = Pad.Pad.Reflect;
    };
  (* The same widths are ordinary in constant mode — the rule is reflect's, not
     padding's. *)
  refuse
    {
      Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = 3; after = 3 }) ];
      mode = Pad.Pad.Constant 0.;
    };
  (* Two entries for one axis: unreachable from either importer, so this is the
     guard on the builder and on JSON decoding. *)
  refuse
    {
      Pad.Pad.pads =
        [
          (Axis.W, { Pad.Pad.before = 1; after = 1 });
          (Axis.W, { Pad.Pad.before = 2; after = 2 });
        ];
      mode = Pad.Pad.Constant 0.;
    };
  [%expect
    {|
    pad of axis H by (-2, -1) over extent 3 leaves 0 elements; the engine has no empty extent
    pad of axis H by (-5, 0) over extent 3 leaves -2 elements; the engine has no empty extent
    reflect pad of axis H by (3, 0) needs each side below the extent 3
    reflect pad of axis H by (0, 3) needs each side below the extent 3
    [H=9 W=4 C=1]
    axis W has more than one pad entry |}]
