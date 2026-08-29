(* slice and select: axis narrowing/dropping and their out-of-range/empty-
   extent rejections. Split from compute_test.ml. *)

open Compute_fixtures

(* ---- Slice ----------------------------------------------------------------

   Four mutations were applied to [Split.Slice] and observed failing here before
   being reverted, which is what makes these goldens evidence rather than
   description:

   - floor instead of the ceiling in [output_shape]: every strided extent drops
     by one, in this file and in graph_json_test.ml;
   - the step multiplication dropped from [Compute.pixel]: [0 2 4] becomes
     [0 1 2];
   - [start] replaced by 0: the selected window slides to the origin on both
     axes;
   - the upper range bound weakened from [stop <= extent]: an out-of-range slice
     produces a shape instead of the typed refusal, and [Compute]'s read goes
     out of bounds behind it.

   A fifth -- the WRONG AXIS -- is refuted by the two tests that run the same
   bounds on H and on W over a non-square fixture, which is why that pair exists
   rather than one test. *)

let slice_eval ~x_shape ~x (p : Split.Slice.params) =
  let module Sl = Split.Slice.Compute (Direct) in
  let open Err.Syntax in
  let r =
    let* out_shape = Split.Slice.output_shape ~x_shape p in
    Err.return (out_shape, Schedule.evaluate out_shape (Sl.pixel p ~x))
  in
  Format.printf "@[<v>%a@]@."
    (pp_result (fun ppf (shape, t) -> pp_grid shape ppf t))
    r

let pos = Op_config.Pos.of_int

let%expect_test "Direct: slice keeps the axis and narrows it" =
  let x_shape, x = coord_hw 3 4 in
  (* Columns 1 and 2 of every row. The values are 10*row + col, so a wrong start
     shows as a column shift and a wrong axis as a row selection. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.W; start = 1; stop = 3; step = pos 1 };
  [%expect {|
    [H=3 W=2 C=1]
    1 2
    11 12
    21 22 |}]

let%expect_test "Direct: slice on the other axis selects rows" =
  let x_shape, x = coord_hw 3 4 in
  (* The same bounds on H. Distinguishable from the W case only because the
     fixture is 3x4 and the values encode both coordinates. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.H; start = 1; stop = 3; step = pos 1 };
  [%expect {|
    [H=2 W=4 C=1]
    10 11 12 13
    20 21 22 23 |}]

let%expect_test "Direct: slice step selects every k-th element" =
  let x_shape, x = coord_hw 1 6 in
  (* [0,6) step 2 -> columns 0,2,4. An implementation that forgot the step
     multiplication would print 0 1 2. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.W; start = 0; stop = 6; step = pos 2 };
  (* [1,6) step 2 -> columns 1,3,5: the span is 5, so the CEILING is what makes
     the count 3 rather than 2. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.W; start = 1; stop = 6; step = pos 2 };
  (* [0,5) step 3 -> columns 0,3. Span 5 over step 3 is 1.67, and both a floor
     and a truncation would print one column. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.W; start = 0; stop = 5; step = pos 3 };
  [%expect
    {|
    [W=3 C=1]
    0 2 4

    [W=3 C=1]
    1 3 5

    [W=2 C=1]
    0 3 |}]

let%expect_test "Direct: slice of the whole axis is the identity" =
  let x_shape, x = coord_hw 2 2 in
  (* The configuration a Default-tier walk would generate, and the reason
     slice.Tensor needs a walk_meta entry rather than the generated default:
     every implementation that returns its input passes this one. *)
  slice_eval ~x_shape ~x
    { Split.Slice.axis = Axis.W; start = 0; stop = 2; step = pos 1 };
  [%expect {|
    [H=2 W=2 C=1]
    0 1
    10 11 |}]

let%expect_test "Slice: the configurations with no Native result" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:4 ~c:1 in
  let refuse (p : Split.Slice.params) =
    Format.printf "%a@." (pp_result Vec6.pp_shape)
      (Split.Slice.output_shape ~x_shape p)
  in
  (* Empty: legal in ATen, which returns a size-0 tensor, and unrepresentable
     here. Both the degenerate [start = stop] and a step wide enough to skip
     everything are the same fault -- the second cannot happen, since a
     non-empty span always yields at least one element under the ceiling, and
     the case is written out to record that rather than leave it implied. *)
  refuse { Split.Slice.axis = Axis.W; start = 2; stop = 2; step = pos 1 };
  refuse { Split.Slice.axis = Axis.W; start = 3; stop = 4; step = pos 9 };
  (* Out of range. Unreachable from either importer -- both build their bounds
     with [Aten_shape.resolve_slice], which clamps -- so these guard the builder
     and JSON decoding, and they are what keeps [Compute]'s read in bounds. *)
  refuse { Split.Slice.axis = Axis.W; start = 0; stop = 5; step = pos 1 };
  refuse { Split.Slice.axis = Axis.W; start = 3; stop = 1; step = pos 1 };
  refuse { Split.Slice.axis = Axis.W; start = -1; stop = 2; step = pos 1 };
  [%expect
    {|
    slice of axis W [2, 2) step 1 over extent 4 selects 0 elements; the engine has no empty extent
    [H=3 W=1 C=1]
    slice of axis W [0, 5) step 1 over extent 4 is not within 0 <= start <= stop <= extent
    slice of axis W [3, 1) step 1 over extent 4 is not within 0 <= start <= stop <= extent
    slice of axis W [-1, 2) step 1 over extent 4 is not within 0 <= start <= stop <= extent |}]

(* ---- Select ----------------------------------------------------------------

   [Select] DROPS the axis it picks along, unlike [Slice] which keeps the
   same rank -- so unlike [slice_eval]'s tests above, this must also confirm
   the surviving axes repack right-aligned exactly as [Unbind]'s do (it
   reuses the same [Aten_shape.repack_dropped] pairing). Value at (h,w,c) is
   h*100 + w*10 + c, so a wrong index reads a different row/column and a
   wrong axis reads the wrong pair of survivors. *)
let select_eval ~x_shape ~x (p : Split.Select.params) =
  let module Se = Split.Select.Compute (Direct) in
  eval_tensor (Split.Select.output_shape ~x_shape p) (Se.pixel p ~x)

let%expect_test "Direct: select along an outer, a middle and an inner axis" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        float_of_int ((row c * 100) + (col c * 10) + chan c))
  in
  let run axis index =
    let p = { Split.Select.axis; index } in
    Format.printf "select %a=%d -> %a@." Axis.pp axis index
      (pp_result Tensor.pp)
      (select_eval ~x_shape ~x p)
  in
  run Axis.H 1;
  run Axis.W 2;
  run Axis.C 1;
  [%expect
    {|
    select H=1 -> tensor f32 [W=3 C=2] {100, 101, 110, 111, 120, 121}
    select W=2 -> tensor f32 [W=2 C=2] {20, 21, 120, 121}
    select C=1 -> tensor f32 [W=2 C=3] {1, 11, 21, 101, 111, 121} |}]

let%expect_test "Select: an out-of-range index has no Native result" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4 in
  let refuse index =
    Format.printf "%a@." (pp_result Vec6.pp_shape)
      (Split.Select.output_shape ~x_shape { Split.Select.axis = Axis.C; index })
  in
  refuse 4;
  refuse (-1);
  [%expect
    {|
    slice of axis C [4, 5) step 1 over extent 4 is not within 0 <= start <= stop <= extent
    slice of axis C [-1, 0) step 1 over extent 4 is not within 0 <= start <= stop <= extent |}]
