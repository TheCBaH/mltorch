open Loop_ir
open Loop_fixtures

let t4 a =
  f32_tensor (shape_w 4) (fun c -> a.((Vec6.offset (shape_w 4) c :> int)))

let show a r =
  Fmt.pr "%a@." Loop_numeric_diff.pp
    (Loop_numeric_diff.compare ~atol:1e-6 ~rtol:1e-6 ~actual:(t4 a)
       ~reference:(t4 r))

let%expect_test "identical, one ulp, and far" =
  let r = [| 1.; 2.; 0.; -3. |] in
  show r r;
  show
    [| Int32.float_of_bits (Int32.succ (Int32.bits_of_float 1.)); 2.; 0.; -3. |]
    r;
  show [| 1.5; 2.; 0.; -3. |] r;
  [%expect
    {|
    4 cells, 0 outside tolerance; max abs 0, max rel 0, max normalized 0; nonfinite 0 actual / 0 reference / 0 mismatched; ulp 0:4 1:0 2-4:0 5-16:0 17-256:0 >256:0
    4 cells, 0 outside tolerance; max abs 1.19e-07, max rel 1.19e-07, max normalized 3.97e-08; nonfinite 0 actual / 0 reference / 0 mismatched; ulp 0:3 1:1 2-4:0 5-16:0 17-256:0 >256:0
    4 cells, 1 outside tolerance; max abs 0.5, max rel 0.5, max normalized 0.167; nonfinite 0 actual / 0 reference / 0 mismatched; ulp 0:3 1:0 2-4:0 5-16:0 17-256:0 >256:1 |}]

let%expect_test "nonfinite cells are matched by kind, never by subtraction" =
  let r = [| nan; infinity; neg_infinity; 1. |] in
  show r r;
  show [| nan; infinity; infinity; 1. |] r;
  show [| 0.; infinity; neg_infinity; nan |] r;
  [%expect
    {|
    4 cells, 0 outside tolerance; max abs 0, max rel 0, max normalized 0; nonfinite 3 actual / 3 reference / 0 mismatched; ulp 0:1 1:0 2-4:0 5-16:0 17-256:0 >256:0
    4 cells, 1 outside tolerance; max abs 0, max rel 0, max normalized 0; nonfinite 3 actual / 3 reference / 1 mismatched; ulp 0:1 1:0 2-4:0 5-16:0 17-256:0 >256:0
    4 cells, 2 outside tolerance; max abs 0, max rel 0, max normalized 0; nonfinite 3 actual / 3 reference / 2 mismatched; ulp 0:0 1:0 2-4:0 5-16:0 17-256:0 >256:0 |}]

let%expect_test "signed zero and values across zero are one ulp apart" =
  show [| -0.; 1e-45; 0.; 0. |] [| 0.; 0.; 1e-45; -0. |];
  [%expect
    {| 4 cells, 0 outside tolerance; max abs 1.4e-45, max rel 1, max normalized 1; nonfinite 0 actual / 0 reference / 0 mismatched; ulp 0:2 1:2 2-4:0 5-16:0 17-256:0 >256:0 |}]
