(* ATen<->Native tensor conversion and payload comparison. Split from the
   former native_bridge_test.ml; promote with
   [dune promote test/native_bridge/conversion_test.ml]. *)

open Helpers

(* ---- of_aten: ATen -> Native conversion --------------------------------- *)

let%expect_test "of_aten: float32 round-trip preserves values" =
  let t = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  Format.printf "%a@."
    (pp_of_aten_result ~ok:Tensor.pp)
    (Tensor_bridge.of_aten t);
  [%expect {| tensor f32 [W=2 C=3] {1, 2, 3, 4, 5, 6} |}]

let%expect_test "of_aten: int64 round-trip preserves values" =
  let t = i64_tensor [ 4 ] [ 10L; 20L; 30L; 40L ] in
  Format.printf "%a@."
    (pp_of_aten_result ~ok:Tensor.pp)
    (Tensor_bridge.of_aten t);
  [%expect {| tensor i64 [C=4] {10, 20, 30, 40} |}]

let%expect_test "of_aten: float64 dtype is supported" =
  let t = double_tensor [ 2 ] [ 1.5; -2.25 ] in
  Format.printf "%a@."
    (pp_of_aten_result ~ok:Tensor.pp)
    (Tensor_bridge.of_aten t);
  [%expect {| tensor f64 [C=2] {1.5, -2.25} |}]

let%expect_test "of_aten: shape is right-aligned to 6D" =
  let t = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  Format.printf "%a@."
    (pp_of_aten_result ~ok:(fun ppf (Tensor.Tensor r) ->
         Fmt.pf ppf "shape=%a" Vec6.pp_shape r.shape))
    (Tensor_bridge.of_aten t);
  [%expect {| shape=[W=3 C=4] |}]

(* A non-contiguous ATen tensor (here a permute view) is materialized contiguous
   before conversion, so the native tensor holds the logical (transposed) order.
   This is what lets addmm's transposed fc weight convert on the real graph. *)
let%expect_test "of_aten: materializes a non-contiguous permute view" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  let aten =
    Aten_tensor.manage
      (Aten_c.Aten_operations.permute x (Interp_decode.arr [ 1; 0 ]) 2)
  in
  Format.printf "%a@."
    (pp_of_aten_result ~ok:Tensor.pp)
    (Tensor_bridge.of_aten aten);
  [%expect {| tensor f32 [W=3 C=2] {0, 3, 1, 4, 2, 5} |}]

(* The permute case above cannot reach this one: a [select] view is CONTIGUOUS,
   so the old [is_contiguous]-only guard passed it through untouched and
   [of_aten] read from the storage base — row 0's values, labelled row 1. Index
   1 is essential; index 0 has offset 0 and hides the bug. *)
let%expect_test
    "of_aten: materializes a contiguous select view at a non-zero offset" =
  let x = float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  let row1 = Aten_tensor.manage (Aten_c.Aten_operations.select_int x 0L 1L) in
  Format.printf "%a@."
    (pp_of_aten_result ~ok:Tensor.pp)
    (Tensor_bridge.of_aten row1);
  [%expect {| tensor f32 [C=2] {2, 3} |}]

(* ---- compare_tensors: shape/type/payload checks ------------------------- *)

let%expect_test "compare_tensors: matching F32 tensors -> Ok" =
  let aten = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let native = native_f32 [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-6 ~output:"y" aten native);
  [%expect {| Ok |}]

let%expect_test "compare_tensors: F32 within atol -> Ok" =
  let aten = float_tensor [ 3 ] [ 1.; 2.; 3. ] in
  let native = native_f32 [ 3 ] [ 1.; 2.; 3.000005 ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-5 ~output:"y" aten native);
  [%expect {| Ok |}]

let%expect_test "compare_tensors: F32 mismatch reports coordinates" =
  let aten = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let native = native_f32 [ 2; 3 ] [ 1.; 2.; 99.; 4.; 5.; 6. ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-6 ~output:"y" aten native);
  [%expect
    {|
    Error: y: payload mismatch (1/1 shown):
    (2) aten=3 native=99 |}]

(* NaN-ness is compared as an exact property before the tolerance test. The
   tolerance test alone cannot do it: [nan -. x] is [nan] and [nan > atol] is
   false, so every NaN disagreement used to read as agreement — which would make
   any NaN test built on this comparator vacuous. Both directions are covered
   because they are separate bugs: native losing a NaN ATen produced, and native
   inventing one ATen did not. *)
let%expect_test "compare_tensors: native NaN where ATen is finite -> Error" =
  let aten = float_tensor [ 3 ] [ 1.; 2.; 3. ] in
  let native = native_f32 [ 3 ] [ 1.; Float.nan; 3. ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-6 ~output:"y" aten native);
  [%expect
    {|
    Error: y: payload mismatch (1/1 shown):
    (1) aten=2 native=nan |}]

let%expect_test "compare_tensors: ATen NaN where native is finite -> Error" =
  let aten = float_tensor [ 3 ] [ 1.; Float.nan; 3. ] in
  let native = native_f32 [ 3 ] [ 1.; 2.; 3. ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-6 ~output:"y" aten native);
  [%expect
    {|
    Error: y: payload mismatch (1/1 shown):
    (1) aten=nan native=2 |}]

(* Both NaN is agreement: ATen produces NaN deliberately in cases the native
   side must reproduce (a clamp with a NaN bound fills its whole result), and
   the engine draws no distinction between NaN bit patterns. *)
let%expect_test "compare_tensors: NaN on both sides -> Ok" =
  let aten = float_tensor [ 3 ] [ 1.; Float.nan; 3. ] in
  let native = native_f32 [ 3 ] [ 1.; Float.nan; 3. ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-6 ~output:"y" aten native);
  [%expect {| Ok |}]

(* Infinities compare by equality for the same reason: [inf -. inf] is [nan], so
   the tolerance test would have passed two infinities of opposite sign. *)
let%expect_test "compare_tensors: opposite-sign infinities -> Error" =
  let aten = float_tensor [ 2 ] [ Float.infinity; 1. ] in
  let native = native_f32 [ 2 ] [ Float.neg_infinity; 1. ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-6 ~output:"y" aten native);
  [%expect
    {|
    Error: y: payload mismatch (1/1 shown):
    (0) aten=inf native=-inf |}]

let%expect_test "compare_tensors: same-sign infinities -> Ok" =
  let aten = float_tensor [ 2 ] [ Float.infinity; 1. ] in
  let native = native_f32 [ 2 ] [ Float.infinity; 1. ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-6 ~output:"y" aten native);
  [%expect {| Ok |}]

let%expect_test "compare_tensors: shape mismatch" =
  let aten = float_tensor [ 2; 3 ] (List.init 6 float_of_int) in
  let native = native_f32 [ 3; 2 ] (List.init 6 float_of_int) in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-6 ~output:"y" aten native);
  [%expect {| Error: y: shape mismatch: aten [2x3] native [W=3 C=2] |}]

let%expect_test "compare_tensors: dtype mismatch" =
  let aten = i64_tensor [ 3 ] [ 1L; 2L; 3L ] in
  let native = native_f32 [ 3 ] [ 1.; 2.; 3. ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-6 ~output:"y" aten native);
  [%expect {| Error: y: type mismatch: aten int64 native f32 |}]

let%expect_test "compare_tensors: materializes non-contiguous permute view" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  let aten =
    Aten_tensor.manage
      (Aten_c.Aten_operations.permute x (Interp_decode.arr [ 1; 0 ]) 2)
  in
  let native = native_f32 [ 3; 2 ] [ 0.; 3.; 1.; 4.; 2.; 5. ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-6 ~output:"y" aten native);
  [%expect {| Ok |}]

(* [Verify.logical_tensor] had the same is_contiguous-only guard as [of_aten],
   so verification compared a select view against row 0 and would report a
   mismatch on a correct native impl (or, worse, agree with a wrong one). *)
let%expect_test
    "compare_tensors: materializes contiguous select view at a non-zero offset"
    =
  let x = float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  let aten = Aten_tensor.manage (Aten_c.Aten_operations.select_int x 0L 1L) in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:1e-6 ~output:"y" aten
       (native_f32 [ 2 ] [ 2.; 3. ]));
  [%expect {| Ok |}]

let%expect_test "compare_tensors: matching I64 -> Ok" =
  let aten = i64_tensor [ 4 ] [ 10L; 20L; 30L; 40L ] in
  let native = native_i64 [ 4 ] [ 10L; 20L; 30L; 40L ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:0. ~output:"y" aten native);
  [%expect {| Ok |}]

let%expect_test "compare_tensors: I64 exact mismatch reports coordinates" =
  let aten = i64_tensor [ 3 ] [ 1L; 999L; 3L ] in
  let native = native_i64 [ 3 ] [ 1L; 2L; 3L ] in
  Format.printf "%a@." pp_result
    (Verify.compare_tensors ~atol:0. ~output:"y" aten native);
  [%expect
    {|
    Error: y: payload mismatch (1/1 shown):
    (1) aten=999 native=2 |}]

(* ---- aten_op_config: pretty printing ------------------------------------ *)

let%expect_test "pp: add.Tensor config" =
  Aten_op_config.find "torch.ops.aten.add.Tensor"
  |> Format.printf "%a@."
       (Core.Pretty.option_or ~none:"not found" Aten_op_config.pp);
  [%expect
    {|
    torch.ops.aten.add.Tensor (Tensor self, Tensor other, Scalar alpha=1) -> T |}]

let%expect_test "pp: relu.default config" =
  Aten_op_config.find "torch.ops.aten.relu.default"
  |> Format.printf "%a@."
       (Core.Pretty.option_or ~none:"not found" Aten_op_config.pp);
  [%expect {| torch.ops.aten.relu.default (Tensor self) -> T |}]
