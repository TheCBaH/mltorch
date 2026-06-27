(* Tests for ATen<->Native tensor conversion and element-wise verification.
   Uses %expect_test blocks; promote with [dune promote test/native_bridge_test.ml]. *)

open Bigarray
module T = Aten_tensor
module D = Aten_dtype
module Stype = Aten_scalar_type

(* ---- helpers ------------------------------------------------------------ *)

let float_tensor shape vals =
  let t = T.create shape in
  let v = Option.get (T.data D.float32 t) in
  List.iteri (fun i x -> v.{i} <- x) vals;
  t

let i64_tensor shape vals =
  let t = T.create ~dtype:Stype.Long shape in
  let v = Option.get (T.data D.int64 t) in
  List.iteri (fun i x -> v.{i} <- x) vals;
  t

let native_f32 shape vals =
  let n = List.fold_left ( * ) 1 shape in
  let data = Array1.create float32 c_layout n in
  List.iteri (fun i x -> data.{i} <- x) vals;
  let shape6 = Result.get_ok (Aten_shape.of_aten (Array.of_list shape)) in
  Tensor.Tensor
    {
      shape = shape6;
      payload = { Payload.fmt = Payload.F32; quant = Payload.No_quant; data };
    }

let native_i64 shape vals =
  let n = List.fold_left ( * ) 1 shape in
  let data = Array1.create int64 c_layout n in
  List.iteri (fun i x -> data.{i} <- x) vals;
  let shape6 = Result.get_ok (Aten_shape.of_aten (Array.of_list shape)) in
  Tensor.Tensor
    {
      shape = shape6;
      payload = { Payload.fmt = Payload.I64; quant = Payload.No_quant; data };
    }

let pp_result =
  Fmt.result
    ~ok:(fun ppf () -> Fmt.string ppf "Ok")
    ~error:(fun ppf e ->
      Fmt.pf ppf "Error: %a" Verify.pp_error e.Core.Error.kind)

(* ---- of_aten: ATen -> Native conversion --------------------------------- *)

let%expect_test "of_aten: float32 round-trip preserves values" =
  let t = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  (match Tensor_bridge.of_aten t with
  | Error msg -> Printf.printf "Error: %s\n" msg
  | Ok native -> Format.printf "%a@." Tensor.pp native);
  [%expect {| tensor f32 [W=2 C=3] {1, 2, 3, 4, 5, 6} |}]

let%expect_test "of_aten: int64 round-trip preserves values" =
  let t = i64_tensor [ 4 ] [ 10L; 20L; 30L; 40L ] in
  (match Tensor_bridge.of_aten t with
  | Error msg -> Printf.printf "Error: %s\n" msg
  | Ok native -> Format.printf "%a@." Tensor.pp native);
  [%expect {| tensor i64 [C=4] {10, 20, 30, 40} |}]

let%expect_test "of_aten: unsupported dtype returns Error" =
  let t = T.create ~dtype:Stype.Double [ 2 ] in
  (match Tensor_bridge.of_aten t with
  | Error msg -> Printf.printf "Error: %s\n" msg
  | Ok _ -> print_string "unexpected Ok");
  [%expect {| Error: unsupported ATen dtype (code 7) |}]

let%expect_test "of_aten: shape is right-aligned to 6D" =
  let t = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  (match Tensor_bridge.of_aten t with
  | Error msg -> Printf.printf "Error: %s\n" msg
  | Ok (Tensor.Tensor r) -> Format.printf "shape=%a@." Vec6.pp_shape r.shape);
  [%expect {| shape=[W=3 C=4] |}]

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
  (match Aten_op_config.find "torch.ops.aten.add.Tensor" with
  | None -> print_string "not found"
  | Some c -> Format.printf "%a@." Aten_op_config.pp c);
  [%expect
    {|
    torch.ops.aten.add.Tensor (Tensor self, Tensor other, Scalar alpha=1) -> T |}]

let%expect_test "pp: relu.default config" =
  (match Aten_op_config.find "torch.ops.aten.relu.default" with
  | None -> print_string "not found"
  | Some c -> Format.printf "%a@." Aten_op_config.pp c);
  [%expect {| torch.ops.aten.relu.default (Tensor self) -> T |}]
