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

(* ---- Op_bridge.dispatch: native compute, evaluated directly ------------- *)

(* These drive the native path of the bridge end to end — build an ATen input
   env + a single-node graph, run [Op_bridge.dispatch], and print the native
   output (shape + values).  Unlike the ATen-vs-native spec tests, the expected
   values here are hand-derived, so they pin the native compute independently of
   ATen as the oracle. *)

module PT = Pytorch_types
module Sm = Schema_runtime.String_map

let targ name = PT.Argument.Tensor (PT.TensorArgument.make name)
let in_tensor name = PT.NamedArgument.make name (targ name) None
let in_ints name xs = PT.NamedArgument.make name (PT.Argument.Ints xs) None
let in_bool name b = PT.NamedArgument.make name (PT.Argument.Bool b) None
let in_float name f = PT.NamedArgument.make name (PT.Argument.Float f) None

(* Bind each (name, ATen tensor) into an env and dispatch a one-node graph;
   print each native output as "shape {values}".  The env key and the input's
   TensorArgument name are the same [name], which is how the bridge resolves it. *)
let dispatch_print ~target ~bindings ~inputs ~noutputs =
  let env = List.fold_left (fun m (k, t) -> Sm.add k t m) Sm.empty bindings in
  let outputs = List.init noutputs (fun i -> targ (Printf.sprintf "out%d" i)) in
  let node = PT.Node.make target inputs outputs Sm.empty None (Some "test") in
  match Op_bridge.dispatch ~aten_env:env node with
  | None -> print_string "no native impl\n"
  | Some (Error e) -> Printf.printf "error: %s\n" e
  | Some (Ok outs) -> List.iter (fun o -> Format.printf "%a@." Tensor.pp o) outs

let%expect_test "dispatch: mul.Tensor elementwise" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 2; 3 ] [ 0.; 10.; 100.; 1.; 2.; 3. ] in
  dispatch_print ~target:"torch.ops.aten.mul.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=3] {0, 20, 300, 4, 10, 18} |}]

let%expect_test "dispatch: bmm 1x2x2 @ 1x2x2" =
  let a = float_tensor [ 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  let b = float_tensor [ 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.bmm.default"
    ~bindings:[ ("self", a); ("mat2", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "mat2" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {7, 10, 15, 22} |}]

let%expect_test "dispatch: mean.dim dim=[1] keepdim=true" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" true ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=1] {1, 4} |}]

let%expect_test "dispatch: mean.dim dim=[1] keepdim=false" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=2] {1, 4} |}]

let%expect_test "dispatch: rms_norm normalized_shape=[3] with weight" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 3 ] [ 1.; 1.; 1. ] in
  dispatch_print ~target:"torch.ops.aten.rms_norm.default"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_ints "normalized_shape" [ 3 ];
        in_tensor "weight";
        in_float "eps" 1e-5;
      ]
    ~noutputs:1;
  [%expect
    {| tensor f32 [W=2 C=3] {0.46291, 0.925819, 1.38873, 0.789542, 0.986927, 1.18431} |}]

let%expect_test "dispatch: rms_norm no weight (ones) matches affine identity" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  dispatch_print ~target:"torch.ops.aten.rms_norm.default"
    ~bindings:[ ("input", x) ]
    ~inputs:
      [
        in_tensor "input"; in_ints "normalized_shape" [ 3 ]; in_float "eps" 1e-5;
      ]
    ~noutputs:1;
  [%expect
    {| tensor f32 [W=2 C=3] {0.46291, 0.925819, 1.38873, 0.789542, 0.986927, 1.18431} |}]

let%expect_test
    "dispatch: permute.default rank-2 dims=[1,0] — transposes W and C" =
  (* ATen [2,3] right-aligns to native [W=2 C=3]; permute([1,0]) swaps dims ->
     output [W=3 C=2], i.e. the transpose [[0,3],[1,4],[2,5]] row-major. *)
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.permute.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dims" [ 1; 0 ] ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=3 C=2] {0, 3, 1, 4, 2, 5} |}]

let%expect_test "dispatch: permute.default rank-3 dims=[2,0,1] — CHW cycle" =
  (* ATen [2,3,4] -> native [H=2 W=3 C=4]; permute([2,0,1]) cycles dims:
     output ATen dim 0 <- input dim 2, dim 1 <- dim 0, dim 2 <- dim 1.
     Frame: output H <- input C, output W <- input H, output C <- input W.
     Output shape: H=C_in=4, W=H_in=2, C=W_in=3. *)
  let vals = List.init 24 float_of_int in
  let x = float_tensor [ 2; 3; 4 ] vals in
  dispatch_print ~target:"torch.ops.aten.permute.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dims" [ 2; 0; 1 ] ]
    ~noutputs:1;
  (* output[h,w,c] = input[w, c, h] (inverse of the cycle applied to indices):
     (h=0,w=0,c=0): input[0,0,0]=0; (h=0,w=0,c=1): input[0,1,0]=4 *)
  [%expect {| tensor f32 [H=4 W=2 C=3] {0, 4, 8, 12, 16, 20, 1, 5, ...} |}]

let%expect_test "dispatch: permute.default identity — output equals input" =
  let x = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.permute.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dims" [ 0; 1 ] ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=3 C=4] {0, 1, 2, 3, 4, 5, 6, 7, ...} |}]
