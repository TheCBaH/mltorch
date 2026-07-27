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
  let shape6 =
    match Aten_shape.of_aten (Array.of_list shape) with
    | Ok shape6 -> shape6
    | Error e ->
        failwith (Format.asprintf "%a" Aten_shape.pp_error e.Core.Error.kind)
  in
  Tensor.Tensor
    {
      shape = shape6;
      payload = { Payload.fmt = Payload.F32; quant = Payload.No_quant; data };
    }

let native_i64 shape vals =
  let n = List.fold_left ( * ) 1 shape in
  let data = Array1.create int64 c_layout n in
  List.iteri (fun i x -> data.{i} <- x) vals;
  let shape6 =
    match Aten_shape.of_aten (Array.of_list shape) with
    | Ok shape6 -> shape6
    | Error e ->
        failwith (Format.asprintf "%a" Aten_shape.pp_error e.Core.Error.kind)
  in
  Tensor.Tensor
    {
      shape = shape6;
      payload = { Payload.fmt = Payload.I64; quant = Payload.No_quant; data };
    }

let pp_result =
  Core.Pretty.core_result ~ok:(Fmt.any "Ok") ~error:(fun ppf e ->
      Fmt.pf ppf "Error: %a" Verify.pp_error e)

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

(* A non-contiguous ATen tensor (here a permute view) is materialized contiguous
   before conversion, so the native tensor holds the logical (transposed) order.
   This is what lets addmm's transposed fc weight convert on the real graph. *)
let%expect_test "of_aten: materializes a non-contiguous permute view" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  let aten =
    Aten_tensor.manage
      (Aten_c.Aten_operations.permute x (Interp_decode.arr [ 1; 0 ]) 2)
  in
  (match Tensor_bridge.of_aten aten with
  | Error msg -> Printf.printf "Error: %s\n" msg
  | Ok native -> Format.printf "%a@." Tensor.pp native);
  [%expect {| tensor f32 [W=3 C=2] {0, 3, 1, 4, 2, 5} |}]

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
let in_int name i = PT.NamedArgument.make name (PT.Argument.Int i) None
let in_ints name xs = PT.NamedArgument.make name (PT.Argument.Ints xs) None
let in_bool name b = PT.NamedArgument.make name (PT.Argument.Bool b) None
let in_float name f = PT.NamedArgument.make name (PT.Argument.Float f) None
let in_none name = PT.NamedArgument.make name (PT.Argument.None false) None

(* Bind each (name, ATen tensor) into an env and dispatch a one-node graph;
   print each native output as "shape {values}".  The env key and the input's
   TensorArgument name are the same [name], which is how the bridge resolves it. *)
let dispatch_print_with_graph ~print_graph ~target ~bindings ~inputs ~noutputs =
  let env = List.fold_left (fun m (k, t) -> Sm.add k t m) Sm.empty bindings in
  let outputs = List.init noutputs (fun i -> targ (Printf.sprintf "out%d" i)) in
  let node = PT.Node.make target inputs outputs Sm.empty None (Some "test") in
  match Op_bridge.dispatch ~aten_env:env node with
  | None -> print_string "no native impl\n"
  | Some (Error e) ->
      Format.printf "error: %a@." Op_bridge.pp_error e.Core.Error.kind
  | Some (Ok (graph, bindings)) -> (
      if print_graph then Format.printf "%a@." Graph_ir.pp graph;
      match Eval_direct.run graph ~inputs:bindings with
      | Error e ->
          Format.printf "eval error: %a@." Eval_direct.pp_error
            e.Core.Error.kind
      | Ok result_env ->
          let outs =
            List.map
              (fun oid -> Graph_ir.Tensor_id.Map.find oid result_env)
              graph.Graph_ir.Graph.outputs
          in
          List.iter (fun o -> Format.printf "%a@." Tensor.pp o) outs)

let dispatch_print ~target ~bindings ~inputs ~noutputs =
  dispatch_print_with_graph ~print_graph:false ~target ~bindings ~inputs
    ~noutputs

let%expect_test "dispatch: div.Tensor elementwise" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 2; 3 ] [ 2.; 4.; 0.5; 1.; 10.; 3. ] in
  dispatch_print ~target:"torch.ops.aten.div.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=3] {0.5, 0.5, 6, 4, 0.5, 2} |}]

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

let%expect_test "dispatch: sqrt.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ 0.; 1.; 4.; 2.25 ] in
  dispatch_print ~target:"torch.ops.aten.sqrt.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {0, 1, 2, 1.5} |}]

let%expect_test "dispatch: sub.Tensor elementwise" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 2; 3 ] [ 0.; 10.; 100.; 1.; 2.; 3. ] in
  dispatch_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=3] {1, -8, -97, 3, 3, 3} |}]

let%expect_test "dispatch: addmm.default relayouts [In,Out] weight" =
  let bias = float_tensor [ 2 ] [ 10.; 100. ] in
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 3; 2 ] [ 1.; 0.; 0.; 1.; 0.; 1. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.addmm.default"
    ~bindings:[ ("self", bias); ("mat1", x); ("mat2", w) ]
    ~inputs:[ in_tensor "self"; in_tensor "mat1"; in_tensor "mat2" ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [C=2], t1 f32 [W=2 C=3], t2 f32 [W=3 C=2]]
    nodes:
      n0: [t3 f32 [N=2 T=1 D=1 H=1 W=1 C=3]] =
        permute x=t2 perm=[N<-C, W<-N, C<-W]
      n1: [t4 f32 [W=2 C=2]] =
        linear x=t1 weight=t3 bias=t0 params={in_features=3}
    outputs: [t4 f32 [W=2 C=2]]
    tensor f32 [W=2 C=2] {11, 105, 14, 111} |}]

let%expect_test "dispatch: _native_batch_norm_legit_no_training per-channel" =
  (* NCHW [1,2,1,2]: c0 = [1,3], c1 = [5,7]. mean=[1,5], var=[4,4] (inv=0.5),
     weight=[2,10], bias=[1,-1], eps=0 -> y = (x-mean)*0.5*w+b: c0=[1,3],
     c1=[-1,9]. Only out0 is produced (the size-0 save_* outputs are dropped). *)
  let x = float_tensor [ 1; 2; 1; 2 ] [ 1.; 3.; 5.; 7. ] in
  let w = float_tensor [ 2 ] [ 2.; 10. ] in
  let b = float_tensor [ 2 ] [ 1.; -1. ] in
  let rm = float_tensor [ 2 ] [ 1.; 5. ] in
  let rv = float_tensor [ 2 ] [ 4.; 4. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten._native_batch_norm_legit_no_training.default"
    ~bindings:
      [
        ("input", x);
        ("weight", w);
        ("bias", b);
        ("running_mean", rm);
        ("running_var", rv);
      ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_tensor "bias";
        in_tensor "running_mean";
        in_tensor "running_var";
        in_float "momentum" 0.1;
        in_float "eps" 0.;
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs:
      [t0 f32 [H=2 W=1 C=2], t1 f32 [C=2], t2 f32 [C=2], t3 f32 [C=2],
       t4 f32 [C=2]]
    nodes:
      n0: [t5 f32 [W=2 C=2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t6 f32 [W=2 C=2]] =
        batch_norm
          x=t5
          weight=t1
          bias=t2
          running_mean=t3
          running_var=t4
          params={channel=C; eps=0}
      n2: [t7 f32 [H=2 W=1 C=2]] = permute x=t6 perm=[H<-C, W<-H, C<-W]
    outputs: [t7 f32 [H=2 W=1 C=2]]
    tensor f32 [H=2 W=1 C=2] {1, 3, -1, 9} |}]

let%expect_test "dispatch: conv2d.default relayouts NCHW/OIHW with bias" =
  let x = float_tensor [ 1; 1; 3; 3 ] (List.init 9 float_of_int) in
  let w = float_tensor [ 1; 1; 2; 2 ] [ 1.; 1.; 1.; 1. ] in
  let bias = float_tensor [ 1 ] [ 10. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.conv2d.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", bias) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_tensor "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 0 ];
        in_ints "dilation" [ 1; 1 ];
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=3 C=3], t1 f32 [W=2 C=2], t2 f32 [C=1]]
    nodes:
      n0: [t3 f32 [H=3 W=3 C=1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t4 f32 [H=2 W=2 C=1]] =
        permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2: [t5 f32 [H=2 W=2 C=1]] =
        conv2d
          x=t3
          weight=t4
          bias=t2
          params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=1;
                 groups=1}
      n3: [t6 f32 [W=2 C=2]] = permute x=t5 perm=[H<-C, W<-H, C<-W]
    outputs: [t6 f32 [W=2 C=2]]
    tensor f32 [W=2 C=2] {18, 22, 30, 34} |}]

let%expect_test "dispatch: conv2d.padding same uses distinct native op" =
  let x = float_tensor [ 1; 1; 3; 3 ] (List.init 9 float_of_int) in
  let w = float_tensor [ 1; 1; 2; 2 ] [ 1.; 1.; 1.; 1. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.conv2d.padding"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_none "bias";
        in_ints "stride" [ 1; 1 ];
        PT.NamedArgument.make "padding" (PT.Argument.String "same") None;
        in_ints "dilation" [ 1; 1 ];
        in_int "groups" 1;
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=3 C=3], t1 f32 [W=2 C=2]]
    nodes:
      n0: [t2 f32 [H=3 W=3 C=1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t3 f32 [H=2 W=2 C=1]] =
        permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2: [t4 f32 [H=3 W=3 C=1]] =
        conv2d_padding
          x=t2
          weight=t3
          bias=none
          params={stride={h=1; w=1}; padding=same; dilation={h=1; w=1}; groups=1}
      n3: [t5 f32 [W=3 C=3]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [W=3 C=3]]
    tensor f32 [W=3 C=3] {8, 12, 7, 20, 24, 13, 13, 15, ...} |}]

let%expect_test "dispatch: conv2d.padding invalid weight rank is typed" =
  let x = float_tensor [ 1; 1; 3; 3 ] (List.init 9 float_of_int) in
  let w = float_tensor [ 1; 4 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.conv2d.padding"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_none "bias";
        in_ints "stride" [ 1; 1 ];
        PT.NamedArgument.make "padding" (PT.Argument.String "same") None;
        in_ints "dilation" [ 1; 1 ];
        in_int "groups" 1;
      ]
    ~noutputs:1;
  [%expect {| error: conv2d.padding: weight must be rank-4, got shape [1, 4] |}]

let%expect_test "dispatch: convolution.default uses distinct native op" =
  let x = float_tensor [ 1; 1; 3; 3 ] (List.init 9 float_of_int) in
  let w = float_tensor [ 1; 1; 2; 2 ] [ 1.; 1.; 1.; 1. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.convolution.default"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_none "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 0 ];
        in_ints "dilation" [ 1; 1 ];
        in_bool "transposed" false;
        in_ints "output_padding" [ 0; 0 ];
        in_int "groups" 1;
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=3 C=3], t1 f32 [W=2 C=2]]
    nodes:
      n0: [t2 f32 [H=3 W=3 C=1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t3 f32 [H=2 W=2 C=1]] =
        permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2: [t4 f32 [H=2 W=2 C=1]] =
        convolution
          x=t2
          weight=t3
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n3: [t5 f32 [W=2 C=2]] = permute x=t4 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [W=2 C=2]]
    tensor f32 [W=2 C=2] {8, 12, 20, 24} |}]

let%expect_test "dispatch: convolution.default grouped conv2d" =
  let x = float_tensor [ 1; 4; 1; 1 ] [ 1.; 2.; 10.; 20. ] in
  let w = float_tensor [ 4; 2; 1; 1 ] [ 1.; 1.; 10.; 0.; 1.; 1.; 0.; 2. ] in
  let bias = float_tensor [ 4 ] [ 0.; 100.; 1000.; 10000. ] in
  dispatch_print ~target:"torch.ops.aten.convolution.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", bias) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_tensor "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 0 ];
        in_ints "dilation" [ 1; 1 ];
        in_bool "transposed" false;
        in_ints "output_padding" [ 0; 0 ];
        in_int "groups" 2;
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [H=4 W=1 C=1] {3, 110, 1030, 10040} |}]

let%expect_test "dispatch: convolution.default transposed" =
  let x = float_tensor [ 1; 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  let w = float_tensor [ 1; 1; 2; 2 ] [ 1.; 1.; 1.; 1. ] in
  dispatch_print ~target:"torch.ops.aten.convolution.default"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_none "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 0 ];
        in_ints "dilation" [ 1; 1 ];
        in_bool "transposed" true;
        in_ints "output_padding" [ 0; 0 ];
        in_int "groups" 1;
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=3 C=3] {1, 3, 2, 4, 10, 6, 3, 7, ...} |}]

let%expect_test "dispatch: conv2d.default dilated spatial window" =
  let x = float_tensor [ 1; 1; 1; 5 ] [ 0.; 1.; 2.; 3.; 4. ] in
  let w = float_tensor [ 1; 1; 1; 3 ] [ 1.; 1.; 1. ] in
  dispatch_print ~target:"torch.ops.aten.conv2d.default"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_none "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 1 ];
        in_ints "dilation" [ 1; 2 ];
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=3] {4, 6, 4} |}]

let%expect_test
    "dispatch: conv2d.default combines bias stride padding dilation and groups"
    =
  let x =
    float_tensor [ 1; 2; 1; 7 ]
      [ 0.; 1.; 2.; 3.; 4.; 5.; 6.; 10.; 11.; 12.; 13.; 14.; 15.; 16. ]
  in
  let w = float_tensor [ 2; 1; 1; 2 ] [ 1.; 1.; 2.; 3. ] in
  let bias = float_tensor [ 2 ] [ 10.; 100. ] in
  dispatch_print ~target:"torch.ops.aten.conv2d.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", bias) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_tensor "bias";
        in_ints "stride" [ 1; 2 ];
        in_ints "padding" [ 0; 1 ];
        in_ints "dilation" [ 1; 2 ];
        in_int "groups" 2;
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [H=2 W=1 C=4] {11, 14, 18, 15, 133, 161, 171, 130} |}]

let%expect_test "dispatch: linear.default relayouts [Out,In] weight with bias" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 2; 3 ] [ 1.; 0.; 0.; 0.; 1.; 1. ] in
  let bias = float_tensor [ 2 ] [ 10.; 100. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.linear.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", bias) ]
    ~inputs:[ in_tensor "input"; in_tensor "weight"; in_tensor "bias" ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3], t1 f32 [W=2 C=3], t2 f32 [C=2]]
    nodes:
      n0: [t3 f32 [N=2 T=1 D=1 H=1 W=1 C=3]] = permute x=t1 perm=[N<-W, W<-N]
      n1: [t4 f32 [W=2 C=2]] =
        linear x=t0 weight=t3 bias=t2 params={in_features=3}
    outputs: [t4 f32 [W=2 C=2]]
    tensor f32 [W=2 C=2] {11, 105, 14, 111} |}]

let%expect_test "dispatch: linear.default accepts explicit None bias" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 2; 3 ] [ 1.; 0.; 0.; 0.; 1.; 1. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.linear.default"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:[ in_tensor "input"; in_tensor "weight"; in_none "bias" ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3], t1 f32 [W=2 C=3]]
    nodes:
      n0: [t2 f32 [N=2 T=1 D=1 H=1 W=1 C=3]] = permute x=t1 perm=[N<-W, W<-N]
      n1: [t3 f32 [W=2 C=2]] =
        linear x=t0 weight=t2 bias=none params={in_features=3}
    outputs: [t3 f32 [W=2 C=2]]
    tensor f32 [W=2 C=2] {1, 5, 4, 11} |}]

let%expect_test "dispatch: max_pool2d.default relayouts NCHW input and output" =
  let x =
    float_tensor [ 1; 1; 3; 3 ] (List.init 9 (fun i -> float_of_int (-(i + 1))))
  in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.max_pool2d.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_ints "kernel_size" [ 2; 2 ];
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 1; 1 ];
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=3 C=3]]
    nodes:
      n0: [t1 f32 [H=3 W=3 C=1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t2 f32 [H=4 W=4 C=1]] =
        max_pool2d
          x=t1
          params={kernel={h=2; w=2}; stride={h=1; w=1}; pad={h=1; w=1}}
      n2: [t3 f32 [W=4 C=4]] = permute x=t2 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=4 C=4]]
    tensor f32 [W=4 C=4] {-1, -1, -2, -3, -1, -1, -2, -3, ...} |}]

let%expect_test "dispatch: max_pool2d_with_indices.default discards indices" =
  (* NCHW [1,1,4,4], value(h,w)=h*4+w. 2x2/stride-2 windows: max is each
     window's bottom-right; the graph output is the relayout'd values, and the
     dead indices edge is routed into a Discard node. *)
  let x = float_tensor [ 1; 1; 4; 4 ] (List.init 16 float_of_int) in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.max_pool2d_with_indices.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_ints "kernel_size" [ 2; 2 ];
        in_ints "stride" [ 2; 2 ];
        in_ints "padding" [ 0; 0 ];
      ]
    ~noutputs:2;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=4 C=4]]
    nodes:
      n0: [t1 f32 [H=4 W=4 C=1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t2 f32 [H=2 W=2 C=1], t3 f32 [H=2 W=2 C=1]] =
        max_pool2d_with_indices
          x=t1
          params={kernel={h=2; w=2}; stride={h=2; w=2}; pad={h=0; w=0}}
      n2: [] = discard x=t3
      n3: [t4 f32 [W=2 C=2]] = permute x=t2 perm=[H<-C, W<-H, C<-W]
    outputs: [t4 f32 [W=2 C=2]]
    tensor f32 [W=2 C=2] {5, 7, 13, 15} |}]

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

let%expect_test "dispatch: mean.dim dim=[] reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" []; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {2.5} |}]

let%expect_test "dispatch: mean.dim omitted dim reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {2.5} |}]

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

let%expect_test "dispatch: view.default contiguous reshape (no permute)" =
  (* NCHW [1,1,2,3] (values 0..5) viewed as [3,2]. of_aten inputs are already
     ATen-row-major, so the graph is a single reshape (no surrounding permute);
     a contiguous reshape leaves the flat buffer unchanged. *)
  let x = float_tensor [ 1; 1; 2; 3 ] (List.init 6 float_of_int) in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.view.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "size" [ 3; 2 ] ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3]]
    nodes:
      n0: [t1 f32 [W=3 C=2]] = reshape x=t0 params={shape=[W=3 C=2]}
    outputs: [t1 f32 [W=3 C=2]]
    tensor f32 [W=3 C=2] {0, 1, 2, 3, 4, 5} |}]

let%expect_test "PT2 provenance: native ids map to qualified source origins" =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let graph =
    match
      Graph_builder.(
        build ~name:"ignored" ~outputs:(fun r -> [ r ])
        @@
        let* x = input ~shape () in
        let* weight = constant ~shape () in
        add x weight)
    with
    | Ok graph -> graph
    | Error _ -> assert false
  in
  let weight = List.nth graph.Graph_ir.Graph.inputs 1 in
  let node = List.hd graph.Graph_ir.Graph.nodes in
  let tensor_origins =
    Graph_ir.Tensor_id.Map.singleton weight
      (Pt2_native_graph.Source
         {
           Pt2_native_graph.Tensor_origin.graph_path = [ 4; 2 ];
           ssa_name = "p_layer_weight";
           meta = None;
         })
  in
  let node_origins =
    Graph_ir.Node_id.Map.singleton node.Graph_ir.Node.id
      [
        {
          Pt2_native_graph.Node_origin.graph_path =
            Pt2_native_graph.Graph_path.root;
          index = 7;
          target = "torch.ops.aten.add.Tensor";
          name = Some "add";
          metadata = Schema_runtime.String_map.empty;
        };
      ]
  in
  match
    Pt2_native_graph.make ~graph ~tensor_origins ~node_origins
      ~captured_targets:(Graph_ir.Tensor_id.Map.singleton weight "layer.weight")
  with
  | Error _ -> print_endline "unexpected error"
  | Ok provenance ->
      let origin =
        Graph_ir.Tensor_id.Map.find weight provenance.tensor_origins
      in
      (match origin with
      | Pt2_native_graph.Source { graph_path; ssa_name; _ } ->
          Format.printf "tensor=t%d path=%a name=%s target=%s@."
            (Graph_ir.Tensor_id.to_int weight)
            Pt2_native_graph.Graph_path.pp graph_path ssa_name
            (Graph_ir.Tensor_id.Map.find weight provenance.captured_targets)
      | Derived -> assert false);
      [%expect
        {| tensor=t1 path=root/4/2 name=p_layer_weight target=layer.weight |}]
