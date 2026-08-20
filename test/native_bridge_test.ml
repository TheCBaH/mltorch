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
    Aten_shape.of_aten (Array.of_list shape)
    |> Err.or_raise ~pp_error:Aten_shape.pp_error
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
    Aten_shape.of_aten (Array.of_list shape)
    |> Err.or_raise ~pp_error:Aten_shape.pp_error
  in
  Tensor.Tensor
    {
      shape = shape6;
      payload = { Payload.fmt = Payload.I64; quant = Payload.No_quant; data };
    }

let pp_result =
  Core.Pretty.err_result ~ok:(Fmt.any "Ok") ~error:(fun ppf e ->
      Fmt.pf ppf "Error: %a" Verify.pp_error e)

(* [Tensor_bridge.of_aten] returns a plain (not [Err.t]) error, so this
   composes with [Core.Pretty.result] rather than [err_result]; [~ok] varies
   by test (the tensor itself, just its shape, or a surprise-success marker). *)
let pp_of_aten_result ~ok =
  Core.Pretty.err_result ~ok ~error:(fun ppf e ->
      Fmt.pf ppf "Error: %a" Tensor_bridge.pp_error e)

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

let%expect_test "of_aten: unsupported dtype returns Error" =
  let t = T.create ~dtype:Stype.Double [ 2 ] in
  Format.printf "%a@."
    (pp_of_aten_result ~ok:(Fmt.any "unexpected Ok"))
    (Tensor_bridge.of_aten t);
  [%expect {| Error: unsupported ATen dtype (code 7) |}]

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
let in_string name s = PT.NamedArgument.make name (PT.Argument.String s) None

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
      Format.printf "error: %a@." Op_bridge.pp_error (Err.Error.kind e)
  | Some (Ok (graph, bindings)) -> (
      if print_graph then Format.printf "%a@." Graph_ir.pp graph;
      match Eval_direct.run graph ~inputs:bindings with
      | Error e ->
          Format.printf "eval error: %a@." Eval_direct.pp_error
            (Err.Error.kind e)
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

(* [view.default] and [_unsafe_view.default] share one dispatch arm
   (op3-impl.md commit 4, Part IV #2), so every case worth proving for one is
   worth proving for both -- [_unsafe_view] cannot be allowed to hide behind
   [view.default]'s coverage. *)
let view_targets =
  [ "torch.ops.aten.view.default"; "torch.ops.aten._unsafe_view.default" ]

let dispatch_view_both ~x ~size =
  List.iter
    (fun target ->
      dispatch_print ~target
        ~bindings:[ ("self", x) ]
        ~inputs:[ in_tensor "self"; in_ints "size" size ]
        ~noutputs:1)
    view_targets

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
    inputs:
      [t0 f32 [C=2] ->[n1], t1 f32 [W=2 C=3] ->[n1], t2 f32 [W=3 C=2] ->[n0]]
    nodes:
      n0: [t3 f32 [N=2 T=1 D=1 H=1 W=1 C=3] ->[n1]] =
        permute x=t2 perm=[N<-C, W<-N, C<-W]
      n1: [t4 f32 [W=2 C=2]] =
        linear x=t1 weight=t3 <-n0 bias=t0 params={in_features=3}
    outputs: [t4 f32 [W=2 C=2] <-n1]
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
      [t0 f32 [H=2 W=1 C=2] ->[n0], t1 f32 [C=2] ->[n1], t2 f32 [C=2] ->[n1],
       t3 f32 [C=2] ->[n1], t4 f32 [C=2] ->[n1]]
    nodes:
      n0: [t5 f32 [W=2 C=2] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t6 f32 [W=2 C=2] ->[n2]] =
        batch_norm
          x=t5 <-n0
          weight=t1
          bias=t2
          running_mean=t3
          running_var=t4
          params={channel=C; eps=0}
      n2: [t7 f32 [H=2 W=1 C=2]] = permute x=t6 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t7 f32 [H=2 W=1 C=2] <-n2]
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
    inputs:
      [t0 f32 [W=3 C=3] ->[n0], t1 f32 [W=2 C=2] ->[n1], t2 f32 [C=1] ->[n2]]
    nodes:
      n0: [t3 f32 [H=3 W=3 C=1] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t4 f32 [H=2 W=2 C=1] ->[n2]] =
        permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2: [t5 f32 [H=2 W=2 C=1] ->[n3]] =
        conv2d
          x=t3 <-n0
          weight=t4 <-n1
          bias=t2
          params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=1;
                 groups=1}
      n3: [t6 f32 [W=2 C=2]] = permute x=t5 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t6 f32 [W=2 C=2] <-n3]
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
    inputs: [t0 f32 [W=3 C=3] ->[n0], t1 f32 [W=2 C=2] ->[n1]]
    nodes:
      n0: [t2 f32 [H=3 W=3 C=1] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t3 f32 [H=2 W=2 C=1] ->[n2]] =
        permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2: [t4 f32 [H=3 W=3 C=1] ->[n3]] =
        conv2d_padding
          x=t2 <-n0
          weight=t3 <-n1
          bias=none
          params={stride={h=1; w=1}; padding=same; dilation={h=1; w=1}; groups=1}
      n3: [t5 f32 [W=3 C=3]] = permute x=t4 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [W=3 C=3] <-n3]
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
    inputs: [t0 f32 [W=3 C=3] ->[n0], t1 f32 [W=2 C=2] ->[n1]]
    nodes:
      n0: [t2 f32 [H=3 W=3 C=1] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t3 f32 [H=2 W=2 C=1] ->[n2]] =
        permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2: [t4 f32 [H=2 W=2 C=1] ->[n3]] =
        convolution
          x=t2 <-n0
          weight=t3 <-n1
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n3: [t5 f32 [W=2 C=2]] = permute x=t4 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [W=2 C=2] <-n3]
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
    inputs:
      [t0 f32 [W=2 C=3] ->[n1], t1 f32 [W=2 C=3] ->[n0], t2 f32 [C=2] ->[n1]]
    nodes:
      n0: [t3 f32 [N=2 T=1 D=1 H=1 W=1 C=3] ->[n1]] =
        permute x=t1 perm=[N<-W, W<-N]
      n1: [t4 f32 [W=2 C=2]] =
        linear x=t0 weight=t3 <-n0 bias=t2 params={in_features=3}
    outputs: [t4 f32 [W=2 C=2] <-n1]
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
    inputs: [t0 f32 [W=2 C=3] ->[n1], t1 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t2 f32 [N=2 T=1 D=1 H=1 W=1 C=3] ->[n1]] =
        permute x=t1 perm=[N<-W, W<-N]
      n1: [t3 f32 [W=2 C=2]] =
        linear x=t0 weight=t2 <-n0 bias=none params={in_features=3}
    outputs: [t3 f32 [W=2 C=2] <-n1]
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
    inputs: [t0 f32 [W=3 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [H=3 W=3 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t2 f32 [H=4 W=4 C=1] ->[n2]] =
        max_pool2d
          x=t1 <-n0
          params={kernel={h=2; w=2}; stride={h=1; w=1}; pad={h=1; w=1}}
      n2: [t3 f32 [W=4 C=4]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=4 C=4] <-n2]
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
    inputs: [t0 f32 [W=4 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=4 W=4 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t2 f32 [H=2 W=2 C=1] ->[n3], t3 f32 [H=2 W=2 C=1] ->[n2]] =
        max_pool2d_with_indices
          x=t1 <-n0
          params={kernel={h=2; w=2}; stride={h=2; w=2}; pad={h=0; w=0}}
      n2: [] = discard x=t3 <-n1
      n3: [t4 f32 [W=2 C=2]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t4 f32 [W=2 C=2] <-n3]
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

(* [Aten_shape.axis_of_dim] asserts its range and raises; before commit 0 this
   escaped [Op_bridge.dispatch] as an uncaught [Invalid_argument] rather than
   the typed row every other bad-argument arm returns. *)
let%expect_test "dispatch: mean.dim rejects an out-of-range dim" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.mean.dim"
        ~bindings:[ ("self", x) ]
        ~inputs:
          [ in_tensor "self"; in_ints "dim" [ d ]; in_bool "keepdim" false ]
        ~noutputs:1)
    [ 7; -3 ];
  [%expect
    {|
    error: mean.dim: invalid dimension 7 for rank 2
    error: mean.dim: invalid dimension -3 for rank 2 |}]

(* ---- the layer_norm.default arm ----------------------------------------- *)

(* Both affine operands are optional in the schema and in [Graph_ir], so all
   four states are dispatched, and the GRAPH is printed for each. An arm that
   materialised a ones/zeros tensor for an absent operand would build a
   structurally different graph from [Native_interp]'s for the same node and
   leave that arm unreachable -- the bug the rms_norm block below records at
   length. "bias but no weight" is the state no exported model produces and the
   one a paired encoding gets wrong, so it is dispatched rather than assumed.

   The values are the claim beneath the structure: with weight all ones and bias
   all zeros the four states must agree elementwise, and they do. *)
let%expect_test "dispatch: layer_norm.default builds only the operands present"
    =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let ones = float_tensor [ 3 ] [ 1.; 1.; 1. ] in
  let zeros = float_tensor [ 3 ] [ 0.; 0.; 0. ] in
  let state label affine =
    Format.printf "%s@." label;
    dispatch_print_with_graph ~print_graph:true
      ~target:"torch.ops.aten.layer_norm.default"
      ~bindings:
        ([ ("input", x) ]
        @ List.map (fun n -> (n, if n = "weight" then ones else zeros)) affine)
      ~inputs:
        ([ in_tensor "input"; in_ints "normalized_shape" [ 3 ] ]
        @ List.map in_tensor affine
        @ [ in_float "eps" 1e-5 ])
      ~noutputs:1
  in
  state "neither:" [];
  state "weight:" [ "weight" ];
  state "bias:" [ "bias" ];
  state "both:" [ "weight"; "bias" ];
  [%expect
    {|
    neither:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] =
        layer_norm x=t0 weight=none bias=none params={dims=[C]; eps=1e-05}
    outputs: [t1 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    weight:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [C=3] ->[n0]]
    nodes:
      n0: [t2 f32 [W=2 C=3]] =
        layer_norm x=t0 weight=t1 bias=none params={dims=[C]; eps=1e-05}
    outputs: [t2 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    bias:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [C=3] ->[n0]]
    nodes:
      n0: [t2 f32 [W=2 C=3]] =
        layer_norm x=t0 weight=none bias=t1 params={dims=[C]; eps=1e-05}
    outputs: [t2 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    both:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [C=3] ->[n0], t2 f32 [C=3] ->[n0]]
    nodes:
      n0: [t3 f32 [W=2 C=3]] =
        layer_norm x=t0 weight=t1 bias=t2 params={dims=[C]; eps=1e-05}
    outputs: [t3 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474} |}]

(* A weight that is not all ones and a bias that is not all zeros, both
   negative-valued somewhere: [* weight] then [+ bias] and the reverse order
   give different answers only when both are non-trivial. *)
let%expect_test "dispatch: layer_norm.default applies weight then bias" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 3 ] [ 2.; -1.; 0.5 ] in
  let b = float_tensor [ 3 ] [ 0.25; 0.; -1. ] in
  dispatch_print ~target:"torch.ops.aten.layer_norm.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", b) ]
    ~inputs:
      [
        in_tensor "input";
        in_ints "normalized_shape" [ 3 ];
        in_tensor "weight";
        in_tensor "bias";
        in_float "eps" 1e-5;
      ]
    ~noutputs:1;
  [%expect
    {| tensor f32 [W=2 C=3] {-2.19947, 0, -0.387632, -2.19947, 0, -0.387632} |}]

(* Same validation as rms_norm -- [normalized_dims] is shared -- but the
   diagnostics now name the op that failed, which is what the [op] field added
   to [Normalized_rank]/[Normalized_shape] buys. *)
let%expect_test "dispatch: layer_norm validates normalized_shape" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let bad normalized_shape =
    dispatch_print ~target:"torch.ops.aten.layer_norm.default"
      ~bindings:[ ("input", x) ]
      ~inputs:[ in_tensor "input"; in_ints "normalized_shape" normalized_shape ]
      ~noutputs:1
  in
  bad [ 2 ];
  bad [ 3; 2 ];
  bad [ 2; 3; 3 ];
  bad [];
  (* and the shapes that DO match still lower, with eps defaulted to 1e-5 *)
  bad [ 3 ];
  bad [ 2; 3 ];
  [%expect
    {|
    error: layer_norm: normalized_shape [2] does not match the input's trailing extents [3]
    error: layer_norm: normalized_shape [3, 2] does not match the input's trailing extents [2, 3]
    error: layer_norm: normalized_shape has 3 entries, outside [1, 2] for this rank
    error: layer_norm: normalized_shape has 0 entries, outside [1, 2] for this rank
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    tensor f32 [W=2 C=3] {-1.46385, -0.878309, -0.29277, 0.29277, 0.878309, 1.46385} |}]

(* [cudnn_enable] is accepted at BOTH values and when omitted, and all three
   produce the same tensor: ATen's own composite names it
   [bool /* cudnn_enable, deprecated */] and drops it, so the native contract is
   backend-independent by construction. It is still DECODED -- a non-boolean
   there is a malformed node, and the fourth line is what says so. *)
let%expect_test "dispatch: layer_norm accepts cudnn_enable at both values" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let run extra =
    dispatch_print ~target:"torch.ops.aten.layer_norm.default"
      ~bindings:[ ("input", x) ]
      ~inputs:([ in_tensor "input"; in_ints "normalized_shape" [ 3 ] ] @ extra)
      ~noutputs:1
  in
  run [];
  run [ in_bool "cudnn_enable" true ];
  run [ in_bool "cudnn_enable" false ];
  run [ in_int "cudnn_enable" 1 ];
  [%expect
    {|
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    error: argument "cudnn_enable": expected bool, got Int |}]

(* Both affine operands carry the whole normalized_shape, so both are rank [k].
   A rank-1 weight against a two-axis normalization right-aligns into the frame
   indistinguishably from a correct one-axis weight, which is why the rank is
   read on the ATen tensor before the bridge crosses. *)
let%expect_test "dispatch: layer_norm rejects a wrong-rank weight or bias" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let flat = float_tensor [ 3 ] [ 1.; 1.; 1. ] in
  let full = float_tensor [ 2; 3 ] [ 1.; 1.; 1.; 1.; 1.; 1. ] in
  let run name t =
    dispatch_print ~target:"torch.ops.aten.layer_norm.default"
      ~bindings:[ ("input", x); (name, t) ]
      ~inputs:
        [
          in_tensor "input"; in_ints "normalized_shape" [ 2; 3 ]; in_tensor name;
        ]
      ~noutputs:1
  in
  run "weight" flat;
  run "bias" flat;
  (* the rank-2 operands the two-axis normalization actually wants *)
  run "weight" full;
  run "bias" full;
  [%expect
    {|
    error: layer_norm weight must be rank-2, got rank-1
    error: layer_norm bias must be rank-2, got rank-1
    tensor f32 [W=2 C=3] {-1.46385, -0.878309, -0.29277, 0.29277, 0.878309, 1.46385}
    tensor f32 [W=2 C=3] {-0.463848, 0.121691, 0.707231, 1.29277, 1.87831, 2.46385} |}]

(* [eps] is added INSIDE the sqrt, so on data whose variance is comparable to it
   the two corpus values are far apart rather than indistinguishable. The
   extents are powers of two so the input, its mean and their differences are
   exact in f32 and the separation is the epsilon alone. *)
let%expect_test "dispatch: layer_norm eps defaults to 1e-5 and is inside sqrt" =
  let x = float_tensor [ 3 ] [ 0.; 0.001953125; 0.00390625 ] in
  let run extra =
    dispatch_print ~target:"torch.ops.aten.layer_norm.default"
      ~bindings:[ ("input", x) ]
      ~inputs:([ in_tensor "input"; in_ints "normalized_shape" [ 3 ] ] @ extra)
      ~noutputs:1
  in
  run [];
  run [ in_float "eps" 1e-5 ];
  run [ in_float "eps" 1e-6 ];
  [%expect
    {|
    tensor f32 [C=3] {-0.551477, 0, 0.551477}
    tensor f32 [C=3] {-0.551477, 0, 0.551477}
    tensor f32 [C=3] {-1.03762, 0, 1.03762} |}]

(* ---- the native_layer_norm.default arm ---------------------------------- *)

(* The DECOMPOSED target shares the whole body above, so what is dispatched here
   is only what differs: the 3-tuple return, the required [eps], and the absent
   [cudnn_enable].

   The node declares three outputs and the bridge builds ONE. That is legal
   rather than an arity bug -- [Verify.requires_exact_outputs] is true only for
   a dynamic [Argument.Tensors] return, so a fixed tuple falls under the
   leading-outputs rule and is verified against the first ATen result alone. The
   graph is printed to show that the missing two leave nothing behind. *)
let%expect_test "dispatch: native_layer_norm.default exposes one output" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 3 ] [ 2.; -1.; 0.5 ] in
  let b = float_tensor [ 3 ] [ 0.25; 0.; -1. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.native_layer_norm.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", b) ]
    ~inputs:
      [
        in_tensor "input";
        in_ints "normalized_shape" [ 3 ];
        in_tensor "weight";
        in_tensor "bias";
        in_float "eps" 1e-6;
      ]
    ~noutputs:3;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [C=3] ->[n0], t2 f32 [C=3] ->[n0]]
    nodes:
      n0: [t3 f32 [W=2 C=3]] =
        layer_norm x=t0 weight=t1 bias=t2 params={dims=[C]; eps=1e-06}
    outputs: [t3 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {-2.19949, 0, -0.387628, -2.19949, 0, -0.387628} |}]

(* [eps] has NO schema default here, unlike the functional overload's 1e-05, so
   its absence is a malformed node. Reading it with a default would substitute a
   number the model never supplied -- and not even the one the corpus uses,
   which is 1e-06 in all 148 occurrences. *)
let%expect_test "dispatch: native_layer_norm requires eps" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let run ?(normalized = [ 3 ]) extra =
    dispatch_print ~target:"torch.ops.aten.native_layer_norm.default"
      ~bindings:[ ("input", x) ]
      ~inputs:
        ([ in_tensor "input"; in_ints "normalized_shape" normalized ] @ extra)
      ~noutputs:3
  in
  run [];
  run [ in_float "eps" 1e-6 ];
  run [ in_int "eps" 0 ];
  (* The shared validation still runs, and names THIS target rather than the
     one the body is shared with. *)
  run ~normalized:[ 2 ] [ in_float "eps" 1e-6 ];
  run ~normalized:[ 2; 3; 3 ] [ in_float "eps" 1e-6 ];
  [%expect
    {|
    error: missing required argument "eps"
    tensor f32 [W=2 C=3] {-1.22474, 0, 1.22474, -1.22474, 0, 1.22474}
    error: argument "eps": expected float, got Int
    error: native_layer_norm: normalized_shape [2] does not match the input's trailing extents [3]
    error: native_layer_norm: normalized_shape has 3 entries, outside [1, 2] for this rank |}]

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

(* The GRAPH is printed here and not above, because this is where the arm used
   to materialize a ones tensor and pass it as a required operand. It no longer
   does: [Graph_ir]'s [Rms_norm] carries [weight : Tensor_ref.t option] and
   Native4D reads the option (lower.ml:293-299), so synthesizing a constant made
   this path build a structurally different graph from [Native_interp]'s for the
   same node and left that arm unreachable from the bridge.

   That is a visible behaviour change, and the values beneath it are the claim
   it must not hide: they are byte-identical to the weighted case above, whose
   weight is all ones -- which is what an absent weight means. *)
let%expect_test "dispatch: rms_norm with no weight builds no weight operand" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.rms_norm.default"
    ~bindings:[ ("input", x) ]
    ~inputs:
      [
        in_tensor "input"; in_ints "normalized_shape" [ 3 ]; in_float "eps" 1e-5;
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] =
        rms_norm x=t0 weight=none params={dims=[C]; eps=1e-05}
    outputs: [t1 f32 [W=2 C=3] <-n0]
    tensor f32 [W=2 C=3] {0.46291, 0.925819, 1.38873, 0.789542, 0.986927, 1.18431} |}]

(* Neither of these was refused before. The first normalized over axes whose
   extents are not the ones the model named; the second reached [trailing_axes]
   with [k > rank], where [List.filteri]'s negative lower bound keeps every
   element, and normalized over the whole tensor. *)
let%expect_test
    "dispatch: rms_norm validates normalized_shape, not just its length" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let bad normalized_shape =
    dispatch_print ~target:"torch.ops.aten.rms_norm.default"
      ~bindings:[ ("input", x) ]
      ~inputs:[ in_tensor "input"; in_ints "normalized_shape" normalized_shape ]
      ~noutputs:1
  in
  bad [ 2 ];
  bad [ 3; 2 ];
  bad [ 2; 3; 3 ];
  bad [];
  (* and the shapes that DO match still lower *)
  bad [ 3 ];
  bad [ 2; 3 ];
  [%expect
    {|
    error: rms_norm: normalized_shape [2] does not match the input's trailing extents [3]
    error: rms_norm: normalized_shape [3, 2] does not match the input's trailing extents [2, 3]
    error: rms_norm: normalized_shape has 3 entries, outside [1, 2] for this rank
    error: rms_norm: normalized_shape has 0 entries, outside [1, 2] for this rank
    tensor f32 [W=2 C=3] {0.46291, 0.92582, 1.38873, 0.789542, 0.986928, 1.18431}
    tensor f32 [W=2 C=3] {0.256776, 0.513553, 0.770329, 1.02711, 1.28388, 1.54066} |}]

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

(* Same hole as unbind.int, on the other arm that resolves a decoded dim: before
   commit 0, [dims.(1) = 5] escaped as an uncaught [Invalid_argument] rather
   than the typed row. *)
let%expect_test "dispatch: permute.default rejects an out-of-range dim" =
  let x = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.permute.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dims" [ 0; 5 ] ]
    ~noutputs:1;
  [%expect {| error: permute.default: invalid dimension 5 for rank 2 |}]

(* A [dims] list whose length disagrees with the operand's rank is a distinct
   fault from any single entry being out of range. *)
let%expect_test "dispatch: permute.default rejects a wrong-length dims list" =
  let x = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.permute.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dims" [ 0 ] ]
    ~noutputs:1;
  [%expect {| error: permute.default: expected 2 dims, got 1 |}]

(* ---- transpose.int --------------------------------------------------------- *)

(* Coordinate-coded values (v = 100*i + j, generalized per rank below) so a
   wrong swap fails on the printed VALUES, not merely on the output shape --
   the mutation table's "wrong pair swapped" case needs asymmetric extents for
   the same reason. *)
let%expect_test "dispatch: transpose.int rank-2 (0,1)" =
  (* [2,3], v[i,j] = 100*i+j. transpose(0,1) -> [3,2], out[j,i] = in[i,j]. *)
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 100.; 101.; 102. ] in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 0; in_int "dim1" 1 ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=3 C=2] {0, 100, 1, 101, 2, 102} |}]

let%expect_test "dispatch: transpose.int rank-3 (1,2)" =
  (* [2,3,4], v[i,j,k] = 100*i+10*j+k. transpose(1,2) swaps the middle two
     axes: out[i,k,j] = in[i,j,k], shape [2,4,3]. *)
  let vals =
    List.concat_map
      (fun i ->
        List.concat_map
          (fun j -> List.init 4 (fun k -> (100 * i) + (10 * j) + k))
          [ 0; 1; 2 ])
      [ 0; 1 ]
  in
  let x = float_tensor [ 2; 3; 4 ] (List.map float_of_int vals) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 1; in_int "dim1" 2 ]
    ~noutputs:1;
  [%expect
    {|
    tensor f32 [H=2 W=4 C=3] {0, 10, 20, 1, 11, 21, 2, 12, ...} |}]

let%expect_test "dispatch: transpose.int rank-4 negative dims (-1,-2)" =
  (* [1,2,3,4] (D,H,W,C), asymmetric extents 3x4 on the swapped axes, v =
     100*h + w. dims -1,-2 normalize to rank-1=3 (C) and rank-2=2 (W), the
     same pair positive dims 2,3 would name. *)
  let vals =
    List.concat_map (fun h -> List.init 4 (fun w -> (100 * h) + w)) [ 0; 1; 2 ]
  in
  let x = float_tensor [ 1; 2; 3; 4 ] (List.map float_of_int vals) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" (-1); in_int "dim1" (-2) ]
    ~noutputs:1;
  [%expect
    {|
    tensor f32 [H=2 W=4 C=3] {0, 100, 200, 1, 101, 201, 2, 102, ...} |}]

let%expect_test "dispatch: transpose.int rank-4 mixed dims (0,-1)" =
  let x = float_tensor [ 2; 3; 4; 5 ] (List.init 120 float_of_int) in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 0; in_int "dim1" (-1) ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [D=2 H=3 W=4 C=5] ->[n0]]
    nodes:
      n0: [t1 f32 [D=5 H=3 W=4 C=2]] = permute x=t0 perm=[D<-C, C<-D]
    outputs: [t1 f32 [D=5 H=3 W=4 C=2] <-n0]
    tensor f32 [D=5 H=3 W=4 C=2] {0, 60, 5, 65, 10, 70, 15, 75, ...} |}]

(* Equal dims: a real identity transpose, not special-cased away. *)
let%expect_test "dispatch: transpose.int equal dims is the identity" =
  let x = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 1; in_int "dim1" 1 ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=3 C=4] {0, 1, 2, 3, 4, 5, 6, 7, ...} |}]

(* Duplicate dims after normalization -- (1, -3) on a rank-4 tensor both name
   axis 1 -- is the same case as literally-equal dims: a well-defined identity
   swap, not a rejection. *)
let%expect_test
    "dispatch: transpose.int duplicate dims after normalization is the identity"
    =
  let x = float_tensor [ 1; 3; 4; 5 ] (List.init 60 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 1; in_int "dim1" (-3) ]
    ~noutputs:1;
  [%expect {| tensor f32 [H=3 W=4 C=5] {0, 1, 2, 3, 4, 5, 6, 7, ...} |}]

let%expect_test "dispatch: transpose.int rejects an out-of-range dim" =
  let x = float_tensor [ 3; 4 ] (List.init 12 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 0; in_int "dim1" 5 ]
    ~noutputs:1;
  [%expect {| error: transpose.int: invalid dimension 5 for rank 2 |}]

(* Rank 6, the frame's full width: transpose the outermost pair. *)
let%expect_test "dispatch: transpose.int rank-6 (0,1)" =
  let x = float_tensor [ 2; 3; 1; 1; 1; 1 ] (List.init 6 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.transpose.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim0" 0; in_int "dim1" 1 ]
    ~noutputs:1;
  [%expect {| tensor f32 [N=3 T=2 D=1 H=1 W=1 C=1] {0, 3, 1, 4, 2, 5} |}]

(* ---- ATen-vs-native on the serialized-scalar path ------------------------- *)

(* The bridge arms above pin the native compute against hand-derived values, and
   the generated walks (test/native_walk_test.ml) compare whole ops against real
   ATen — but neither reaches THIS case. A compile-time scalar in a Tensor slot
   (`aten.add.Tensor(x, 3)`, which is how MobileNet-v3's hardsigmoid is
   serialised) is skipped by [bin/pt2_spec_gen] when it writes node fixtures,
   because a Tensor-typed param holding an [Argument.Int] is not something an
   op-spec can express; and the walk generator synthesises tensor arguments, so
   it never produces one either.

   [Interp_verify.dispatch ~verify:true] is the dual path: it runs the node
   through [Interp_dispatch] (real ATen, which materialises the scalar with
   [full_like]) AND through [Op_bridge] + [Eval_direct] (native, which routes it
   into an op parameter), then compares element-wise with [Verify.verify_node].
   Silence means the two agree. *)
let verify_print ~target ~bindings ~inputs =
  let env = List.fold_left (fun m (k, t) -> Sm.add k t m) Sm.empty bindings in
  let node =
    PT.Node.make target inputs [ targ "out0" ] Sm.empty None (Some "test")
  in
  match
    Interp_verify.dispatch ~verify:true ~ppf:Format.std_formatter env node
  with
  | Error e ->
      Format.printf "dispatch error: %a@." Interp_verify.pp_interp_error
        (Err.Error.kind e)
  | Ok _ -> print_string "aten and native agree\n"

let%expect_test "verify: add.Tensor with a serialized Int scalar" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  verify_print ~target:"torch.ops.aten.add.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3 ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: div.Tensor with a serialized Int scalar" =
  let a = float_tensor [ 2; 3 ] [ 0.; 1.; 3.; 6.; 9.; 12. ] in
  verify_print ~target:"torch.ops.aten.div.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 6 ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: div.Tensor with a serialized Float scalar" =
  let a = float_tensor [ 2; 3 ] [ 0.; 1.; 3.; 6.; 9.; 12. ] in
  verify_print ~target:"torch.ops.aten.div.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "other" 0.1 ];
  [%expect {| aten and native agree |}]

(* op3-impl.md F6: [sub.Tensor]'s scalar spelling cannot be expressed as an
   [Op_spec] fixture (there is no [Arg_value] constructor for "a Tensor-typed
   slot the exporter wrote as a bare scalar"), so this is the only route to
   real ATen evidence for it. [x - s] legalizes to [x + (-s)] on the native
   side (op_bridge.ml, native_interp.ml) -- if that negation were ever wrong
   (e.g. [add_scalar s] instead of [add_scalar (-.s)]), this would print a
   mismatch rather than agreement. *)
let%expect_test "verify: sub.Tensor with a serialized Int scalar" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  verify_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3 ];
  [%expect {| aten and native agree |}]

let%expect_test "verify: sub.Tensor with a serialized Float scalar" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  verify_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "other" 0.5 ];
  [%expect {| aten and native agree |}]

(* Broadcasting a [1,3] against the [2,3] self, then a [2,1]: two different
   axes carry the extent-1 side, checked against real ATen rather than a
   hand-derived expectation. *)
let%expect_test "verify: sub.Tensor broadcasts [1,3] and [2,1] against [2,3]" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  let row = float_tensor [ 1; 3 ] [ 1.; 2.; 3. ] in
  let col = float_tensor [ 2; 1 ] [ 10.; 20. ] in
  verify_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a); ("other", row) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ];
  verify_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a); ("other", col) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

(* [other] of a shape that cannot broadcast against [self] at all: the
   existing native [`Broadcast] row, at the native boundary -- not a new
   importer check, since [sub.Tensor] adds no shape rule of its own. *)
let%expect_test "dispatch: sub.Tensor rejects an incompatible other shape" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  let b = float_tensor [ 2; 4 ] (List.init 8 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect {| error: incompatible broadcast extents on axis C: 3 vs 4 |}]

(* The whole hardsigmoid chain, each node checked against ATen in turn: the
   scalar add, both one-sided clamps, and the scalar divide. *)
let%expect_test "verify: MobileNet-v3 hardsigmoid chain against ATen" =
  let a = float_tensor [ 2; 3 ] [ -4.; -3.; 0.; 3.; 4.; 10. ] in
  verify_print ~target:"torch.ops.aten.add.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3 ];
  verify_print ~target:"torch.ops.aten.clamp.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "min" 0 ];
  verify_print ~target:"torch.ops.aten.clamp.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_none "min"; in_int "max" 6 ];
  verify_print ~target:"torch.ops.aten.div.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 6 ];
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

(* alpha is neither implemented nor silently dropped: [self + alpha * other] is
   not what the arm computes, so a non-default alpha must be refused on both the
   tensor and the scalar path, and for sub as well as add. *)
let%expect_test "dispatch: a non-default alpha is rejected, not ignored" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 2; 3 ] [ 1.; 1.; 1.; 1.; 1.; 1. ] in
  dispatch_print ~target:"torch.ops.aten.add.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other"; in_int "alpha" 2 ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.add.Tensor"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "other" 3; in_float "alpha" 2.5 ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other"; in_int "alpha" 2 ]
    ~noutputs:1;
  (* alpha=1 is the default and must still go through *)
  dispatch_print ~target:"torch.ops.aten.add.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other"; in_int "alpha" 1 ]
    ~noutputs:1;
  [%expect
    {|
    error: alpha=2 is not supported (only 1)
    error: alpha=2.5 is not supported (only 1)
    error: alpha=2 is not supported (only 1)
    tensor f32 [W=2 C=3] {2, 3, 4, 5, 6, 7} |}]

(* A requested memory_format is a layout change clone does not perform. *)
let%expect_test "dispatch: clone with a memory_format is rejected" =
  let a = float_tensor [ 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.clone.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "memory_format" 0 ]
    ~noutputs:1;
  dispatch_print ~target:"torch.ops.aten.clone.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_none "memory_format" ]
    ~noutputs:1;
  [%expect
    {|
    error: clone: memory_format is not supported
    tensor f32 [W=2 C=2] {1, 2, 3, 4} |}]

(* clamp with neither bound is refused where the node is built, mirroring
   ATen's own meta-function check. *)
let%expect_test "dispatch: clamp with no bounds is rejected" =
  let a = float_tensor [ 2; 2 ] [ -1.; 0.; 3.; 9. ] in
  dispatch_print ~target:"torch.ops.aten.clamp.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_none "min"; in_none "max" ]
    ~noutputs:1;
  [%expect {| error: clamp: at least one of 'min' or 'max' must be given |}]

(* Hardtanh's bounds are schema Scalars with defaults: they may be omitted
   entirely, or arrive as Int rather than Float. *)
let%expect_test "dispatch: hardtanh bound spellings" =
  let a = float_tensor [ 2; 3 ] [ -3.; -1.; 0.; 0.5; 2.5; 7. ] in
  (* omitted -> schema defaults (-1, 1) *)
  dispatch_print ~target:"torch.ops.aten.hardtanh.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  (* Int bounds, as MobileNet-v3-style graphs spell them *)
  dispatch_print ~target:"torch.ops.aten.hardtanh.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "min_val" 0; in_int "max_val" 6 ]
    ~noutputs:1;
  (* Float bounds, as MobileNet-v2 spells them *)
  dispatch_print ~target:"torch.ops.aten.hardtanh.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_float "min_val" 0.; in_float "max_val" 6. ]
    ~noutputs:1;
  [%expect
    {|
    tensor f32 [W=2 C=3] {-1, -1, 0, 0.5, 1, 1}
    tensor f32 [W=2 C=3] {0, 0, 0, 0.5, 2.5, 6}
    tensor f32 [W=2 C=3] {0, 0, 0, 0.5, 2.5, 6} |}]

(* ---- Group 5 activations: silu, hardsigmoid, hardswish (op5-impl) -------- *)

let%expect_test "dispatch: silu.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.silu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {-0.0148357, -0.18877, 0.31123, 5.98516} |}]

let%expect_test "dispatch: hardsigmoid.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.hardsigmoid.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {0, 0.416667, 0.583333, 1} |}]

let%expect_test "dispatch: hardswish.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ -6.; -0.5; 0.5; 6. ] in
  dispatch_print ~target:"torch.ops.aten.hardswish.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {-0, -0.208333, 0.291667, 6} |}]

(* Real-ATen agreement, functional AND in-place spellings both, over the
   float32-threshold-neighbourhood fixture (op5.md). [verify_print] runs the
   node through BOTH [Interp_dispatch] (real ATen) and [Op_bridge] +
   [Eval_direct] (native) and compares element-wise -- silence means
   agreement. The in-place spellings are the coverage that matters: efficientnet
   serialises [silu_], never [silu] (op5-impl F1), and getting this to agree
   required fixing [Interp_verify.dispatch] to capture native's read of [self]
   BEFORE the ATen in-place call mutates it through the same handle -- the
   previous order silently compared native's output against ATen's INPUT
   already overwritten with ATen's own output, invisible for relu_/hardtanh_
   only because clamping is idempotent (op5-impl). *)
let activation_fixture =
  [
    -1e4;
    -6.;
    -3.0000005;
    -3.;
    -2.9999995;
    -0.5;
    -0.;
    0.;
    0.5;
    2.9999995;
    3.;
    3.0000005;
    6.;
    1e4;
  ]

let%expect_test "verify: silu against real ATen, functional and in-place" =
  let a = float_tensor [ 2; 7 ] activation_fixture in
  verify_print ~target:"torch.ops.aten.silu.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  verify_print ~target:"torch.ops.aten.silu_.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: hardsigmoid against real ATen, functional and in-place"
    =
  let a = float_tensor [ 2; 7 ] activation_fixture in
  verify_print ~target:"torch.ops.aten.hardsigmoid.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  verify_print ~target:"torch.ops.aten.hardsigmoid_.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: hardswish against real ATen, functional and in-place" =
  let a = float_tensor [ 2; 7 ] activation_fixture in
  verify_print ~target:"torch.ops.aten.hardswish.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  verify_print ~target:"torch.ops.aten.hardswish_.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

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
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=3 C=2]] = reshape x=t0 params={shape=[W=3 C=2]}
    outputs: [t1 f32 [W=3 C=2] <-n0]
    tensor f32 [W=3 C=2] {0, 1, 2, 3, 4, 5} |}]

(* op3-impl.md F1, the zero-guard half: [Aten_shape.resolve_view_size]
   validates every non-[-1] entry through [Dim.extent_checked] before any
   division runs, so [0] alongside [-1] is refused rather than dividing by the
   zero it would otherwise fold [known] to. Before commit 1 the bridge's own
   [resolve_view_size] had NO zero guard at all and raised [Division_by_zero]
   here -- unlike [Native_interp], which already special-cased the [-1] case
   (commit 9390ae6). *)
let%expect_test "dispatch: view.default rejects a target sizing both 0 and -1" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_view_both ~x ~size:[ 0; -1 ];
  [%expect
    {|
    error: extent must be >= 1, got 0
    error: extent must be >= 1, got 0 |}]

(* The oversized-TARGET case, not the source: [Aten_shape.resolve_view_size]'s
   divide-first fold rejects it (its [known] already exceeds the source's tiny
   [numel] partway through) before [Aten_shape.of_aten] ever converts a
   resolved list into a native shape, so no large allocation happens -- unlike
   an oversized SOURCE, which would have to reach the bridge through a real
   (unconstructible-as-a-fixture) ATen handle; see [Tensor_bridge.of_aten]'s
   own preflight, tested at the primitive level in test/native/vec6_test.ml. *)
let%expect_test "dispatch: view.default rejects a target past the numel ceiling"
    =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  (* numel 6 either way: a small, ordinary target still lowers, establishing
     that the rejection below is about the OVERSIZED case and not a general
     regression. *)
  dispatch_view_both ~x ~size:[ 2; 3 ];
  dispatch_view_both ~x ~size:[ 65536; 65536 ];
  [%expect
    {|
    tensor f32 [W=2 C=3] {0, 1, 2, 3, 4, 5}
    tensor f32 [W=2 C=3] {0, 1, 2, 3, 4, 5}
    error: view size [65536, 65536] does not match 6 elements
    error: view size [65536, 65536] does not match 6 elements |}]

(* Rank-changing targets, under both overloads: rank-increasing (a flat
   6-vector split into [2,3]) and rank-decreasing (a rank-3 volume flattened
   to [24]). Row-major values, so a wrong element order fails visibly rather
   than by shape alone. *)
let%expect_test
    "dispatch: view/_unsafe_view rank-increasing and rank-decreasing targets" =
  let flat = float_tensor [ 6 ] (List.init 6 float_of_int) in
  dispatch_view_both ~x:flat ~size:[ 2; 3 ];
  let vol = float_tensor [ 2; 3; 4 ] (List.init 24 float_of_int) in
  dispatch_view_both ~x:vol ~size:[ 24 ];
  [%expect
    {|
    tensor f32 [W=2 C=3] {0, 1, 2, 3, 4, 5}
    tensor f32 [W=2 C=3] {0, 1, 2, 3, 4, 5}
    tensor f32 [C=24] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [C=24] {0, 1, 2, 3, 4, 5, 6, 7, ...} |}]

(* One inferred [-1] in leading, middle and trailing position, on a source
   with no repeated extents (2,3,4) so a wrong resolved position is visible in
   the printed shape, not just the values. *)
let%expect_test
    "dispatch: view/_unsafe_view resolve a leading, middle and trailing -1" =
  let vol = float_tensor [ 2; 3; 4 ] (List.init 24 float_of_int) in
  dispatch_view_both ~x:vol ~size:[ -1; 2; 3 ];
  dispatch_view_both ~x:vol ~size:[ 3; -1; 2 ];
  dispatch_view_both ~x:vol ~size:[ 3; 2; -1 ];
  [%expect
    {|
    tensor f32 [H=4 W=2 C=3] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [H=4 W=2 C=3] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [H=3 W=4 C=2] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [H=3 W=4 C=2] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [H=3 W=2 C=4] {0, 1, 2, 3, 4, 5, 6, 7, ...}
    tensor f32 [H=3 W=2 C=4] {0, 1, 2, 3, 4, 5, 6, 7, ...} |}]

(* A rank-6 target is the boundary: exactly 6 axes is accepted (the frame's
   width), rank-7 is `Rank_out_of_range`. *)
let%expect_test "dispatch: view/_unsafe_view accepts rank 6, rejects rank 7" =
  let flat = float_tensor [ 6 ] (List.init 6 float_of_int) in
  dispatch_view_both ~x:flat ~size:[ 1; 1; 1; 1; 1; 6 ];
  dispatch_view_both ~x:flat ~size:[ 1; 1; 1; 1; 1; 1; 6 ];
  [%expect
    {|
    tensor f32 [C=6] {0, 1, 2, 3, 4, 5}
    tensor f32 [C=6] {0, 1, 2, 3, 4, 5}
    error: rank 7 out of [0, 6]
    error: rank 7 out of [0, 6] |}]

(* op3-impl.md F1's three accepted-then-wrong-answer holes, now three typed
   rejections instead: more than one inferred dim, a target whose declared
   product disagrees with the source's element count, and a target that does
   not evenly divide it. *)
let%expect_test "dispatch: view/_unsafe_view reject F1's three invalid targets"
    =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_view_both ~x ~size:[ -1; -1 ];
  dispatch_view_both ~x ~size:[ 4; 2 ];
  dispatch_view_both ~x ~size:[ 4; -1 ];
  [%expect
    {|
    error: view size [-1, -1] has more than one inferred (-1) dimension
    error: view size [-1, -1] has more than one inferred (-1) dimension
    error: view size [4, 2] does not match 6 elements
    error: view size [4, 2] does not match 6 elements
    error: view size [4, -1] does not divide 6 elements
    error: view size [4, -1] does not divide 6 elements |}]

(* A declared 0 with no [-1] present -- a distinct case from the
   [-1]-alongside-[0] one above, caught by [Dim.extent_checked] before the
   divisibility check that follows it even runs. *)
let%expect_test "dispatch: view/_unsafe_view reject a target with a zero extent"
    =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_view_both ~x ~size:[ 0; 6 ];
  [%expect
    {|
    error: extent must be >= 1, got 0
    error: extent must be >= 1, got 0 |}]

(* The rejected witness cannot be the PRODUCT [prefix * extent]: [Dim.extent]
   bounds an extent only BELOW, so on this (63-bit) backend a single axis can
   sit near [max_int], and [prefix * extent] here would itself overflow
   [int64] -- reintroducing exactly the wrap this design exists to prevent.
   Native-only on purpose (unlike every other numel fixture in this group):
   the constant is not representable in js_of_ocaml's 32-bit [int], and this
   file's stanza is the one inline suite with plain [(inline_tests)], no [js]
   mode (test/dune). *)
let%expect_test
    "vec6: numel_bounded reports the witness pair, never their \
     (unrepresentable) product" =
  let s = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1_073_741_824 ~c:max_int in
  (match Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel s with
  | Ok n -> Format.printf "Ok %Ld@." n
  | Error e -> (
      match Err.Error.kind e with
      | `Numel_over_limit b -> Format.printf "Error %a@." Vec6.Numel_bound.pp b));
  [%expect
    {| Error axis C: 1073741824 elements so far times extent 4611686018427387903 reaches the maximum of 2147483648 |}]

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

(* --- Interp_decode.output_names ---

   Five call sites used to open-code a [Argument.Tensor]-only filter, which
   yields ZERO names for a Tensor[]-returning node -- a report with no output
   lines reads as "nothing to say" rather than as the omission it is. This pins
   the helper's own behaviour (flattening order, None handling) independently of
   any CLI formatting that consumes it. *)

let names_of outputs =
  let node = PT.Node.make "t" [] outputs Sm.empty None (Some "test") in
  print_endline (String.concat "," (Interp_decode.output_names node))

let%expect_test "output_names: flattens Tensor[] in place, skips None" =
  (* single tensor: one name *)
  names_of [ targ "a" ];
  [%expect "a"];
  (* fixed tuple: N separate outputs, in order *)
  names_of [ targ "a"; targ "b"; targ "c" ];
  [%expect "a,b,c"];
  (* Tensor[]: ONE output carrying N names -- the shape unbind serializes as *)
  names_of
    [
      PT.Argument.Tensors
        [
          PT.TensorArgument.make "x";
          PT.TensorArgument.make "y";
          PT.TensorArgument.make "z";
        ];
    ];
  [%expect "x,y,z"];
  (* a dead output stays skipped, and flattening happens in position *)
  names_of
    [
      targ "a";
      PT.Argument.None false;
      PT.Argument.Tensors
        [ PT.TensorArgument.make "x"; PT.TensorArgument.make "y" ];
      targ "b";
    ];
  [%expect "a,x,y,b"];
  (* an empty Tensor[] contributes nothing, but is not an error *)
  names_of [ PT.Argument.Tensors [] ];
  [%expect ""]

(* --- Interp handle ownership ---

   [Interp.top_predictions] and [Interp.argmax] read their results through
   [Aten_tensor.data] / [item_int], neither of which manages a handle:
   [data]'s finaliser only anchors an already-registered handle for the
   Bigarray's lifetime, it never calls atc_free. So every op result these two
   allocate has to be piped through [Aten_tensor.manage] at the call site or it
   leaks for the process's lifetime. Counting handles is the only way to see
   that -- the values are correct either way. *)

let collect () =
  Gc.full_major ();
  Gc.full_major ()

let%expect_test "top_predictions and argmax leave no live tensors" =
  collect ();
  let base = T.live_count () in
  let logits = float_tensor [ 1; 4 ] [ 0.5; 3.0; 1.0; 2.0 ] in
  (* Run in an inner scope so nothing but [logits] is still referenced when the
     collection below runs. *)
  (match
     ( Interp.top_predictions logits 2 |> Err.payload,
       Interp.argmax logits |> Err.payload )
   with
  | Ok top, Ok am ->
      Printf.printf "argmax=%d top=%s\n" am
        (String.concat ","
           (List.map (fun (i, p) -> Printf.sprintf "%d:%.3f" i p) top))
  | _ -> print_endline "unexpected error");
  collect ();
  (* [logits] is still live and still managed: +1, not +0. *)
  Printf.printf "after gc: +%d\n" (T.live_count () - base);
  ignore (Sys.opaque_identity logits);
  [%expect {|
    argmax=1 top=1:0.631,3:0.232
    after gc: +1 |}]

(* --- Interp_dispatch: the Tensor[] output shape --------------------------

   A Tensor[] return serializes as ONE output of kind Tensor[] carrying every
   result name, not as N separate outputs, so [bind_tensor_list] handles it
   instead of the positional [bind_many]. These drive the generated arm through
   hand-built nodes, because the fixture format cannot express a wrong output
   shape (Aten_spec_run synthesizes outputs, so the counts always agree). *)

let pp_dispatch_error ppf = function
  | #Interp_decode.error as e -> Interp_decode.pp_error ppf e
  | `Aten_runtime_failure (op, st) ->
      Fmt.pf ppf "ATen op %s failed with status %d" op st
  | `Unhandled_op target -> Fmt.pf ppf "unhandled op %S" target

(* Dispatch a one-node unbind graph and print each bound name's values. Reads go
   through [materialize_for_raw_read]: unbind returns views. *)
let unbind_dispatch ~inputs ~outputs ~self =
  let env = Sm.add "self" self Sm.empty in
  let node =
    PT.Node.make "torch.ops.aten.unbind.int"
      (PT.NamedArgument.make "self" (targ "self") None :: inputs)
      outputs Sm.empty None (Some "test")
  in
  match Interp_dispatch.dispatch env node |> Err.payload with
  | Error e -> Format.printf "Error: %a@." pp_dispatch_error e
  | Ok env' ->
      List.iter
        (fun name ->
          Format.printf "%s = %a@." name
            (Core.Pretty.option_or ~none:"<unbound>" (fun ppf t ->
                 Aten_tensor.pp_float32 ppf
                   (Option.get
                      (Aten_tensor.as_float32
                         (Aten_tensor.materialize_for_raw_read t)))))
            (Sm.find_opt name env'))
        (Interp_decode.output_names node)

let tensors names =
  [ PT.Argument.Tensors (List.map PT.TensorArgument.make names) ]

(* The real exported node omits [dim] entirely — the schema default 0 is applied
   by the generated decoder, so this is the shape that actually has to work. *)
let%expect_test "dispatch: unbind.int with dim absent binds every name" =
  unbind_dispatch ~inputs:[]
    ~outputs:(tensors [ "a"; "b"; "c" ])
    ~self:(float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect {|
    a = [0; 1]
    b = [2; 3]
    c = [4; 5] |}]

let%expect_test "dispatch: unbind.int with a negative dim" =
  unbind_dispatch
    ~inputs:[ in_int "dim" (-1) ]
    ~outputs:(tensors [ "a"; "b"; "c" ])
    ~self:(float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect {|
    a = [0; 3]
    b = [1; 4]
    c = [2; 5] |}]

let%expect_test "dispatch: unbind.int of a zero-length dim binds nothing" =
  unbind_dispatch
    ~inputs:[ in_int "dim" 0 ]
    ~outputs:(tensors [])
    ~self:(Aten_tensor.create [ 0; 2 ]);
  [%expect {| |}]

(* A fixed tuple's output shape must not be accepted here: zipping [Tensors]
   against separate [Tensor] outputs would leave SSA names unbound. *)
let%expect_test "dispatch: unbind.int rejects the fixed-tuple output shape" =
  unbind_dispatch ~inputs:[]
    ~outputs:[ targ "a"; targ "b"; targ "c" ]
    ~self:(float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect
    {| Error: expected one Tensor[] output, got [Tensor; Tensor; Tensor] |}]

(* The arity check rides on the pairing it guards (Err.List.map2), so it cannot
   drift from it. Drop a name and it fires with both counts. *)
let%expect_test "dispatch: unbind.int reports a name/tensor count mismatch" =
  unbind_dispatch ~inputs:[]
    ~outputs:(tensors [ "a"; "b" ])
    ~self:(float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect
    {| Error: tensor-list output arity: 2 serialized names, 3 tensors returned |}]

let%expect_test "pp: unbind.int config" =
  Aten_op_config.find "torch.ops.aten.unbind.int"
  |> Format.printf "%a@."
       (Core.Pretty.option_or ~none:"not found" Aten_op_config.pp);
  [%expect {| torch.ops.aten.unbind.int (Tensor self, Int dim=0) -> T[] |}]

(* --- the native side of unbind ------------------------------------------- *)

(* [dispatch_print] is reusable here even though it synthesises a fixed-tuple
   output shape: the bridge arm reads [self] and [dim] only, never the node's
   outputs. The values are hand-derived, so this does not lean on ATen. *)
let%expect_test "dispatch: unbind.int slices along the leading dim" =
  (* [3,2] right-aligns to [W=3 C=2]; dim 0 is W, so each slice drops W and the
     survivors re-pack to [C=2]. *)
  dispatch_print ~target:"torch.ops.aten.unbind.int"
    ~bindings:[ ("self", float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:0;
  [%expect
    {|
    tensor f32 [C=2] {0, 1}
    tensor f32 [C=2] {2, 3}
    tensor f32 [C=2] {4, 5} |}]

(* dim=-1 on a rank-2 input is C. Each slice reads a COLUMN, so the values are
   strided rather than contiguous — the case a naive flat read gets wrong. *)
let%expect_test "dispatch: unbind.int with a negative dim" =
  dispatch_print ~target:"torch.ops.aten.unbind.int"
    ~bindings:[ ("self", float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]) ]
    ~inputs:[ in_tensor "self"; in_int "dim" (-1) ]
    ~noutputs:0;
  [%expect
    {|
    tensor f32 [C=2] {0, 3}
    tensor f32 [C=2] {1, 4}
    tensor f32 [C=2] {2, 5} |}]

(* [Aten_shape.axis_of_dim] asserts its range and raises, so the arm checks
   first and reports a typed row. The ORIGINAL dim is echoed, not the
   normalised one. *)
let%expect_test "dispatch: unbind.int rejects an out-of-range dim" =
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.unbind.int"
        ~bindings:[ ("self", float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]) ]
        ~inputs:[ in_tensor "self"; in_int "dim" d ]
        ~noutputs:0)
    [ 2; -3 ];
  [%expect
    {|
    error: unbind.int: invalid dimension 2 for rank 2
    error: unbind.int: invalid dimension -3 for rank 2 |}]

(* The engine computes in f32 and [Graph_builder] gives every op output F32, so
   an i64 operand would silently become an f32 result. [Tensor_bridge.of_aten]
   accepts i64, so the refusal has to be the arm's. *)
let%expect_test "dispatch: unbind.int rejects a non-f32 input" =
  dispatch_print ~target:"torch.ops.aten.unbind.int"
    ~bindings:
      [ ("self", Aten_tensor.create ~dtype:Aten_scalar_type.Long [ 3; 2 ]) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:0;
  [%expect {| error: self: the native engine computes in f32, got Long |}]

(* The real oracle: ATen runs the op, the native side runs [Graph_ir.Unbind]
   through [Eval_direct], and [Verify.verify_node] compares EVERY slice —
   exactly, because a single [Argument.Tensors] output has no dead-output story
   and a bridge returning fewer would otherwise report a match.

   [verify_print] cannot drive this: it synthesises one [Argument.Tensor]
   output, which [bind_tensor_list] rejects. Silence is agreement. *)
let verify_unbind ~inputs ~outputs ~self =
  let env = Sm.add "self" self Sm.empty in
  let node =
    PT.Node.make "torch.ops.aten.unbind.int"
      (PT.NamedArgument.make "self" (targ "self") None :: inputs)
      outputs Sm.empty None (Some "test")
  in
  match
    Interp_verify.dispatch ~verify:true ~ppf:Format.std_formatter env node
  with
  | Error e ->
      Format.printf "dispatch error: %a@." Interp_verify.pp_interp_error
        (Err.Error.kind e)
  | Ok _ -> print_string "aten and native agree\n"

let%expect_test "verify: unbind.int agrees with ATen on every slice" =
  let self = float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  verify_unbind ~inputs:[] ~outputs:(tensors [ "a"; "b"; "c" ]) ~self;
  (* And along the strided axis, where each slice is a column. *)
  verify_unbind
    ~inputs:[ in_int "dim" (-1) ]
    ~outputs:(tensors [ "a"; "b" ])
    ~self:(float_tensor [ 3; 2 ] [ 0.; 1.; 2.; 3.; 4.; 5. ]);
  [%expect {|
    aten and native agree
    aten and native agree |}]

(* A TRANSPOSED convolution's ATen weight is [Cin, Cout/groups, kH, kW], so its
   output channel count -- and therefore its bias extent -- comes from
   [weight.C * groups], not from [weight.N] as it does for every other affine op.
   [Conv.Convolution.bias_shape] has always encoded that; the shared bias check
   briefly did not consult it, and rejected every valid transposed convolution
   whose input and output channel counts differ.

   Unequal channel counts are the whole point of the fixture: with Cin = Cout the
   two rules agree and the case is invisible. *)
let%expect_test "dispatch: transposed convolution bias uses weight.C * groups" =
  let x = float_tensor [ 1; 2; 2; 2 ] (List.init 8 float_of_int) in
  let w = float_tensor [ 2; 3; 1; 1 ] (List.init 6 float_of_int) in
  let b = float_tensor [ 3 ] [ 1.; 2.; 3. ] in
  dispatch_print ~target:"torch.ops.aten.convolution.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", b) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_tensor "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 0 ];
        in_ints "dilation" [ 1; 1 ];
        in_bool "transposed" true;
        in_ints "output_padding" [ 0; 0 ];
        in_int "groups" 1;
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [H=3 W=2 C=2] {13, 16, 19, 22, 18, 23, 28, 33, ...} |}]

(* Ordinary malformed op-config arguments must be CONTAINED at this boundary --
   the bridge returns [Some (Error _)], never raises. [Dim.extent],
   [Op_config.Pos.of_int] and [Nonneg.of_int] all assert their preconditions,
   and the aggregate bounds added for Group 2 check magnitude only, so every one
   of these still reaches an assertion; what keeps it from escaping is that the
   construction happens inside the arm's exception boundary.

   These four are the cheapest cases that reach a different constructor each:
   groups zero (a zero channel PRODUCT, so [Dim.extent]), stride zero and
   dilation zero ([Pos]), negative padding ([Nonneg]). *)
(* The payload, not the rendering. The point of a structured row is that a
   caller can BRANCH on it -- which is what an undifferentiated
   [`Validation_failure "Op_config.Pos.of_int: not positive"] cannot support,
   and what these arms used to return. Destructuring here is the assertion that
   the fields exist and carry the op, the parameter and the offending value. *)
let config_fault ~target ~inputs bindings =
  let env = List.fold_left (fun m (k, t) -> Sm.add k t m) Sm.empty bindings in
  let node =
    PT.Node.make target inputs [ targ "out0" ] Sm.empty None (Some "test")
  in
  match Op_bridge.dispatch ~aten_env:env node with
  | None -> print_string "no native impl\n"
  | Some (Ok _) -> print_string "accepted\n"
  | Some (Error e) -> (
      match Err.Error.kind e with
      | `Bad_config { Op_config.Bad.op; param; fault } ->
          Format.printf "op=%s param=%a fault=%s@." op Op_config.Bad.pp_param
            param
            (match fault with
            | `Not_positive n -> Printf.sprintf "not_positive %d" n
            | `Negative n -> Printf.sprintf "negative %d" n)
      | other -> Format.printf "OTHER ROW: %a@." Op_bridge.pp_error other)

let%expect_test
    "dispatch: every Group-2 bridge arm reports a structured config fault" =
  let x = float_tensor [ 1; 2; 4; 4 ] (List.init 32 float_of_int) in
  let w = float_tensor [ 2; 2; 1; 1 ] (List.init 4 float_of_int) in
  let conv target extra =
    config_fault ~target
      ~inputs:([ in_tensor "input"; in_tensor "weight" ] @ extra)
      [ ("input", x); ("weight", w) ]
  in
  conv "torch.ops.aten.conv2d.default" [ in_ints "stride" [ 0; 1 ] ];
  conv "torch.ops.aten.conv2d.default" [ in_ints "padding" [ 0; -2 ] ];
  conv "torch.ops.aten.conv2d.padding"
    [ in_string "padding" "same"; in_ints "dilation" [ 3; 0 ] ];
  conv "torch.ops.aten.conv2d.padding"
    [ in_string "padding" "valid"; in_int "groups" 0 ];
  (* [padding] and [output_padding] are DIFFERENT arguments of the same op, so
     the op name cannot disambiguate them -- only the tag can. Both spellings
     appear here for that reason, and they must differ. *)
  conv "torch.ops.aten.convolution.default"
    [
      in_ints "padding" [ 0; -1 ];
      in_bool "transposed" false;
      in_ints "output_padding" [ 0; 0 ];
    ];
  conv "torch.ops.aten.convolution.default"
    [
      in_ints "padding" [ 0; 0 ];
      in_bool "transposed" true;
      in_ints "output_padding" [ 0; -1 ];
    ];
  (* Both COMPONENTS, not just one: they are validated by separate calls, and a
     tag fixed on only one of them looks correct from whichever side is
     tested. *)
  conv "torch.ops.aten.convolution.default"
    [
      in_ints "padding" [ 0; 0 ];
      in_bool "transposed" true;
      in_ints "output_padding" [ -1; 0 ];
    ];
  config_fault ~target:"torch.ops.aten.max_pool2d.default"
    ~inputs:[ in_tensor "self"; in_ints "kernel_size" [ 0; 2 ] ]
    [ ("self", x) ];
  config_fault ~target:"torch.ops.aten.max_pool2d.default"
    ~inputs:
      [
        in_tensor "self";
        in_ints "kernel_size" [ 2; 2 ];
        in_ints "padding" [ -1; 0 ];
      ]
    [ ("self", x) ];
  [%expect
    {|
    op=torch.ops.aten.conv2d.default param=stride fault=not_positive 0
    op=torch.ops.aten.conv2d.default param=padding fault=negative -2
    op=torch.ops.aten.conv2d.padding param=dilation fault=not_positive 0
    op=torch.ops.aten.conv2d.padding param=groups fault=not_positive 0
    op=torch.ops.aten.convolution.default param=padding fault=negative -1
    op=torch.ops.aten.convolution.default param=output_padding fault=negative -1
    op=torch.ops.aten.convolution.default param=output_padding fault=negative -1
    op=torch.ops.aten.max_pool2d.default param=kernel_size fault=not_positive 0
    op=torch.ops.aten.max_pool2d.default param=padding fault=negative -1 |}]

(* The padding MODE, which is model data and was reaching an asserting parser.
   Contained by the arm's [try], but as a [`Validation_failure] string -- where
   the serialized importer returns the offered mode in a typed row. Both now
   report the same thing, through one checked parser, so the accepted set cannot
   drift between them. Destructured, not just rendered. *)
let%expect_test "dispatch: an unsupported conv2d.padding mode is a typed row" =
  let x = float_tensor [ 1; 2; 4; 4 ] (List.init 32 float_of_int) in
  let w = float_tensor [ 2; 2; 1; 1 ] (List.init 4 float_of_int) in
  let mode m =
    let env = Sm.add "input" x (Sm.add "weight" w Sm.empty) in
    let node =
      PT.Node.make "torch.ops.aten.conv2d.padding"
        [ in_tensor "input"; in_tensor "weight"; in_string "padding" m ]
        [ targ "out0" ]
        Sm.empty None (Some "test")
    in
    match Op_bridge.dispatch ~aten_env:env node with
    | None -> print_string "no native impl\n"
    | Some (Ok _) -> Format.printf "%S accepted@." m
    | Some (Error e) -> (
        match Err.Error.kind e with
        | `Unsupported_padding_mode s -> Format.printf "mode=%S refused@." s
        | other -> Format.printf "OTHER ROW: %a@." Op_bridge.pp_error other)
  in
  mode "reflect";
  mode "SAME";
  mode "";
  mode "valid";
  mode "same";
  [%expect
    {|
    mode="reflect" refused
    mode="SAME" refused
    mode="" refused
    "valid" accepted
    "same" accepted |}]

let%expect_test "dispatch: malformed conv2d config is contained, not raised" =
  let x = float_tensor [ 1; 2; 2; 2 ] (List.init 8 float_of_int) in
  let w = float_tensor [ 2; 2; 1; 1 ] (List.init 4 float_of_int) in
  let conv extra =
    dispatch_print ~target:"torch.ops.aten.conv2d.default"
      ~bindings:[ ("input", x); ("weight", w) ]
      ~inputs:([ in_tensor "input"; in_tensor "weight" ] @ extra)
      ~noutputs:1
  in
  conv [ in_int "groups" 0 ];
  conv [ in_ints "stride" [ 0; 1 ] ];
  conv [ in_ints "padding" [ -1; 0 ] ];
  conv [ in_ints "dilation" [ 0; 1 ] ];
  (* and the well-formed node still lowers *)
  conv [ in_int "groups" 1 ];
  [%expect
    {|
    error: torch.ops.aten.conv2d.default: groups must be positive, got 0
    error: torch.ops.aten.conv2d.default: stride must be positive, got 0
    error: torch.ops.aten.conv2d.default: padding must not be negative, got -1
    error: torch.ops.aten.conv2d.default: dilation must be positive, got 0
    tensor f32 [H=2 W=2 C=2] {4, 5, 6, 7, 12, 17, 22, 27} |}]

(* RANK, which right-alignment into the six-axis frame erases. A bias declared
   [1,Cout] arrives indistinguishable from [Cout], so the shared [Graph_shape]
   check passes it -- while ATen refuses it ("expected bias to be
   1-dimensional"). The rank survives only on the ATen tensor. *)
let%expect_test "dispatch: a leading-singleton bias is refused on rank" =
  let x = float_tensor [ 1; 2; 2; 2 ] (List.init 8 float_of_int) in
  let w = float_tensor [ 3; 2; 1; 1 ] (List.init 6 float_of_int) in
  let conv_bias b =
    dispatch_print ~target:"torch.ops.aten.conv2d.default"
      ~bindings:[ ("input", x); ("weight", w); ("bias", b) ]
      ~inputs:[ in_tensor "input"; in_tensor "weight"; in_tensor "bias" ]
      ~noutputs:1
  in
  conv_bias (float_tensor [ 1; 3 ] [ 1.; 2.; 3. ]);
  conv_bias (float_tensor [ 3 ] [ 1.; 2.; 3. ]);
  [%expect
    {|
    error: bias must be rank-1, got rank-2
    tensor f32 [H=3 W=2 C=2] {5, 6, 7, 8, 14, 19, 24, 29, ...} |}]

(* ---- pooling options the native params cannot hold ---------------------- *)

(* Both arms decoded kernel/stride/padding and never read [dilation] or
   [ceil_mode]. [Pool.MaxPool2d.params] has a field for neither, so a serialized
   non-default value was not approximated -- it was DROPPED, and the bridge
   computed an undilated floor-mode pool under the dilated or ceil-mode name.

   The default spellings still pass, which is the half of this that a rejection
   test alone would not show: refusing every export that mentions the argument
   would be its own regression. *)
let%expect_test "dispatch: max_pool2d dilation and ceil_mode are refused" =
  let a =
    float_tensor [ 1; 1; 5; 5 ] (List.init 25 (fun i -> float_of_int i))
  in
  let pool ?dilation ?ceil_mode target =
    dispatch_print ~target
      ~bindings:[ ("self", a) ]
      ~inputs:
        ([ in_tensor "self"; in_ints "kernel_size" [ 2; 2 ] ]
        @ (match dilation with
          | None -> []
          | Some d -> [ in_ints "dilation" d ])
        @
        match ceil_mode with
        | None -> []
        | Some b -> [ in_bool "ceil_mode" b ])
      ~noutputs:(if target = "torch.ops.aten.max_pool2d.default" then 1 else 2)
  in
  let both f =
    List.iter f
      [
        "torch.ops.aten.max_pool2d.default";
        "torch.ops.aten.max_pool2d_with_indices.default";
      ]
  in
  both (fun t -> pool ~dilation:[ 2; 2 ] t);
  both (fun t -> pool ~ceil_mode:true t);
  (* the defaults, written out, still lower *)
  both (fun t -> pool ~dilation:[ 1; 1 ] ~ceil_mode:false t);
  (* A single int is ATen's other legal spelling and normalizes to (1,1); three
     is not, and a value-only check accepted it because no element differed. *)
  both (fun t -> pool ~dilation:[ 1 ] t);
  both (fun t -> pool ~dilation:[ 1; 1; 1 ] t);
  [%expect
    {|
    error: torch.ops.aten.max_pool2d.default: dilation=[2, 2] is not supported (only 1)
    error: torch.ops.aten.max_pool2d_with_indices.default: dilation=[2, 2] is not supported (only 1)
    error: torch.ops.aten.max_pool2d.default: ceil_mode=true is not supported
    error: torch.ops.aten.max_pool2d_with_indices.default: ceil_mode=true is not supported
    tensor f32 [W=2 C=2] {6, 8, 16, 18}
    tensor f32 [W=2 C=2] {6, 8, 16, 18}
    tensor f32 [W=2 C=2] {6, 8, 16, 18}
    tensor f32 [W=2 C=2] {6, 8, 16, 18}
    error: dilation: expected [h; w] or [v], got [1, 1, 1]
    error: dilation: expected [h; w] or [v], got [1, 1, 1] |}]

(* ---- the PT2 importer's conv2d.default arm, against ATen ---------------- *)

(* [Native_interp] is the OTHER importer: it reads SERIALIZED metadata where
   [Op_bridge] reads live tensors. Nothing above compares the two. The generated
   walks verify the bridge against ATen and test/native_interp/conv_test.ml pins
   the importer's graph structure, but neither says the importer computes ATen's
   answer -- and since no downloadable model serialises `conv2d.default` (every
   one is exported post-decomposition, carrying `convolution.default`), no cram
   over real weights ever will either.

   ONE serialized node feeds both paths: the ATen side takes it out of the
   DECODED graph rather than rebuilding it, so the two cannot come to disagree
   about what was asked. What is left to differ is the only thing under test --
   the weight relayout, the H/W pairing, and the channel arithmetic. *)

let jstr fmt = Printf.sprintf fmt

(* A local, deliberately minimal copy of test/native_interp/programs.ml's
   builder. Depending on that library would pull its inline tests into this
   runner, where they would run a second time under a different profile. *)
let meta_json sizes =
  jstr
    {|{"dtype":7,"sizes":[%s],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    (String.concat "," (List.map (fun i -> jstr {|{"as_int":%d}|} i) sizes))

(* [padding] is the whole serialized argument, because the two overloads spell it
   differently: `conv2d.default` takes an int pair and `conv2d.padding` a mode
   string. Everything else about the node -- and everything the comparison is
   actually about -- is identical, so they share one builder. *)
let conv_program ~target ~x_sizes ~w_sizes ~b_size ~stride ~padding ~dilation
    ~groups =
  let captured = [ ("w", w_sizes); ("b", [ b_size ]) ] in
  let as_t n = jstr {|{"as_tensor":{"name":"%s"}}|} n in
  let node =
    jstr
      {|{"target":"%s","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1},{"name":"bias","arg":%s,"kind":1},{"name":"stride","arg":{"as_ints":[%d,%d]},"kind":1},{"name":"padding","arg":%s,"kind":1},{"name":"dilation","arg":{"as_ints":[%d,%d]},"kind":1},{"name":"groups","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
      target (as_t "x") (as_t "w") (as_t "b") (fst stride) (snd stride) padding
      (fst dilation) (snd dilation) groups (as_t "y")
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[%s],"output_specs":[{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (String.concat "," (as_t "x" :: List.map (fun (n, _) -> as_t n) captured))
    (as_t "y") node
    (String.concat ","
       (jstr {|"x":%s|} (meta_json x_sizes)
       :: List.map (fun (n, s) -> jstr {|"%s":%s|} n (meta_json s)) captured))
    (String.concat ","
       (jstr {|{"user_input":{"arg":%s}}|} (as_t "x")
       :: List.map
            (fun (n, _) ->
              jstr {|{"parameter":{"arg":{"name":"%s"},"parameter_name":"%s"}}|}
                n n)
            captured))
    (as_t "y")

(* A repeatable non-constant fill: constant inputs make a transposed kernel or a
   swapped relayout indistinguishable from the correct one. *)
let ramp n = List.init n (fun i -> float_of_int ((i mod 7) - 3) /. 2.)

let importer_vs_aten ?(target = "torch.ops.aten.conv2d.default") label ~x_sizes
    ~w_sizes ~stride ~padding ~dilation ~groups =
  let numel = List.fold_left ( * ) 1 in
  let b_size = List.nth w_sizes 0 in
  let xs = ramp (numel x_sizes) and ws = ramp (numel w_sizes) in
  let bs = ramp b_size in
  let json =
    conv_program ~target ~x_sizes ~w_sizes ~b_size ~stride ~padding ~dilation
      ~groups
  in
  Format.printf "%-22s " label;
  match Jsont_bytesrw.decode_string PT.ExportedProgram.jsont json with
  | Error e -> Format.printf "fixture did not decode: %s@." e
  | Ok program -> (
      let node = List.hd program.PT.ExportedProgram.graph_module.graph.nodes in
      let aten_env =
        List.fold_left
          (fun m (k, sizes, vals) -> Sm.add k (float_tensor sizes vals) m)
          Sm.empty
          [ ("x", x_sizes, xs); ("w", w_sizes, ws); ("b", [ b_size ], bs) ]
      in
      match Interp_dispatch.dispatch aten_env node with
      | Error e ->
          Format.printf "aten: %a@." Interp_verify.pp_interp_error
            (Err.Error.kind e)
      | Ok aten_out -> (
          match Native_interp.lower program with
          | Error e ->
              Format.printf "lower: %a@." Native_interp.pp_error
                (Err.Error.kind e)
          | Ok lowered -> (
              let g = lowered.Pt2_native_graph.graph in
              (* Captured tensors are graph inputs of kind [Constant], and
                 [Eval_direct] takes those through [~constants] -- the split the
                 importer's own [run] makes when it resolves payloads. *)
              let inputs, constants =
                List.partition
                  (fun (id, _) ->
                    Graph_ir.input_kind g id = Graph_ir.Input.Input)
                  (List.combine g.Graph_ir.Graph.inputs
                     [
                       native_f32 x_sizes xs;
                       native_f32 w_sizes ws;
                       native_f32 [ b_size ] bs;
                     ])
              in
              match Eval_direct.run g ~constants ~inputs with
              | Error e ->
                  Format.printf "eval: %a@." Eval_direct.pp_error
                    (Err.Error.kind e)
              | Ok result ->
                  let native_y =
                    Graph_ir.Tensor_id.Map.find
                      (List.hd g.Graph_ir.Graph.outputs)
                      result
                  in
                  Format.printf "%a@." pp_result
                    (Verify.compare_tensors ~atol:1e-6 ~output:"y"
                       (Sm.find "y" aten_out) native_y))))

let ints_pad (h, w) = jstr {|{"as_ints":[%d,%d]}|} h w
let mode_pad m = jstr {|{"as_string":"%s"}|} m

let%expect_test "importer: conv2d.default matches ATen" =
  importer_vs_aten "defaults:" ~x_sizes:[ 1; 2; 5; 5 ] ~w_sizes:[ 3; 2; 3; 3 ]
    ~stride:(1, 1)
    ~padding:(ints_pad (0, 0))
    ~dilation:(1, 1) ~groups:1;
  (* Asymmetric everything: a transposed H/W pair survives every square
     configuration and nothing else here would catch it. *)
  importer_vs_aten "asymmetric:" ~x_sizes:[ 1; 2; 7; 5 ] ~w_sizes:[ 3; 2; 3; 2 ]
    ~stride:(2, 1)
    ~padding:(ints_pad (1, 0))
    ~dilation:(1, 2) ~groups:1;
  (* Depthwise: [in_channels] is the weight's per-group extent TIMES groups, so
     dropping the factor builds a conv over one channel instead of four. *)
  importer_vs_aten "depthwise:" ~x_sizes:[ 1; 4; 5; 5 ] ~w_sizes:[ 4; 1; 3; 3 ]
    ~stride:(1, 1)
    ~padding:(ints_pad (1, 1))
    ~dilation:(1, 1) ~groups:4;
  (* General grouping, which Native holds and only Native4D refuses. *)
  importer_vs_aten "groups=2:" ~x_sizes:[ 1; 4; 5; 5 ] ~w_sizes:[ 6; 2; 3; 3 ]
    ~stride:(1, 1)
    ~padding:(ints_pad (1, 1))
    ~dilation:(1, 1) ~groups:2;
  [%expect
    {|
    defaults:              Ok
    asymmetric:            Ok
    depthwise:             Ok
    groups=2:              Ok |}]

(* "same" is where an ATen oracle earns its place. The importer does not resolve
   the mode -- [Conv2d_padding.same_padding] does -- so what these check is that
   the mode SURVIVES the importer intact and that the resolution downstream of
   it agrees with ATen for the same weight.

   NOT COVERED HERE: an even kernel, where the total padding is odd and the
   split is genuinely asymmetric. ATen reaches [constant_pad_nd] for that case
   and this repository's minimal static-dispatch build does not carry it, so the
   comparison cannot be made -- it aborts the process rather than returning an
   error. [walk_meta.ml]'s conv2d_padding axes are all-odd kernels for the same
   reason. The asymmetric split is pinned against hand-computed values in
   test/native instead, where the rule actually lives. *)
let%expect_test "importer: conv2d.padding matches ATen" =
  let pad ?(groups = 1) label ~mode ~x_sizes ~w_sizes ~dilation =
    importer_vs_aten ~target:"torch.ops.aten.conv2d.padding" label ~x_sizes
      ~w_sizes ~stride:(1, 1) ~padding:(mode_pad mode) ~dilation ~groups
  in
  pad "valid, 3x3:" ~mode:"valid" ~x_sizes:[ 1; 2; 5; 5 ]
    ~w_sizes:[ 3; 2; 3; 3 ] ~dilation:(1, 1);
  pad "same, 3x3:" ~mode:"same" ~x_sizes:[ 1; 2; 5; 5 ] ~w_sizes:[ 3; 2; 3; 3 ]
    ~dilation:(1, 1);
  (* Asymmetric spatial extents, so a transposed H/W pair is visible. *)
  pad "same, 5x3:" ~mode:"same" ~x_sizes:[ 1; 2; 7; 5 ] ~w_sizes:[ 3; 2; 5; 3 ]
    ~dilation:(1, 1);
  (* Dilated "same": the total is [dilation * (kernel - 1)], not [kernel - 1],
     so dropping the dilation factor pads too little and moves the shape. *)
  pad "same, dilated:" ~mode:"same" ~x_sizes:[ 1; 2; 9; 9 ]
    ~w_sizes:[ 3; 2; 3; 3 ] ~dilation:(2, 3);
  (* Depthwise, to pin that the mode and the grouping compose. *)
  pad ~groups:4 "same, depthwise:" ~mode:"same" ~x_sizes:[ 1; 4; 5; 5 ]
    ~w_sizes:[ 4; 1; 3; 3 ] ~dilation:(1, 1);
  [%expect
    {|
    valid, 3x3:            Ok
    same, 3x3:             Ok
    same, 5x3:             Ok
    same, dilated:         Ok
    same, depthwise:       Ok |}]

(* ---- the PT2 importer's linear.default arm, against ATen ----------------- *)

(* Same shape of evidence, and the same reason for it: `linear.default` is not
   serialized by any downloadable model either. The weights here are all
   NON-SQUARE and the input rank varies, because the two mutations that matter
   -- transposing the weight permutation, and taking [in_features] from dim 0
   instead of dim 1 -- both leave a square weight's graph buildable. *)
(* [bias] is three-valued for the same reason it is in conv_test.ml, and here it
   turns up a divergence: the ATen decode path requires the argument to be
   PRESENT (Interp_decode.tensor_arg reports `Missing_argument for an omitted
   one, and maps an explicit none to a null tensor), while the importer and
   Op_bridge both read omission as None. The generated dispatch cannot tell a
   `Tensor?` slot from a `Tensor` one, so this is the ATen decoder's gap rather
   than the importer's -- pinned here rather than worked around. *)
let linear_program ~x_sizes ~w_sizes ~bias =
  let as_t n = jstr {|{"as_tensor":{"name":"%s"}}|} n in
  let captured =
    ("w", w_sizes)
    :: (if bias = `Tensor then [ ("b", [ List.nth w_sizes 0 ]) ] else [])
  in
  let node =
    jstr
      {|{"target":"torch.ops.aten.linear.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1}%s],"outputs":[%s],"metadata":{}}|}
      (as_t "x") (as_t "w")
      (match bias with
      | `Absent -> ""
      | `None -> {|,{"name":"bias","arg":{"as_none":true},"kind":1}|}
      | `Tensor -> jstr {|,{"name":"bias","arg":%s,"kind":1}|} (as_t "b"))
      (as_t "y")
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[%s],"output_specs":[{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (String.concat "," (as_t "x" :: List.map (fun (n, _) -> as_t n) captured))
    (as_t "y") node
    (String.concat ","
       (jstr {|"x":%s|} (meta_json x_sizes)
       :: List.map (fun (n, sz) -> jstr {|"%s":%s|} n (meta_json sz)) captured))
    (String.concat ","
       (jstr {|{"user_input":{"arg":%s}}|} (as_t "x")
       :: List.map
            (fun (n, _) ->
              jstr {|{"parameter":{"arg":{"name":"%s"},"parameter_name":"%s"}}|}
                n n)
            captured))
    (as_t "y")

let linear_vs_aten label ~x_sizes ~w_sizes ~bias =
  let numel = List.fold_left ( * ) 1 in
  let xs = ramp (numel x_sizes) and ws = ramp (numel w_sizes) in
  let out_features = List.nth w_sizes 0 in
  let bs = ramp out_features in
  Format.printf "%-22s " label;
  match
    Jsont_bytesrw.decode_string PT.ExportedProgram.jsont
      (linear_program ~x_sizes ~w_sizes ~bias)
  with
  | Error e -> Format.printf "fixture did not decode: %s@." e
  | Ok program -> (
      let node = List.hd program.PT.ExportedProgram.graph_module.graph.nodes in
      let aten_env =
        List.fold_left
          (fun m (k, sizes, vals) -> Sm.add k (float_tensor sizes vals) m)
          Sm.empty
          ([ ("x", x_sizes, xs); ("w", w_sizes, ws) ]
          @ if bias = `Tensor then [ ("b", [ out_features ], bs) ] else [])
      in
      match Interp_dispatch.dispatch aten_env node with
      | Error e ->
          Format.printf "aten: %a@." Interp_verify.pp_interp_error
            (Err.Error.kind e)
      | Ok aten_out -> (
          match Native_interp.lower program with
          | Error e ->
              Format.printf "lower: %a@." Native_interp.pp_error
                (Err.Error.kind e)
          | Ok lowered -> (
              let g = lowered.Pt2_native_graph.graph in
              let natives =
                [ native_f32 x_sizes xs; native_f32 w_sizes ws ]
                @
                if bias = `Tensor then [ native_f32 [ out_features ] bs ]
                else []
              in
              let inputs, constants =
                List.partition
                  (fun (id, _) ->
                    Graph_ir.input_kind g id = Graph_ir.Input.Input)
                  (List.combine g.Graph_ir.Graph.inputs natives)
              in
              match Eval_direct.run g ~constants ~inputs with
              | Error e ->
                  Format.printf "eval: %a@." Eval_direct.pp_error
                    (Err.Error.kind e)
              | Ok result ->
                  Format.printf "%a@." pp_result
                    (Verify.compare_tensors ~atol:1e-6 ~output:"y"
                       (Sm.find "y" aten_out)
                       (Graph_ir.Tensor_id.Map.find
                          (List.hd g.Graph_ir.Graph.outputs)
                          result)))))

let%expect_test "importer: linear.default matches ATen" =
  linear_vs_aten "bias omitted:" ~x_sizes:[ 4; 3 ] ~w_sizes:[ 5; 3 ]
    ~bias:`Absent;
  linear_vs_aten "bias explicit none:" ~x_sizes:[ 4; 3 ] ~w_sizes:[ 5; 3 ]
    ~bias:`None;
  linear_vs_aten "bias tensor:" ~x_sizes:[ 4; 3 ] ~w_sizes:[ 5; 3 ]
    ~bias:`Tensor;
  (* Leading axes pass through untouched; only the trailing extent reduces. *)
  linear_vs_aten "rank 4 input:" ~x_sizes:[ 2; 7; 4; 3 ] ~w_sizes:[ 5; 3 ]
    ~bias:`Tensor;
  (* out < in as well as out > in, so a transposed permutation is unbuildable in
     one direction and merely wrong in the other. *)
  linear_vs_aten "in > out:" ~x_sizes:[ 4; 6 ] ~w_sizes:[ 2; 6 ] ~bias:`Tensor;
  [%expect
    {|
    bias omitted:          aten: missing required argument "bias"
    bias explicit none:    Ok
    bias tensor:           Ok
    rank 4 input:          Ok
    in > out:              Ok |}]

(* ---- the PT2 importer's max_pool2d.default arm, against ATen ------------- *)

(* The input values are all NEGATIVE on purpose. Max-pooling's padding region
   must contribute -inf, not 0; a naive guarded read returning 0 beats every
   real value here and the result is wrong at every border window -- which, with
   pad = 1 on a small input, is everywhere. A non-negative fixture cannot see
   that at all. *)
let pool_program ~x_sizes ~kernel ~stride ~padding =
  let as_t n = jstr {|{"as_tensor":{"name":"%s"}}|} n in
  let ints (h, w) = jstr {|{"as_ints":[%d,%d]}|} h w in
  let node =
    jstr
      {|{"target":"torch.ops.aten.max_pool2d.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"kernel_size","arg":%s,"kind":1},{"name":"stride","arg":%s,"kind":1},{"name":"padding","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
      (as_t "x") (ints kernel) (ints stride) (ints padding) (as_t "y")
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[{"user_input":{"arg":%s}}],"output_specs":[{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (as_t "x") (as_t "y") node
    (jstr {|"x":%s|} (meta_json x_sizes))
    (as_t "x") (as_t "y")

let pool_vs_aten label ~x_sizes ~kernel ~stride ~padding =
  let n = List.fold_left ( * ) 1 x_sizes in
  let xs = List.init n (fun i -> -1. -. (float_of_int (i mod 11) /. 2.)) in
  Format.printf "%-22s " label;
  match
    Jsont_bytesrw.decode_string PT.ExportedProgram.jsont
      (pool_program ~x_sizes ~kernel ~stride ~padding)
  with
  | Error e -> Format.printf "fixture did not decode: %s@." e
  | Ok program -> (
      let node = List.hd program.PT.ExportedProgram.graph_module.graph.nodes in
      match
        Interp_dispatch.dispatch
          (Sm.add "x" (float_tensor x_sizes xs) Sm.empty)
          node
      with
      | Error e ->
          Format.printf "aten: %a@." Interp_verify.pp_interp_error
            (Err.Error.kind e)
      | Ok aten_out -> (
          match Native_interp.lower program with
          | Error e ->
              Format.printf "lower: %a@." Native_interp.pp_error
                (Err.Error.kind e)
          | Ok lowered -> (
              let g = lowered.Pt2_native_graph.graph in
              match
                Eval_direct.run g
                  ~inputs:
                    [ (List.hd g.Graph_ir.Graph.inputs, native_f32 x_sizes xs) ]
              with
              | Error e ->
                  Format.printf "eval: %a@." Eval_direct.pp_error
                    (Err.Error.kind e)
              | Ok result ->
                  Format.printf "%a@." pp_result
                    (Verify.compare_tensors ~atol:0. ~output:"y"
                       (Sm.find "y" aten_out)
                       (Graph_ir.Tensor_id.Map.find
                          (List.hd g.Graph_ir.Graph.outputs)
                          result)))))

let%expect_test "importer: max_pool2d.default matches ATen" =
  pool_vs_aten "square, no pad:" ~x_sizes:[ 1; 3; 8; 8 ] ~kernel:(2, 2)
    ~stride:(2, 2) ~padding:(0, 0);
  pool_vs_aten "padded:" ~x_sizes:[ 1; 3; 5; 5 ] ~kernel:(3, 3) ~stride:(1, 1)
    ~padding:(1, 1);
  (* Rectangular everything: a swapped stride or padding pair is invariant for a
     square configuration and visible here. *)
  pool_vs_aten "rectangular:" ~x_sizes:[ 1; 3; 9; 7 ] ~kernel:(3, 2)
    ~stride:(2, 1) ~padding:(1, 0);
  [%expect
    {|
    square, no pad:        Ok
    padded:                Ok
    rectangular:           Ok |}]

(* ---- the PT2 importer's rms_norm.default arm, against ATen --------------- *)

(* This is the row where shape-only evidence is worth least. Normalizing over
   the LEADING axes instead of the trailing ones, dividing by the wrong count,
   adding eps outside the sqrt instead of inside, or indexing the weight on the
   wrong axis all preserve the output shape exactly. Only a value comparison
   separates them, and ATen is the only oracle for it.

   The input is NON-UNIFORM across every axis and the extents are all distinct,
   so a wrong axis choice cannot coincide with the right one. *)
let rms_program ~x_sizes ~normalized ~weight =
  let as_t n = jstr {|{"as_tensor":{"name":"%s"}}|} n in
  let captured = if weight = `Tensor then [ ("w", normalized) ] else [] in
  let node =
    jstr
      {|{"target":"torch.ops.aten.rms_norm.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"normalized_shape","arg":{"as_ints":[%s]},"kind":1}%s],"outputs":[%s],"metadata":{}}|}
      (as_t "x")
      (String.concat "," (List.map string_of_int normalized))
      (match weight with
      | `Absent -> ""
      | `None -> {|,{"name":"weight","arg":{"as_none":true},"kind":1}|}
      | `Tensor -> jstr {|,{"name":"weight","arg":%s,"kind":1}|} (as_t "w"))
      (as_t "y")
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[%s],"output_specs":[{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (String.concat "," (as_t "x" :: List.map (fun (n, _) -> as_t n) captured))
    (as_t "y") node
    (String.concat ","
       (jstr {|"x":%s|} (meta_json x_sizes)
       :: List.map (fun (n, sz) -> jstr {|"%s":%s|} n (meta_json sz)) captured))
    (String.concat ","
       (jstr {|{"user_input":{"arg":%s}}|} (as_t "x")
       :: List.map
            (fun (n, _) ->
              jstr {|{"parameter":{"arg":{"name":"%s"},"parameter_name":"%s"}}|}
                n n)
            captured))
    (as_t "y")

let rms_vs_aten label ~x_sizes ~normalized ~weight =
  let numel = List.fold_left ( * ) 1 in
  (* Well away from zero, and never repeating within a normalized group: a
     constant input has RMS equal to itself and normalizes to 1 whatever axes
     were chosen. *)
  let scale n =
    List.init n (fun i -> 0.5 +. (float_of_int (i * 7 mod 13) /. 4.))
  in
  let xs = scale (numel x_sizes) and ws = scale (numel normalized) in
  Format.printf "%-22s " label;
  match
    Jsont_bytesrw.decode_string PT.ExportedProgram.jsont
      (rms_program ~x_sizes ~normalized ~weight)
  with
  | Error e -> Format.printf "fixture did not decode: %s@." e
  | Ok program -> (
      let node = List.hd program.PT.ExportedProgram.graph_module.graph.nodes in
      let aten_env =
        List.fold_left
          (fun m (k, sizes, vals) -> Sm.add k (float_tensor sizes vals) m)
          Sm.empty
          ([ ("x", x_sizes, xs) ]
          @ if weight = `Tensor then [ ("w", normalized, ws) ] else [])
      in
      match Interp_dispatch.dispatch aten_env node with
      | Error e ->
          Format.printf "aten: %a@." Interp_verify.pp_interp_error
            (Err.Error.kind e)
      | Ok aten_out -> (
          match Native_interp.lower program with
          | Error e ->
              Format.printf "lower: %a@." Native_interp.pp_error
                (Err.Error.kind e)
          | Ok lowered -> (
              let g = lowered.Pt2_native_graph.graph in
              let natives =
                [ native_f32 x_sizes xs ]
                @ if weight = `Tensor then [ native_f32 normalized ws ] else []
              in
              let inputs, constants =
                List.partition
                  (fun (id, _) ->
                    Graph_ir.input_kind g id = Graph_ir.Input.Input)
                  (List.combine g.Graph_ir.Graph.inputs natives)
              in
              match Eval_direct.run g ~constants ~inputs with
              | Error e ->
                  Format.printf "eval: %a@." Eval_direct.pp_error
                    (Err.Error.kind e)
              | Ok result ->
                  Format.printf "%a@." pp_result
                    (Verify.compare_tensors ~atol:1e-5 ~output:"y"
                       (Sm.find "y" aten_out)
                       (Graph_ir.Tensor_id.Map.find
                          (List.hd g.Graph_ir.Graph.outputs)
                          result)))))

let%expect_test "importer: rms_norm.default matches ATen" =
  rms_vs_aten "one axis, weight:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ]
    ~weight:`Tensor;
  (* No weight, spelled as an explicit none. The OMITTED spelling cannot be
     compared: Interp_decode.tensor_arg reports it as missing, the same gap the
     linear.default block above pins for [bias]. Both importers read omission as
     None, so it is the ATen decoder that cannot express this node. *)
  rms_vs_aten "one axis, none:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ]
    ~weight:`None;
  rms_vs_aten "one axis, omitted:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ]
    ~weight:`Absent;
  (* Two axes: the divisor is the PRODUCT of the normalized extents, so a count
     taken from one axis alone is off by a factor of the other. *)
  rms_vs_aten "two axes:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 3; 4 ]
    ~weight:`Tensor;
  (* Every axis normalized -- the one case a leading/trailing mix-up CANNOT be
     distinguished by, included so that its agreement is not read as evidence. *)
  rms_vs_aten "all axes:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 2; 3; 4 ]
    ~weight:`None;
  (* Rank 4 with four distinct extents, one normalized axis: the leading three
     must pass through and the weight must be read on the trailing one. *)
  rms_vs_aten "rank 4, one axis:" ~x_sizes:[ 2; 3; 4; 5 ] ~normalized:[ 5 ]
    ~weight:`Tensor;
  (* Rank 4, two normalized axes, so the pass-through and the multi-axis
     reduction are exercised together. *)
  rms_vs_aten "rank 4, two axes:" ~x_sizes:[ 2; 3; 4; 5 ] ~normalized:[ 4; 5 ]
    ~weight:`Tensor;
  [%expect
    {|
    one axis, weight:      Ok
    one axis, none:        Ok
    one axis, omitted:     aten: missing required argument "weight"
    two axes:              Ok
    all axes:              Ok
    rank 4, one axis:      Ok
    rank 4, two axes:      Ok |}]

(* ---- the PT2 importer's layer_norm.default arm, against ATen ------------- *)

(* Everything the rms_norm block above says about shape-only evidence applies
   here and then some: layer_norm has a second reduction and an affine step with
   an order, so OMITTING the centring (which turns it into rms_norm), dividing
   either reduction by the wrong count, putting eps outside the sqrt, indexing
   the affine operands at the full output coordinate, and applying bias before
   weight are all shape-preserving. ATen is the only oracle that separates them,
   and this is the only fixture in the tree that consults it for this row.

   The input is non-uniform across every axis and the extents are all distinct,
   so a wrong axis choice cannot coincide with the right one; the weight and the
   bias are drawn from DIFFERENT sequences, so swapping them is visible. *)
let ln_program ~target ~x_sizes ~normalized ~weight ~bias ~cudnn =
  let as_t n = jstr {|{"as_tensor":{"name":"%s"}}|} n in
  (* The decomposed target returns [(out, mean, rstd)]; the functional one
     returns a bare tensor. Only the first is compared -- the leading-outputs
     rule -- but the node must DECLARE all three, or neither importer sees the
     shape a real graph has. Both statistics stay dead, which is the state the
     corpus is in and the only one the importer accepts. *)
  let outs =
    if target = "torch.ops.aten.native_layer_norm.default" then
      String.concat "," [ as_t "y"; as_t "mean"; as_t "rstd" ]
    else as_t "y"
  in
  let captured =
    List.filter_map
      (fun (n, present) ->
        if present = `Tensor then Some (n, normalized) else None)
      [ ("w", weight); ("b", bias) ]
  in
  let affine name id = function
    | `Absent -> ""
    | `None -> jstr {|,{"name":"%s","arg":{"as_none":true},"kind":1}|} name
    | `Tensor -> jstr {|,{"name":"%s","arg":%s,"kind":1}|} name (as_t id)
  in
  let node =
    jstr
      {|{"target":"%s","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"normalized_shape","arg":{"as_ints":[%s]},"kind":1}%s%s,{"name":"eps","arg":{"as_float":1e-05},"kind":1}%s],"outputs":[%s],"metadata":{}}|}
      target (as_t "x")
      (String.concat "," (List.map string_of_int normalized))
      (affine "weight" "w" weight)
      (affine "bias" "b" bias)
      (match cudnn with
      | None -> ""
      | Some b ->
          jstr {|,{"name":"cudnn_enable","arg":{"as_bool":%b},"kind":1}|} b)
      outs
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[%s],"output_specs":[{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (String.concat "," (as_t "x" :: List.map (fun (n, _) -> as_t n) captured))
    (as_t "y") node
    (String.concat ","
       (jstr {|"x":%s|} (meta_json x_sizes)
       :: List.map (fun (n, sz) -> jstr {|"%s":%s|} n (meta_json sz)) captured))
    (String.concat ","
       (jstr {|{"user_input":{"arg":%s}}|} (as_t "x")
       :: List.map
            (fun (n, _) ->
              jstr {|{"parameter":{"arg":{"name":"%s"},"parameter_name":"%s"}}|}
                n n)
            captured))
    (as_t "y")

let ln_vs_aten ?(target = "torch.ops.aten.layer_norm.default") label ~x_sizes
    ~normalized ?(weight = `None) ?(bias = `None) ?cudnn () =
  let numel = List.fold_left ( * ) 1 in
  (* Well away from zero and never repeating within a normalized group. Unlike
     rms_norm, a CONSTANT input is not the degenerate case here (it normalizes
     to 0, not to 1) -- but a constant SPACING is: an arithmetic progression has
     the same normalized values whichever of its axes was reduced when the
     extents happen to agree, so the sequence is deliberately not one. *)
  let scale off n =
    List.init n (fun i -> off +. (float_of_int (i * 7 mod 13) /. 4.))
  in
  let xs = scale 0.5 (numel x_sizes) in
  let ws = scale 1.25 (numel normalized) in
  let bs = scale (-2.5) (numel normalized) in
  Format.printf "%-24s " label;
  match
    Jsont_bytesrw.decode_string PT.ExportedProgram.jsont
      (ln_program ~target ~x_sizes ~normalized ~weight ~bias ~cudnn)
  with
  | Error e -> Format.printf "fixture did not decode: %s@." e
  | Ok program -> (
      let node = List.hd program.PT.ExportedProgram.graph_module.graph.nodes in
      let aten_env =
        List.fold_left
          (fun m (k, sizes, vals) -> Sm.add k (float_tensor sizes vals) m)
          Sm.empty
          ([ ("x", x_sizes, xs) ]
          @ (if weight = `Tensor then [ ("w", normalized, ws) ] else [])
          @ if bias = `Tensor then [ ("b", normalized, bs) ] else [])
      in
      match Interp_dispatch.dispatch aten_env node with
      | Error e ->
          Format.printf "aten: %a@." Interp_verify.pp_interp_error
            (Err.Error.kind e)
      | Ok aten_out -> (
          match Native_interp.lower program with
          | Error e ->
              Format.printf "lower: %a@." Native_interp.pp_error
                (Err.Error.kind e)
          | Ok lowered -> (
              let g = lowered.Pt2_native_graph.graph in
              let natives =
                [ native_f32 x_sizes xs ]
                @ (if weight = `Tensor then [ native_f32 normalized ws ] else [])
                @ if bias = `Tensor then [ native_f32 normalized bs ] else []
              in
              let inputs, constants =
                List.partition
                  (fun (id, _) ->
                    Graph_ir.input_kind g id = Graph_ir.Input.Input)
                  (List.combine g.Graph_ir.Graph.inputs natives)
              in
              match Eval_direct.run g ~constants ~inputs with
              | Error e ->
                  Format.printf "eval: %a@." Eval_direct.pp_error
                    (Err.Error.kind e)
              | Ok result ->
                  Format.printf "%a@." pp_result
                    (Verify.compare_tensors ~atol:1e-5 ~output:"y"
                       (Sm.find "y" aten_out)
                       (Graph_ir.Tensor_id.Map.find
                          (List.hd g.Graph_ir.Graph.outputs)
                          result)))))

let%expect_test "importer: layer_norm.default matches ATen" =
  (* All four affine states, at the default tolerance. *)
  ln_vs_aten "neither:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ] ();
  ln_vs_aten "weight:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ] ~weight:`Tensor ();
  ln_vs_aten "bias:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ] ~bias:`Tensor ();
  ln_vs_aten "both:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ] ~weight:`Tensor
    ~bias:`Tensor ();
  (* The OMITTED spelling cannot be compared here, exactly as for rms_norm's
     weight: [Interp_decode.tensor_arg] reports an absent [Tensor?] key as
     missing rather than decoding it to None. Both native importers read
     omission as None (op_bridge's [optional_tensor_present],
     native_interp's), so it is the ATen decoder that cannot express the node,
     not the arm under test. *)
  ln_vs_aten "omitted:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ] ~weight:`Absent
    ~bias:`Absent ();
  (* Two axes: BOTH reduction divisors are the product of the normalized
     extents, so a count taken from one axis alone is off by a factor of the
     other in the mean and again in the variance. *)
  ln_vs_aten "two axes:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 3; 4 ]
    ~weight:`Tensor ~bias:`Tensor ();
  (* Every axis normalized -- the one case a leading/trailing mix-up CANNOT be
     distinguished by, included so its agreement is not read as evidence. *)
  ln_vs_aten "all axes:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 2; 3; 4 ] ();
  (* Rank 4, four distinct extents: the leading three must pass through and both
     affine operands must be read on the trailing one. *)
  ln_vs_aten "rank 4, one axis:" ~x_sizes:[ 2; 3; 4; 5 ] ~normalized:[ 5 ]
    ~weight:`Tensor ~bias:`Tensor ();
  ln_vs_aten "rank 4, three axes:" ~x_sizes:[ 2; 3; 4; 5 ]
    ~normalized:[ 3; 4; 5 ] ~weight:`Tensor ~bias:`Tensor ();
  (* cudnn_enable at both values against the same ATen call: the composite
     discards it, and so must the native arm. *)
  ln_vs_aten "cudnn_enable=true:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ]
    ~weight:`Tensor ~bias:`Tensor ~cudnn:true ();
  ln_vs_aten "cudnn_enable=false:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ]
    ~weight:`Tensor ~bias:`Tensor ~cudnn:false ();
  (* The DECOMPOSED target, through the same comparison. Its arithmetic is the
     functional one's -- ATen's composite IS a call to it -- so agreement here
     is not new evidence about the formula; what it proves is that the shared
     body reaches the same node from a 3-tuple node with a required eps and no
     cudnn_enable, and that exposing one output verifies against [out] rather
     than silently against [mean]. *)
  let decomposed = "torch.ops.aten.native_layer_norm.default" in
  ln_vs_aten ~target:decomposed "decomposed, neither:" ~x_sizes:[ 2; 3; 4 ]
    ~normalized:[ 4 ] ();
  ln_vs_aten ~target:decomposed "decomposed, both:" ~x_sizes:[ 2; 3; 4 ]
    ~normalized:[ 4 ] ~weight:`Tensor ~bias:`Tensor ();
  (* The corpus configuration: rank 3, one normalized axis, both affine
     operands present -- all 148 nodes across the four ViT models. *)
  ln_vs_aten ~target:decomposed "decomposed, corpus:" ~x_sizes:[ 1; 50; 768 ]
    ~normalized:[ 768 ] ~weight:`Tensor ~bias:`Tensor ();
  ln_vs_aten ~target:decomposed "decomposed, two axes:" ~x_sizes:[ 2; 3; 4 ]
    ~normalized:[ 3; 4 ] ~weight:`Tensor ~bias:`Tensor ();
  [%expect
    {|
    neither:                 Ok
    weight:                  Ok
    bias:                    Ok
    both:                    Ok
    omitted:                 aten: missing required argument "weight"
    two axes:                Ok
    all axes:                Ok
    rank 4, one axis:        Ok
    rank 4, three axes:      Ok
    cudnn_enable=true:       Ok
    cudnn_enable=false:      Ok
    decomposed, neither:     Ok
    decomposed, both:        Ok
    decomposed, corpus:      Ok
    decomposed, two axes:    Ok |}]

(* ---- SymInt-spelled arguments on the ATen path -------------------------- *)

(* A schema [SymInt] slot crosses as [Argument.Int] in every graph in the
   corpus, but the export schema also admits [Argument.Sym_int], which carries
   EITHER a resolved value ([as_int]) or an unresolved symbol ([as_name]). No
   decoder in the tree accepted that carrier at all until now, so both spellings
   were refused as the wrong kind and neither the acceptance nor the refusal had
   a stated rule.

   The rule, applied by every SymInt-shaped decoder in [Interp_decode]: a
   resolved value decodes as the int it is; a symbol is [`Unresolved_sym_arg],
   its own row rather than a wrong-kind, because the argument IS the kind the
   schema declares and what is missing is a shape environment to evaluate it in.
   Same distinction [Native_interp] already draws for a symbolic tensor
   dimension.

   [select.int]'s [SymInt index] covers the scalar decoder and [view.default]'s
   [SymInt[] size] the list one — the two shapes slice.Tensor's bounds and
   pad.default's pad list will arrive in.

   [Op_bridge] inherits the rule rather than restating it: its [int_arg] /
   [ints_arg] are [decode_result] wrappers over the [Interp_decode] ones
   (op_bridge.ml:169-174), so the two importers that share a decoder cannot
   disagree about a spelling. [Native_interp], which has its own hand-written
   decoders, is the third and is NOT covered here.

   Read the goldens precisely: "aten and native agree" is [verify_print]'s Ok
   line, and there is no [Op_bridge] arm for [select.int], so for that pair it
   says the ATen dispatch succeeded and the verifier found nothing to compare —
   not that a native kernel ran. The refusal lines are the load-bearing half. *)

let in_sym_int name i =
  PT.NamedArgument.make name (PT.Argument.Sym_int (PT.SymIntArgument.Int i))
    None

let in_sym_name name s =
  PT.NamedArgument.make name (PT.Argument.Sym_int (PT.SymIntArgument.Name s))
    None

let in_sym_ints name xs =
  PT.NamedArgument.make name
    (PT.Argument.Sym_ints (List.map (fun i -> PT.SymIntArgument.Int i) xs))
    None

let in_sym_ints_named name xs =
  PT.NamedArgument.make name
    (PT.Argument.Sym_ints
       (List.map
          (function
            | `I i -> PT.SymIntArgument.Int i | `N s -> PT.SymIntArgument.Name s)
          xs))
    None

let%expect_test "sym_int: a resolved SymInt argument decodes as its value" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  (* index spelled as_int and as_sym_int(as_int) must reach the same ATen call. *)
  verify_print ~target:"torch.ops.aten.select.int"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 0; in_int "index" 1 ];
  verify_print ~target:"torch.ops.aten.select.int"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 0; in_sym_int "index" 1 ];
  [%expect {|
    aten and native agree
    aten and native agree |}]

let%expect_test "sym_int: an unresolved SymInt argument is refused by name" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  verify_print ~target:"torch.ops.aten.select.int"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 0; in_sym_name "index" "s3" ];
  [%expect
    {| dispatch error: argument "index": unresolved symbolic value "s3" |}]

let%expect_test "sym_ints: a resolved SymInt[] decodes elementwise" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  verify_print ~target:"torch.ops.aten.view.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_sym_ints "size" [ 3; 2 ] ];
  [%expect {| aten and native agree |}]

let%expect_test "sym_ints: one unresolved entry refuses the whole list" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  verify_print ~target:"torch.ops.aten.view.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self"; in_sym_ints_named "size" [ `I 3; `N "s0" ] ];
  [%expect
    {| dispatch error: argument "size": unresolved symbolic value "s0" |}]

(* ---- aten.pad.default: ATen as the oracle ------------------------------- *)

(* [verify_print] runs the node through real ATen (Interp_dispatch) AND through
   Op_bridge + Eval_direct, then compares element-wise. Silence means agreement,
   so this is the independent oracle the Direct-vs-Symbolic walk cannot be: both
   sides of that walk instantiate the same Compute functor.

   It is what pins the two things the native side derives rather than reads: the
   pad list's innermost-first ORDER, and the reflect mirror. A reversal or a
   replicate-style clamp changes values that ATen does not, so it shows up here
   as a numeric disagreement rather than as a shape mismatch. *)

let pad_verify ~sizes ~pad ?mode ?value () =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  verify_print ~target:"torch.ops.aten.pad.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      ([ in_tensor "self"; in_ints "pad" pad ]
      @ (match mode with None -> [] | Some m -> [ in_string "mode" m ])
      @ match value with None -> [] | Some v -> [ in_float "value" v ])

let%expect_test "verify: pad constant, one and two pairs" =
  (* Asymmetric amounts on both axes, and DIFFERENT amounts per axis: a
     reversal of the pair list swaps H's amounts with W's, which changes the
     output shape as well as the values. *)
  pad_verify ~sizes:[ 2; 3 ] ~pad:[ 1; 2 ] ();
  pad_verify ~sizes:[ 2; 3 ] ~pad:[ 1; 2; 3; 0 ] ();
  pad_verify ~sizes:[ 2; 3; 4 ] ~pad:[ 1; 0; 0; 2 ] ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: pad constant fill, present and absent" =
  pad_verify ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ~value:0.5 ();
  pad_verify ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ~value:(-7.25) ();
  (* Absent means 0.0, which the source values (1..6) never take, so a dropped
     fill would be visible rather than accidentally right. *)
  pad_verify ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: pad reflect at every boundary" =
  (* rank 2 with one pair: ATen's reflection_pad1d. *)
  pad_verify ~sizes:[ 2; 5 ] ~pad:[ 2; 2 ] ~mode:"reflect" ();
  pad_verify ~sizes:[ 2; 5 ] ~pad:[ 3; 0 ] ~mode:"reflect" ();
  pad_verify ~sizes:[ 2; 5 ] ~pad:[ 0; 3 ] ~mode:"reflect" ();
  (* rank 3 with two pairs: reflection_pad2d, both spatial axes mirrored. *)
  pad_verify ~sizes:[ 2; 4; 5 ] ~pad:[ 1; 2; 2; 1 ] ~mode:"reflect" ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: negative pads crop, and mix with padding" =
  pad_verify ~sizes:[ 3; 5 ] ~pad:[ -1; -1 ] ();
  pad_verify ~sizes:[ 3; 5 ] ~pad:[ -2; 0 ] ();
  pad_verify ~sizes:[ 3; 5 ] ~pad:[ -1; 0; 2; 2 ] ();
  pad_verify ~sizes:[ 3; 5 ] ~pad:[ 2; 2; -1; 0 ] ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

(* The refusals, through [dispatch_print] rather than [verify_print]: these are
   nodes the native side declines, so there is nothing to compare against. Each
   asserts the typed row rather than an exception. *)
let pad_dispatch ~sizes ~pad ?mode ?value () =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  dispatch_print ~target:"torch.ops.aten.pad.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      ([ in_tensor "self"; in_ints "pad" pad ]
      @ (match mode with None -> [] | Some m -> [ in_string "mode" m ])
      @ match value with None -> [] | Some v -> [ in_float "value" v ])
    ~noutputs:1

let%expect_test "dispatch: pad.default refusals carry a typed row" =
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 1; 2; 3 ] ();
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 1; 1; 1; 1; 1; 1 ] ();
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ~mode:"replicate" ();
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ~mode:"circular" ();
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 3; 0 ] ~mode:"reflect" ();
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ 1; 1 ] ~mode:"reflect" ~value:1.5 ();
  (* A crop that consumes the axis: ATen returns a size-0 tensor, which the
     engine has no representation for. Refused by the SHAPE rule, so the row is
     a Native shape error and not an importer one. *)
  pad_dispatch ~sizes:[ 2; 3 ] ~pad:[ -2; -2 ] ();
  [%expect
    {|
    error: pad list has 3 entries; a pair per padded dimension is required
    error: pad list covers 3 dimensions of a rank-2 input
    error: pad mode "replicate" is outside the Native domain (constant and reflect are supported)
    error: pad mode "circular" is outside the Native domain (constant and reflect are supported)
    error: reflect pad of axis C by (3, 0) needs each side below the extent 3
    error: pad mode "reflect" takes no non-zero value argument
    error: pad of axis C by (-2, -2) over extent 3 leaves -1 elements; the engine has no empty extent |}]

(* ---- aten.slice.Tensor: ATen as the oracle ------------------------------ *)

(* The independent oracle for row 6.2. What it pins that nothing else can: the
   bound RESOLUTION -- defaulting, negative normalization, clamping -- against
   ATen's own, on the same tensor. test/native/aten_shape_test.ml checks
   [resolve_slice] against hand values; this checks that the values it was
   given are the ones ATen would have used. A rule that is self-consistently
   wrong passes the first and fails here. *)
let slice_verify ~sizes ?dim ?start ?stop ?step () =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  verify_print ~target:"torch.ops.aten.slice.Tensor"
    ~bindings:[ ("self", x) ]
    ~inputs:
      ([ in_tensor "self" ]
      @ (match dim with None -> [] | Some d -> [ in_int "dim" d ])
      @ (match start with None -> [] | Some s -> [ s ])
      @ (match stop with None -> [] | Some s -> [ s ])
      @ match step with None -> [] | Some s -> [ in_int "step" s ])

let%expect_test "verify: slice bounds, absent and explicit" =
  (* Every argument at its schema default: the identity slice on dim 0. Kept
     because it is the configuration a generated Default-tier walk would draw,
     and it has to be RIGHT even though it proves nothing on its own. *)
  slice_verify ~sizes:[ 4; 5 ] ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" 1) ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~stop:(in_int "end" 3) ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" 1)
    ~stop:(in_int "end" 4) ();
  (* An explicit null is a different ARGUMENT from an absent one, and both have
     to mean the schema default. *)
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_none "start")
    ~stop:(in_none "end") ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: slice negative bounds and clamping" =
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" (-2)) ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~stop:(in_int "end" (-1)) ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" (-4))
    ~stop:(in_int "end" (-1)) ();
  (* Both clamps. ATen clamps rather than rejecting, so a node ATen runs has to
     lower -- these would be refusals if the native side had chosen to reject
     out-of-range bounds instead of narrowing them. *)
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~stop:(in_int "end" 99) ();
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" (-99)) ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: slice step, exact and inexact division" =
  slice_verify ~sizes:[ 4; 6 ] ~dim:1 ~step:2 ();
  slice_verify ~sizes:[ 4; 6 ] ~dim:1 ~step:3 ();
  (* Span 5 over step 2 and span 5 over step 3: the ceiling is what makes these
     3 and 2 rather than 2 and 1, and ATen is the authority on which. *)
  slice_verify ~sizes:[ 4; 6 ] ~dim:1 ~start:(in_int "start" 1) ~step:2 ();
  slice_verify ~sizes:[ 4; 6 ] ~dim:1 ~start:(in_int "start" 1) ~step:3 ();
  slice_verify ~sizes:[ 4; 6 ] ~dim:1 ~start:(in_int "start" 1)
    ~stop:(in_int "end" 4) ~step:2 ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

let%expect_test "verify: slice dim spellings name the same axis" =
  slice_verify ~sizes:[ 4; 5; 6 ] ~dim:0 ~stop:(in_int "end" 2) ();
  slice_verify ~sizes:[ 4; 5; 6 ] ~dim:(-3) ~stop:(in_int "end" 2) ();
  slice_verify ~sizes:[ 4; 5; 6 ] ~dim:2 ~stop:(in_int "end" 2) ();
  slice_verify ~sizes:[ 4; 5; 6 ] ~dim:(-1) ~stop:(in_int "end" 2) ();
  [%expect
    {|
    aten and native agree
    aten and native agree
    aten and native agree
    aten and native agree |}]

(* op6-impl decision 3, on the BRIDGE decoder. The same pair is asserted on
   [Interp_decode] (test/native_bridge_test.ml's sym_int rows above, through
   [select.int]) and on [Native_interp] (test/native_interp/slice_test.ml).
   Three separate code paths, one rule -- which is the point, since before this
   they agreed only by all three refusing every [Sym_int] alike. *)
let%expect_test "verify: slice bounds spelled as resolved SymInt" =
  slice_verify ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_sym_int "start" 1)
    ~stop:(in_sym_int "end" 4) ();
  [%expect {| aten and native agree |}]

let slice_dispatch ~sizes ?dim ?start ?stop ?step () =
  let n = List.fold_left ( * ) 1 sizes in
  let x = float_tensor sizes (List.init n (fun i -> float_of_int (i + 1))) in
  dispatch_print ~target:"torch.ops.aten.slice.Tensor"
    ~bindings:[ ("self", x) ]
    ~inputs:
      ([ in_tensor "self" ]
      @ (match dim with None -> [] | Some d -> [ in_int "dim" d ])
      @ (match start with None -> [] | Some s -> [ s ])
      @ (match stop with None -> [] | Some s -> [ s ])
      @ match step with None -> [] | Some s -> [ in_int "step" s ])
    ~noutputs:1

let%expect_test "dispatch: slice.Tensor refusals carry a typed row" =
  (* Empty: ATen returns a size-0 tensor and the engine has no representation
     for one. Refused by the SHAPE rule, so the row is a Native shape error
     rather than an importer one -- the same layering pad's crop follows. *)
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" 2)
    ~stop:(in_int "end" 2) ();
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_int "start" 99) ();
  (* A non-positive step. ATen refuses it too, and the row comes from
     [Aten_shape.resolve_slice] rather than from the op. *)
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:1 ~step:0 ();
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:1 ~step:(-1) ();
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:2 ();
  slice_dispatch ~sizes:[ 4; 5 ] ~dim:1 ~start:(in_sym_name "start" "s4") ();
  [%expect
    {|
    error: slice of axis C [2, 2) step 1 over extent 5 selects 0 elements; the engine has no empty extent
    error: slice of axis C [5, 5) step 1 over extent 5 selects 0 elements; the engine has no empty extent
    error: slice step must be >= 1, got 0
    error: slice step must be >= 1, got -1
    error: slice.Tensor: invalid dimension 2 for rank 2
    error: argument "start": unresolved symbolic value "s4" |}]
