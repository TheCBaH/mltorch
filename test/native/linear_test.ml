(* linear (addmm), bmm and batched_matmul (batched matrix multiply). Split
   from compute_test.ml. *)

open Compute_fixtures

let%expect_test "Direct: linear (addmm) — out_features mix in_features" =
  let module L = Linear.Linear.Compute (Direct) in
  let chan c = Dim.to_int (Vec6.get c Axis.C) in
  let x_shape = s1c 3 in
  let weight_shape = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 in
  let bias_shape = s1c 2 in
  let x = Tensor.materialize x_shape (fun c -> [| 1.; 2.; 3. |].(chan c)) in
  (* weight[0,:] selects x0; weight[1,:] sums x1+x2 *)
  let weight =
    Tensor.materialize weight_shape (fun c ->
        match (Dim.to_int (Vec6.get c Axis.N), chan c) with
        | 0, 0 -> 1.
        | 1, 1 | 1, 2 -> 1.
        | _ -> 0.)
  in
  let bias =
    Tensor.materialize bias_shape (fun c -> [| 10.; 100. |].(chan c))
  in
  let p = { Linear.Linear.in_features = Dim.extent 3 } in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Linear.Linear.output_shape p ~x_shape ~weight_shape)
       (L.pixel p ~x ~weight ~bias));
  [%expect {| tensor f32 [C=2] {11, 105} |}]

let%expect_test
    "Direct: bmm — B=2 batch matrix multiply, exercises batch axis isolation" =
  let module B = Matmul.Bmm.Compute (Direct) in
  (* input[H=2, W=2, C=3]: two 2×3 matrices laid out H=batch, W=row, C=inner.
     batch 0: [[1,2,3],[4,5,6]]   batch 1: [[1,0,0],[0,1,0]] *)
  let input_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:3 in
  let mat0 = [| [| 1.; 2.; 3. |]; [| 4.; 5.; 6. |] |] in
  let mat1 = [| [| 1.; 0.; 0. |]; [| 0.; 1.; 0. |] |] in
  let input =
    Tensor.materialize input_shape (fun c ->
        let b = Dim.to_int (Vec6.get c Axis.H) in
        let r = Dim.to_int (Vec6.get c Axis.W) in
        let k = Dim.to_int (Vec6.get c Axis.C) in
        [| mat0; mat1 |].(b).(r).(k))
  in
  (* mat2[H=2, W=3, C=2]: two 3×2 matrices laid out H=batch, W=inner, C=col.
     batch 0: [[1,0],[0,1],[1,1]]   batch 1: [[2,0],[0,2],[0,0]] *)
  let mat2_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:2 in
  let m2_0 = [| [| 1.; 0. |]; [| 0.; 1. |]; [| 1.; 1. |] |] in
  let m2_1 = [| [| 2.; 0. |]; [| 0.; 2. |]; [| 0.; 0. |] |] in
  let mat2 =
    Tensor.materialize mat2_shape (fun c ->
        let b = Dim.to_int (Vec6.get c Axis.H) in
        let k = Dim.to_int (Vec6.get c Axis.W) in
        let j = Dim.to_int (Vec6.get c Axis.C) in
        [| m2_0; m2_1 |].(b).(k).(j))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Matmul.Bmm.output_shape ~input_shape ~mat2_shape)
       (B.pixel ~input_shape ~input ~mat2));
  (* batch 0: [1,2,3]·[[1,0],[0,1],[1,1]] = [4,5]; [4,5,6]·same = [10,11]
     batch 1: [1,0,0]·[[2,0],[0,2],[0,0]] = [2,0]; [0,1,0]·same = [0,2] *)
  [%expect {| tensor f32 [H=2 W=2 C=2] {4, 5, 10, 11, 2, 0, 0, 2} |}]

(* [Batched_matmul] generalizes [Bmm] by reading [mat2] at the OUTPUT's own
   [D]/[H] rather than hard-coding 0 -- batch on [D] here (with [H]=1) is
   chosen specifically to prove that: [mat2] varies per [D] slice (an
   identity matrix at D=0, a swap matrix at D=1), so a regression back to
   [Bmm]'s zero-based read would silently apply D=0's identity to BOTH
   output slices, leaving D=1 unchanged instead of swapped. *)
let%expect_test
    "Direct: batched_matmul — D=2 batch isolation on the mat2 operand" =
  let module B = Matmul.Batched_matmul.Compute (Direct) in
  let input_shape = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:1 ~w:2 ~c:2 in
  let in0 = [| [| 1.; 2. |]; [| 3.; 4. |] |] in
  let in1 = [| [| 0.; 1. |]; [| 1.; 0. |] |] in
  let input =
    Tensor.materialize input_shape (fun c ->
        let d = Dim.to_int (Vec6.get c Axis.D) in
        let r = Dim.to_int (Vec6.get c Axis.W) in
        let k = Dim.to_int (Vec6.get c Axis.C) in
        [| in0; in1 |].(d).(r).(k))
  in
  let mat2_shape = Vec6.shape ~n:1 ~t:1 ~d:2 ~h:1 ~w:2 ~c:2 in
  let identity = [| [| 1.; 0. |]; [| 0.; 1. |] |] in
  let swap = [| [| 0.; 1. |]; [| 1.; 0. |] |] in
  let mat2 =
    Tensor.materialize mat2_shape (fun c ->
        let d = Dim.to_int (Vec6.get c Axis.D) in
        let k = Dim.to_int (Vec6.get c Axis.W) in
        let j = Dim.to_int (Vec6.get c Axis.C) in
        [| identity; swap |].(d).(k).(j))
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Matmul.Batched_matmul.output_shape ~input_shape ~mat2_shape)
       (B.pixel ~input_shape ~input ~mat2));
  (* D=0: input @ identity = input unchanged: [[1,2],[3,4]].
     D=1: input @ swap = column-swap of input: [[1,0],[1,0]] @ ... =
     [[0,1],[1,0]] @ [[0,1],[1,0]] = [[1,0],[0,1]] (NOT [[0,1],[1,0]], which
     is what a D=0-hardcoded read would wrongly produce). *)
  [%expect {| tensor f32 [D=2 H=1 W=2 C=2] {1, 2, 3, 4, 1, 0, 0, 1} |}]
