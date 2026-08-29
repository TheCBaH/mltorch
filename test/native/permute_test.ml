(* permute.default: an axis swap, the identity permutation, and a 3-axis
   cycle. Split from compute_test.ml. *)

open Compute_fixtures

let pp_shape_tensor ppf (out_shape, tensor) =
  Format.fprintf ppf "shape: %a@.%a" Vec6.pp_shape out_shape Tensor.pp tensor

let pp_named_shape_tensor name ppf (out_shape, tensor) =
  Format.fprintf ppf "%s: %a@.%a" name Vec6.pp_shape out_shape Tensor.pp tensor

let%expect_test "Direct: permute [W=2 C=3] — swap W and C gives [W=3 C=2]" =
  let module P = Permute.Permute.Compute (Direct) in
  (* [W=2 C=3] tensor: x[w,c] = w*3 + c, row-major values 0..5. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((col c * 3) + chan c))
  in
  let perm =
    [
      (Axis.N, Axis.N);
      (Axis.T, Axis.T);
      (Axis.D, Axis.D);
      (Axis.H, Axis.H);
      (Axis.W, Axis.C);
      (Axis.C, Axis.W);
    ]
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Permute.Permute.output_shape ~x_shape perm) (P.pixel perm ~x));
  (* transpose of [[0,1,2],[3,4,5]] is [[0,3],[1,4],[2,5]]: row-major 0,3,1,4,2,5 *)
  [%expect {| tensor f32 [W=3 C=2] {0, 3, 1, 4, 2, 5} |}]

let%expect_test "Direct: permute identity — output equals input" =
  let module P = Permute.Permute.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:4 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int (row c + col c + chan c))
  in
  let perm = List.map (fun a -> (a, a)) Axis.all in
  let result =
    let open Err.Syntax in
    let* out_shape = Permute.Permute.output_shape ~x_shape perm in
    let tensor = Schedule.evaluate out_shape (P.pixel perm ~x) in
    Err.return (out_shape, tensor)
  in
  Format.printf "%a@." (pp_result pp_shape_tensor) result;
  [%expect
    {|
    shape: [H=2 W=3 C=4]
    tensor f32 [H=2 W=3 C=4] {0, 1, 2, 3, 1, 2, 3, 4, ...} |}]

let%expect_test "Direct: permute 3D [H=2 W=3 C=4] — cycle H->W->C->H" =
  let module P = Permute.Permute.Compute (Direct) in
  (* Cycle: output H <- input W, output W <- input C, output C <- input H.
     Input shape [H=2 W=3 C=4] -> output shape [H=3 W=4 C=2]. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:4 in
  let x =
    Tensor.materialize x_shape (fun c ->
        (* unique value per element: h*12 + w*4 + c *)
        float_of_int ((row c * 12) + (col c * 4) + chan c))
  in
  let perm =
    [
      (Axis.N, Axis.N);
      (Axis.T, Axis.T);
      (Axis.D, Axis.D);
      (Axis.H, Axis.W);
      (Axis.W, Axis.C);
      (Axis.C, Axis.H);
    ]
  in
  let result =
    let open Err.Syntax in
    let* out_shape = Permute.Permute.output_shape ~x_shape perm in
    let tensor = Schedule.evaluate out_shape (P.pixel perm ~x) in
    Err.return (out_shape, tensor)
  in
  Format.printf "%a@." (pp_result (pp_named_shape_tensor "out shape")) result;
  (* output[h,w,c] = input[c, h, w]  (inverse cycle: C->H->W->C)
     (h=0,w=0,c=0): input[c=0,h=0,w=0]=0; (h=0,w=0,c=1): input[c=1,h=0,w=0]=12
     (h=0,w=1,c=0): input[c=0,h=0,w=1]=1; (h=0,w=1,c=1): input[c=1,h=0,w=1]=13 *)
  [%expect
    {|
    out shape: [H=3 W=4 C=2]
    tensor f32 [H=3 W=4 C=2] {0, 12, 1, 13, 2, 14, 3, 15, ...} |}]
