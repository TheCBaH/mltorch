(* reshape.default: contiguous reinterpretation and the numel-mismatch
   rejection. Split from compute_test.ml. *)

open Compute_fixtures

let%expect_test "Direct: reshape [H=2 W=3 C=1] -> [W=3 C=2] (contiguous)" =
  let module R = Reshape.Reshape.Compute (Direct) in
  (* row-major elements 0..5; a contiguous reshape reinterprets the same flat
     buffer, so the values are unchanged, only the shape differs. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 3) + col c))
  in
  let p =
    { Reshape.Reshape.shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:2 }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Reshape.Reshape.output_shape ~x_shape p)
       (R.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [W=3 C=2] {0, 1, 2, 3, 4, 5} |}]

let%expect_test
    "Direct: reshape rejects a target that changes the element count" =
  (* Numel 6 either way for the valid case above; here the target's numel (8)
     disagrees with the source's (6). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:1 in
  let bad_target = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:4 ~c:2 in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Reshape.Reshape.output_shape ~x_shape
       { Reshape.Reshape.shape = bad_target });
  [%expect
    {| reshape target [W=4 C=2] does not preserve the element count of [H=2 W=3 C=1] |}]
