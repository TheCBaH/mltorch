open Func_ast

let parse s = match Func_schema.parse s with Ok v -> v | Error e -> failwith e

let%expect_test "simple positional args" =
  Format.printf "%a%!" pp (parse "relu(Tensor self) -> Tensor");
  [%expect {| relu(Tensor self) -> Tensor |}]

let%expect_test "optional and list types" =
  Format.printf "%a%!" pp
    (parse "add.Tensor(Tensor self, Tensor other, Scalar? alpha=None) -> Tensor");
  [%expect
    {| add.Tensor(Tensor self, Tensor other, Scalar? alpha=None) -> Tensor |}]

let%expect_test "annotated tensor arg and out" =
  Format.printf "%a%!" pp
    (parse "abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)");
  [%expect {| abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!) |}]

let%expect_test "multiple returns" =
  Format.printf "%a%!" pp
    (parse "chunk(Tensor self, int chunks, int dim=0) -> Tensor[]");
  [%expect {| chunk(Tensor self, int chunks, int dim=0) -> Tensor[] |}]

let%expect_test "defaults: float, bool, list" =
  Format.printf "%a%!" pp
    (parse
       "batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? \
        running_mean, Tensor? running_var, bool training, float momentum, \
        float eps=1e-05, bool cudnn_enabled=True) -> Tensor");
  [%expect
    {| batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps=1e-05, bool cudnn_enabled=True) -> Tensor |}]
