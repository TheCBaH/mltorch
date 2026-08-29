open Aten_func_ast

let parse s = Result.fold ~ok:Fun.id ~error:failwith (Aten_func_schema.parse s)

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

let%expect_test "default vocabulary renders every constructor" =
  List.iter
    (fun default -> print_endline (Default.to_string default))
    [
      Default.Bool false;
      Default.Bool true;
      Default.Float "1e-05";
      Default.Ident "contiguous_format";
      Default.Int 3;
      Default.IntList [ 1; 2 ];
      Default.None;
      Default.Str "quoted";
    ];
  [%expect
    {|
    False
    True
    1e-05
    contiguous_format
    3
    [1,2]
    None
    "quoted" |}]
