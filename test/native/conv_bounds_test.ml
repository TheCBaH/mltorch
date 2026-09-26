(* Aggregates in a convolution's shape rule that pass every per-factor limit and
   still cross the per-axis ceiling (2^31). Each product below wraps a 32-bit
   [int] under js_of_ocaml (2^32 wraps to 0), so an unchecked [*] would build
   [Dim.extent 0] there instead of refusing. The same verdicts hold on both
   backends. *)

let pos = Op_config.Pos.of_int
let nonneg = Op_config.Nonneg.of_int
let s = Vec6.shape

let params ~transposed ~groups ~stride : Conv.Convolution.params =
  {
    stride = { h = pos stride; w = pos 1 };
    padding = { h = nonneg 0; w = nonneg 0 };
    dilation = { h = pos 1; w = pos 1 };
    transposed;
    output_padding = { h = nonneg 0; w = nonneg 0 };
    groups = pos groups;
  }

let show r =
  match Err.payload r with
  | Ok shape -> Format.asprintf "ok %a" Vec6.pp_shape shape
  | Error e -> Format.asprintf "%a" Shape_error.pp e

(* weight [out=1, C=2^20 per group] x groups 2^12 = 2^32 input channels *)
let%expect_test "forward: per-group channels times groups is bounded" =
  let x_shape = s ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let weight_shape = s ~n:4096 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1048576 in
  print_endline
    (show
       (Conv.Convolution.output_shape ~x_shape ~weight_shape
          (params ~transposed:false ~groups:4096 ~stride:1)));
  [%expect
    {| the input channel count weight.C * groups is 4294967296, over the engine maximum of 2147483648 |}]

(* the transposed output channels are weight.C * groups, the same product *)
let%expect_test "transposed: output channels are bounded" =
  let x_shape = s ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4096 in
  let weight_shape = s ~n:4096 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1048576 in
  print_endline
    (show
       (Conv.Convolution.output_shape ~x_shape ~weight_shape
          (params ~transposed:true ~groups:4096 ~stride:1)));
  [%expect
    {| the output channel count weight.C * groups is 4294967296, over the engine maximum of 2147483648 |}]

(* (in - 1) * stride = (2^20 - 1) * 2^12 wraps a 32-bit int *)
let%expect_test "transposed: the spatial output extent is bounded" =
  let x_shape = s ~n:1 ~t:1 ~d:1 ~h:1048576 ~w:1 ~c:1 in
  let weight_shape = s ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  print_endline
    (show
       (Conv.Convolution.output_shape ~x_shape ~weight_shape
          (params ~transposed:true ~groups:1 ~stride:4096)));
  [%expect
    {| the output extent is 4294963201, over the engine maximum of 2147483648 |}]

let%expect_test "controls: in-range shapes are accepted" =
  let x_shape = s ~n:1 ~t:1 ~d:1 ~h:5 ~w:1 ~c:4 in
  let weight_shape = s ~n:4 ~t:1 ~d:1 ~h:3 ~w:1 ~c:2 in
  print_endline
    (show
       (Conv.Convolution.output_shape ~x_shape ~weight_shape
          (params ~transposed:false ~groups:2 ~stride:1)));
  let weight_shape = s ~n:4 ~t:1 ~d:1 ~h:3 ~w:1 ~c:2 in
  print_endline
    (show
       (Conv.Convolution.output_shape ~x_shape ~weight_shape
          (params ~transposed:true ~groups:2 ~stride:2)));
  [%expect {|
    ok [H=3 W=1 C=4]
    ok [H=11 W=1 C=4] |}]
