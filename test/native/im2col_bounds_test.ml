(* im2col's column channel count (C * kh * kw) and location count (oh * ow) are
   products of extents that each pass their own limit. 2^16 * 2^16 and
   2^11 * 2^11 * 2^11 cross the per-axis ceiling (2^31) and wrap a 32-bit [int]
   under js_of_ocaml, so an unchecked [*] would build a wrong extent there
   instead of refusing. The same verdicts hold on both backends. *)

let pos = Op_config.Pos.of_int
let nonneg = Op_config.Nonneg.of_int
let s = Vec6.shape

let window kernel : Im2col.Window.t =
  {
    kernel = Dim.extent kernel;
    dilation = pos 1;
    pad = nonneg 0;
    stride = pos 1;
  }

let params ~kh ~kw : Im2col.Params.t = { h = window kh; w = window kw }

let show r =
  match Err.payload r with
  | Ok shape -> Format.asprintf "ok %a" Vec6.pp_shape shape
  | Error e -> Format.asprintf "%a" Shape_error.pp e

let%expect_test "im2col: channels times kernel area is bounded" =
  let x_shape = s ~n:1 ~t:1 ~d:1 ~h:65536 ~w:65536 ~c:2048 in
  print_endline
    (show (Im2col.Im2col.output_shape ~x_shape (params ~kh:2048 ~kw:2048)));
  [%expect
    {| the im2col column channel count (input channels * kh * kw) is 8589934592, over the engine maximum of 2147483648 |}]

let%expect_test "col2im: kernel area is bounded" =
  let x_shape = s ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let params : Im2col.Col2im.params =
    {
      window = params ~kh:65536 ~kw:65536;
      output_h = Dim.extent 1;
      output_w = Dim.extent 1;
    }
  in
  print_endline (show (Im2col.Col2im.output_shape ~x_shape params));
  [%expect
    {| the im2col column channel count (input channels * kh * kw) is 4294967296, over the engine maximum of 2147483648 |}]

let%expect_test "controls: in-range shapes are accepted" =
  let x_shape = s ~n:1 ~t:1 ~d:2 ~h:5 ~w:5 ~c:3 in
  print_endline
    (show (Im2col.Im2col.output_shape ~x_shape (params ~kh:3 ~kw:3)));
  let x_shape = s ~n:1 ~t:1 ~d:1 ~h:2 ~w:27 ~c:9 in
  let params : Im2col.Col2im.params =
    {
      window = params ~kh:3 ~kw:3;
      output_h = Dim.extent 5;
      output_w = Dim.extent 5;
    }
  in
  print_endline (show (Im2col.Col2im.output_shape ~x_shape params));
  [%expect {|
    ok [H=2 W=27 C=9]
    ok [D=2 H=5 W=5 C=3] |}]
