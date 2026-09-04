(* The exact `conv2d.default` overload through the serialized path.

   These fixtures are not a supplement to real-model coverage — they ARE the
   coverage. No model this repository can download serialises this target: every
   one of resnet18, efficientnet_b0, mobilenet_v2/v3_small and vit_b_32 is
   exported post-decomposition and carries `convolution.default` instead. So the
   `interp_*_cram.t` suites never reach this arm, and nothing else will notice
   if it regresses. *)

open Programs

(* [bias] is spelled three ways on purpose: absent from the input list entirely,
   present as an explicit None, and present as a tensor. The schema default is
   None and the first two must reach the same builder call. *)
let conv ?(stride = "[1,1]") ?(padding = "[0,0]") ?(dilation = "[1,1]")
    ?(groups = 1) ?(bias = `Absent) () =
  let bias_arg =
    match bias with
    | `Absent -> ""
    | `None -> {|{"name":"bias","arg":{"as_none":true},"kind":1},|}
    | `Tensor -> jstr {|{"name":"bias","arg":%s,"kind":1},|} (as_tensor "b")
  in
  jstr
    {|{"target":"torch.ops.aten.conv2d.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1},%s{"name":"stride","arg":{"as_ints":%s},"kind":1},{"name":"padding","arg":{"as_ints":%s},"kind":1},{"name":"dilation","arg":{"as_ints":%s},"kind":1},{"name":"groups","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "w") bias_arg stride padding dilation groups
    (as_tensor "y")

(* Weight is OIHW, the ATen order, and the arm has to relayout it. [cin] here is
   the PER-GROUP input extent, so the activation's channel count is [cin *
   groups] -- the distinction the depthwise cases below turn on. *)
let prog ?(x_sizes = [ 1; 4; 8; 8 ]) ?(w_sizes = [ 8; 4; 3; 3 ]) ?bias_size node
    =
  let bias = match bias_size with None -> [] | Some n -> [ ("b", [ n ]) ] in
  let captured = ("w", w_sizes) :: bias in
  program ~x_sizes
    ~extra_tensor_values:(List.map (fun (n, s) -> (n, tensor_meta s)) captured)
    ~params:(List.map fst captured) ~nodes:[ node ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

(* A node COUNT cannot falsify a layout or parameter mistake: every mutation
   worth testing here -- a swapped weight permutation, a transposed H/W pair, a
   dropped output relayout -- keeps the count at four. So the shape of the
   evidence has to be the graph itself. *)
let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "conv2d.default lowers with schema defaults" =
  dump "defaults:" (prog (conv ()));
  [%expect
    {|
    defaults:
    graph
    inputs:
      [t0 f32 [H=4 W=8 C=8] ->[n0], t1 f32 [D=8 H=4 W=3 C=3] ->[n1] constant]
    nodes:
      group g1 torch.ops.aten.conv2d.default:
        n0: [t2 f32 [H=8 W=8 C=4] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t3 f32 [N=8 T=1 D=1 H=3 W=3 C=4] ->[n2]] =
          permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n2: [t4 f32 [H=6 W=6 C=8] ->[n3]] =
          conv2d
            x=t2 <-n0
            weight=t3 <-n1
            bias=none
            params={h={kernel=3; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=3; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=4;
                   groups=1}
        n3: [t5 f32 [H=8 W=6 C=6]] = permute x=t4 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [H=8 W=6 C=6] <-n3] |}]

let%expect_test "bias absent and explicit None reach the same graph" =
  show "bias absent:" (prog (conv ()));
  show "bias None:" (prog (conv ~bias:`None ()));
  show "bias tensor:" (prog ~bias_size:8 (conv ~bias:`Tensor ()));
  [%expect
    {|
    bias absent:               lowered, nodes=4
    bias None:                 lowered, nodes=4
    bias tensor:               lowered, nodes=4 |}]

let%expect_test "non-default stride, padding and dilation lower" =
  show "stride 2:" (prog (conv ~stride:"[2,2]" ()));
  show "padding 1:" (prog (conv ~padding:"[1,1]" ()));
  show "dilation 2:"
    (prog ~x_sizes:[ 1; 4; 16; 16 ] (conv ~dilation:"[2,2]" ()));
  [%expect
    {|
    stride 2:                  lowered, nodes=4
    padding 1:                 lowered, nodes=4
    dilation 2:                lowered, nodes=4 |}]

(* Every parameter distinct on H and W, so that a transposed pair -- kernel,
   stride or padding -- changes this output instead of hiding behind a square
   config. The weight is 5x3 and the input 16x12. *)
let%expect_test "asymmetric H/W parameters keep their axes" =
  dump "asymmetric:"
    (prog ~x_sizes:[ 1; 4; 16; 12 ] ~w_sizes:[ 8; 4; 5; 3 ]
       (conv ~stride:"[2,1]" ~padding:"[2,1]" ()));
  [%expect
    {|
    asymmetric:
    graph
    inputs:
      [t0 f32 [H=4 W=16 C=12] ->[n0], t1 f32 [D=8 H=4 W=5 C=3] ->[n1] constant]
    nodes:
      group g1 torch.ops.aten.conv2d.default:
        n0: [t2 f32 [H=16 W=12 C=4] ->[n2]] =
          permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t3 f32 [N=8 T=1 D=1 H=5 W=3 C=4] ->[n2]] =
          permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n2: [t4 f32 [H=8 W=12 C=8] ->[n3]] =
          conv2d
            x=t2 <-n0
            weight=t3 <-n1
            bias=none
            params={h={kernel=5; stride=2; pad_before=2; pad_after=2; dilation=1};
                   w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                   in_channels=4;
                   groups=1}
        n3: [t5 f32 [H=8 W=8 C=12]] = permute x=t4 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [H=8 W=8 C=12] <-n3] |}]

(* [in_channels] is the ACTIVATION's channels, [cin * groups] -- 1 * 4 here, not
   the weight's 1. Dropping the factor leaves a graph that still builds. *)
let%expect_test "depthwise grouping derives in_channels from cin times groups" =
  dump "depthwise:"
    (prog ~w_sizes:[ 4; 1; 3; 3 ] (conv ~groups:4 ~padding:"[1,1]" ()));
  [%expect
    {|
    depthwise:
    graph
    inputs:
      [t0 f32 [H=4 W=8 C=8] ->[n0], t1 f32 [D=4 H=1 W=3 C=3] ->[n1] constant]
    nodes:
      group g1 torch.ops.aten.conv2d.default:
        n0: [t2 f32 [H=8 W=8 C=4] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t3 f32 [N=4 T=1 D=1 H=3 W=3 C=1] ->[n2]] =
          permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n2: [t4 f32 [H=8 W=8 C=4] ->[n3]] =
          conv2d
            x=t2 <-n0
            weight=t3 <-n1
            bias=none
            params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                   w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                   in_channels=4;
                   groups=4}
        n3: [t5 f32 [H=4 W=8 C=8]] = permute x=t4 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [H=4 W=8 C=8] <-n3] |}]

let%expect_test "an explicit bias is wired into the conv, not relayouted" =
  dump "bias:" (prog ~bias_size:8 (conv ~bias:`Tensor ()));
  [%expect
    {|
    bias:
    graph
    inputs:
      [t0 f32 [H=4 W=8 C=8] ->[n0], t1 f32 [D=8 H=4 W=3 C=3] ->[n1] constant,
       t2 f32 [C=8] ->[n2] constant]
    nodes:
      group g1 torch.ops.aten.conv2d.default:
        n0: [t3 f32 [H=8 W=8 C=4] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t4 f32 [N=8 T=1 D=1 H=3 W=3 C=4] ->[n2]] =
          permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n2: [t5 f32 [H=6 W=6 C=8] ->[n3]] =
          conv2d
            x=t3 <-n0
            weight=t4 <-n1
            bias=t2
            params={h={kernel=3; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=3; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=4;
                   groups=1}
        n3: [t6 f32 [H=8 W=6 C=6]] = permute x=t5 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t6 f32 [H=8 W=6 C=6] <-n3] |}]

(* A one-element list is the symmetric spelling ATen accepts for all three. *)
let%expect_test "single-element H/W lists are accepted as symmetric" =
  show "one-element:"
    (prog (conv ~stride:"[1]" ~padding:"[1]" ~dilation:"[1]" ()));
  [%expect {| one-element:               lowered, nodes=4 |}]

let%expect_test "depthwise grouping lowers" =
  (* cin = 1 per group, groups = 4, so the activation has 4 channels. *)
  show "depthwise:"
    (prog ~w_sizes:[ 4; 1; 3; 3 ] (conv ~groups:4 ~padding:"[1,1]" ()));
  [%expect {| depthwise:                 lowered, nodes=4 |}]

let%expect_test "general grouping lowers in Native" =
  (* cin = 2 per group, groups = 2: neither 1 nor depthwise. This file checks
     only the Native boundary; whether Native4D also accepts the shape is
     the OTHER test's question, below ("the three group modes..."), and
     checking it here would be checking the wrong boundary. *)
  show "groups=2:"
    (prog ~x_sizes:[ 1; 4; 8; 8 ] ~w_sizes:[ 8; 2; 3; 3 ] (conv ~groups:2 ()));
  [%expect {| groups=2:                  lowered, nodes=4 |}]

(* ---- rejections -------------------------------------------------------- *)

let%expect_test "a non-rank-four weight is refused" =
  show "rank 2 weight:" (prog ~w_sizes:[ 8; 4 ] (conv ()));
  [%expect
    {| rank 2 weight:             malformed PT2 graph: w is rank 2, expected 4 |}]

let%expect_test "config faults are typed, not raised" =
  show "groups 0:" (prog (conv ~groups:0 ()));
  show "stride 0:" (prog (conv ~stride:"[0,0]" ()));
  show "padding -1:" (prog (conv ~padding:"[-1,-1]" ()));
  show "dilation 0:" (prog (conv ~dilation:"[0,0]" ()));
  show "stride arity:" (prog (conv ~stride:"[1,1,1]" ()));
  [%expect
    {|
    groups 0:                  malformed PT2 graph: torch.ops.aten.conv2d.default: groups must be positive, got 0
    stride 0:                  malformed PT2 graph: torch.ops.aten.conv2d.default: stride must be positive, got 0
    padding -1:                malformed PT2 graph: torch.ops.aten.conv2d.default: padding must not be negative, got -1
    dilation 0:                malformed PT2 graph: torch.ops.aten.conv2d.default: dilation must be positive, got 0
    stride arity:              malformed PT2 graph: stride must have one or two values, got 3 |}]

(* The 32-bit rule, exercised rather than argued. [in_channels] is
   [cin * groups]; both factors are individually representable and their product
   is not. Computed in [int], on a backend where [int] is 32 bits, this wraps to
   a small positive number and builds a plausible wrong graph -- which is why
   the product is bounded in [int64] before it is narrowed. This test runs under
   node as well as natively ([modes best js]). *)
let%expect_test "a channel product past the engine maximum is refused" =
  show "cin*groups overflow:"
    (prog ~w_sizes:[ 8; 1073741824; 3; 3 ] (conv ~groups:4 ()));
  [%expect
    {| cin*groups overflow:       malformed PT2 graph: w has extent 4294967296, over the engine maximum of 2147483648 |}]

(* Same gap, same rule, on the convolution side: the bias carries the weight's
   OUT-channel extent and nothing compared the two. *)
let%expect_test
    "a conv bias whose extent disagrees with out_channels is refused" =
  show "bias 3, out 8:" (prog ~bias_size:3 (conv ~bias:`Tensor ()));
  show "bias 8, out 8:" (prog ~bias_size:8 (conv ~bias:`Tensor ()));
  [%expect
    {|
    bias 3, out 8:             bias shape must be [C=8], got [C=3]
    bias 8, out 8:             lowered, nodes=4 |}]

let%expect_test "a leading-singleton conv bias is refused on rank" =
  show "bias [1,8]:"
    (program ~x_sizes:[ 1; 4; 8; 8 ]
       ~extra_tensor_values:
         [ ("w", tensor_meta [ 8; 4; 3; 3 ]); ("b", tensor_meta [ 1; 8 ]) ]
       ~params:[ "w"; "b" ]
       ~nodes:[ conv ~bias:`Tensor () ]
       ~graph_outputs:[ as_tensor "y" ]
       ());
  show "bias [8]:" (prog ~bias_size:8 (conv ~bias:`Tensor ()));
  [%expect
    {|
    bias [1,8]:                malformed PT2 graph: b is rank 2, expected 1
    bias [8]:                  lowered, nodes=4 |}]

let%expect_test "missing weight metadata names the conv2d role" =
  let node = conv () in
  show "no weight meta:"
    (program ~x_sizes:[ 1; 4; 8; 8 ] ~nodes:[ node ]
       ~graph_outputs:[ as_tensor "y" ]
       ());
  [%expect
    {| no weight meta:            malformed PT2 graph: no conv2d weight metadata for "w" |}]

(* ---- convolution.default's output_padding -------------------------------- *)

(* [output_padding] is an argument of this overload and was being forced to
   zero. It matters in both directions, and every existing fixture used [0,0],
   so neither could show. *)
let convolution ?(stride = "[1,1]") ?(padding = "[0,0]") ?(dilation = "[1,1]")
    ?(transposed = false) ?(output_padding = "[0,0]") ?(groups = 1) () =
  jstr
    {|{"target":"torch.ops.aten.convolution.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1},{"name":"bias","arg":{"as_none":true},"kind":1},{"name":"stride","arg":{"as_ints":%s},"kind":1},{"name":"padding","arg":{"as_ints":%s},"kind":1},{"name":"dilation","arg":{"as_ints":%s},"kind":1},{"name":"transposed","arg":{"as_bool":%b},"kind":1},{"name":"output_padding","arg":{"as_ints":%s},"kind":1},{"name":"groups","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "w") stride padding dilation transposed
    output_padding groups (as_tensor "y")

(* A transposed convolution's output extent INCLUDES output_padding:
   (in-1)*stride - 2*pad + dilation*(kernel-1) + output_padding + 1. On a 2x2
   input with stride 2 and a 2x2 kernel that is 4x4 at zero and 5x5 at one --
   so forcing the zero silently built a smaller op than the model asked for. *)
let%expect_test "transposed convolution carries output_padding into the extent"
    =
  dump "output_padding [0,0]:"
    (prog ~x_sizes:[ 1; 2; 2; 2 ] ~w_sizes:[ 2; 3; 2; 2 ]
       (convolution ~transposed:true ~stride:"[2,2]" ()));
  dump "output_padding [1,1]:"
    (prog ~x_sizes:[ 1; 2; 2; 2 ] ~w_sizes:[ 2; 3; 2; 2 ]
       (convolution ~transposed:true ~stride:"[2,2]" ~output_padding:"[1,1]" ()));
  [%expect
    {|
    output_padding [0,0]:
    graph
    inputs:
      [t0 f32 [H=2 W=2 C=2] ->[n0], t1 f32 [D=2 H=3 W=2 C=2] ->[n1] constant]
    nodes:
      group g1 torch.ops.aten.convolution.default:
        n0: [t2 f32 [H=2 W=2 C=2] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t3 f32 [N=2 T=1 D=1 H=2 W=2 C=3] ->[n2]] =
          permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n2: [t4 f32 [H=4 W=4 C=3] ->[n3]] =
          convolution
            x=t2 <-n0
            weight=t3 <-n1
            bias=none
            params={stride={h=2; w=2};
                   padding={h=0; w=0};
                   dilation={h=1; w=1};
                   transposed=true;
                   output_padding={h=0; w=0};
                   groups=1}
        n3: [t5 f32 [H=3 W=4 C=4]] = permute x=t4 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [H=3 W=4 C=4] <-n3]
    output_padding [1,1]:
    graph
    inputs:
      [t0 f32 [H=2 W=2 C=2] ->[n0], t1 f32 [D=2 H=3 W=2 C=2] ->[n1] constant]
    nodes:
      group g1 torch.ops.aten.convolution.default:
        n0: [t2 f32 [H=2 W=2 C=2] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t3 f32 [N=2 T=1 D=1 H=2 W=2 C=3] ->[n2]] =
          permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n2: [t4 f32 [H=5 W=5 C=3] ->[n3]] =
          convolution
            x=t2 <-n0
            weight=t3 <-n1
            bias=none
            params={stride={h=2; w=2};
                   padding={h=0; w=0};
                   dilation={h=1; w=1};
                   transposed=true;
                   output_padding={h=1; w=1};
                   groups=1}
        n3: [t5 f32 [H=3 W=5 C=5]] = permute x=t4 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [H=3 W=5 C=5] <-n3] |}]

(* And the other direction: a NON-transposed convolution with a nonzero
   output_padding is invalid. [Convolution.output_shape] rejects it; discarding
   the argument let the node through as though it had asked for something
   else. *)
let%expect_test "a non-transposed convolution refuses a nonzero output_padding"
    =
  show "not transposed, [1,1]:"
    (prog ~x_sizes:[ 1; 4; 8; 8 ] (convolution ~output_padding:"[1,1]" ()));
  show "not transposed, [0,0]:" (prog ~x_sizes:[ 1; 4; 8; 8 ] (convolution ()));
  [%expect
    {|
    not transposed, [1,1]:     output_padding must be zero for non-transposed convolution, got {h=1; w=1}
    not transposed, [0,0]:     lowered, nodes=4 |}]

let%expect_test "output_padding is validated like every other H/W argument" =
  show "negative:"
    (prog ~x_sizes:[ 1; 4; 8; 8 ] (convolution ~output_padding:"[-1,0]" ()));
  show "arity 3:"
    (prog ~x_sizes:[ 1; 4; 8; 8 ] (convolution ~output_padding:"[0,0,0]" ()));
  [%expect
    {|
    negative:                  malformed PT2 graph: torch.ops.aten.convolution.default: output_padding must not be negative, got -1
    arity 3:                   malformed PT2 graph: output_padding must have one or two values, got 3 |}]

(* ---- conv2d.padding ------------------------------------------------------ *)

(* The same decode and the same relayouts as the arm above; what differs is that
   the padding is a MODE. It is carried into the IR unresolved, so the fixtures
   below pin the mode that arrives, not a pad count. *)
let conv_pad ?(stride = "[1,1]") ?padding ?(dilation = "[1,1]") ?(groups = 1)
    ?(bias = `Absent) () =
  let bias_arg =
    match bias with
    | `Absent -> ""
    | `None -> {|{"name":"bias","arg":{"as_none":true},"kind":1},|}
    | `Tensor -> jstr {|{"name":"bias","arg":%s,"kind":1},|} (as_tensor "b")
  in
  let padding_arg =
    match padding with
    | None -> ""
    | Some m -> jstr {|{"name":"padding","arg":{"as_string":"%s"},"kind":1},|} m
  in
  jstr
    {|{"target":"torch.ops.aten.conv2d.padding","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1},%s{"name":"stride","arg":{"as_ints":%s},"kind":1},%s{"name":"dilation","arg":{"as_ints":%s},"kind":1},{"name":"groups","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "w") bias_arg stride padding_arg dilation groups
    (as_tensor "y")

let%expect_test "conv2d.padding: valid is the schema default" =
  dump "padding omitted:" (prog (conv_pad ()));
  [%expect
    {|
    padding omitted:
    graph
    inputs:
      [t0 f32 [H=4 W=8 C=8] ->[n0], t1 f32 [D=8 H=4 W=3 C=3] ->[n1] constant]
    nodes:
      group g1 torch.ops.aten.conv2d.padding:
        n0: [t2 f32 [H=8 W=8 C=4] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
        n1: [t3 f32 [N=8 T=1 D=1 H=3 W=3 C=4] ->[n2]] =
          permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
        n2: [t4 f32 [H=6 W=6 C=8] ->[n3]] =
          conv2d_padding
            x=t2 <-n0
            weight=t3 <-n1
            bias=none
            params={stride={h=1; w=1};
                   padding=valid;
                   dilation={h=1; w=1};
                   groups=1}
        n3: [t5 f32 [H=8 W=6 C=6]] = permute x=t4 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [H=8 W=6 C=6] <-n3] |}]

let%expect_test "conv2d.padding: same keeps the spatial extent" =
  (* An ODD kernel, which is where the asymmetric split matters: total padding
     is [dilation * (kernel - 1)] and "same" puts the smaller half before. *)
  show "same, 3x3:" (prog (conv_pad ~padding:"same" ()));
  show "valid, 3x3:" (prog (conv_pad ~padding:"valid" ()));
  show "same, bias:"
    (prog ~bias_size:8 (conv_pad ~padding:"same" ~bias:`Tensor ()));
  show "same, none bias:" (prog (conv_pad ~padding:"same" ~bias:`None ()));
  [%expect
    {|
    same, 3x3:                 lowered, nodes=4
    valid, 3x3:                lowered, nodes=4
    same, bias:                lowered, nodes=4
    same, none bias:           lowered, nodes=4 |}]

let%expect_test "conv2d.padding: an unknown mode is refused, not asserted" =
  (* [Conv2d_padding.padding_of_string] would [invalid_arg] on each of these and
     leave [lower] as an uncaught exception. *)
  show "SAME:" (prog (conv_pad ~padding:"SAME" ()));
  show "reflect:" (prog (conv_pad ~padding:"reflect" ()));
  show "empty:" (prog (conv_pad ~padding:"" ()));
  [%expect
    {|
    SAME:                      malformed PT2 graph: padding mode "SAME" is neither "valid" nor "same"
    reflect:                   malformed PT2 graph: padding mode "reflect" is neither "valid" nor "same"
    empty:                     malformed PT2 graph: padding mode "" is neither "valid" nor "same" |}]

(* WHERE it fails is the property, not THAT it fails. "same" with a stride above
   one has no answer, but the rule belongs to [Conv2d_padding.same_padding] --
   the single definition shape inference, [Compute] and Native4D all resolve
   through. Restating it in the importer would make a second, and the two would
   drift. *)
let%expect_test "conv2d.padding: same with stride > 1 fails in shape inference"
    =
  show "same, stride 2:" (prog (conv_pad ~padding:"same" ~stride:"[2,2]" ()));
  show "valid, stride 2:" (prog (conv_pad ~padding:"valid" ~stride:"[2,2]" ()));
  [%expect
    {|
    same, stride 2:            padding="same" is not supported for strided convolutions (stride=2)
    valid, stride 2:           lowered, nodes=4 |}]

let%expect_test "conv2d.padding: config faults are typed here too" =
  show "groups 0:" (prog (conv_pad ~groups:0 ()));
  show "dilation 0:" (prog (conv_pad ~dilation:"[0,0]" ()));
  show "rank 2 weight:" (prog ~w_sizes:[ 8; 4 ] (conv_pad ()));
  [%expect
    {|
    groups 0:                  malformed PT2 graph: torch.ops.aten.conv2d.padding: groups must be positive, got 0
    dilation 0:                malformed PT2 graph: torch.ops.aten.conv2d.padding: dilation must be positive, got 0
    rank 2 weight:             malformed PT2 graph: w is rank 2, expected 4 |}]

(* The OTHER aggregate F5 names, and the one the conv2d.padding path reaches:
   [dilation * (kernel - 1)]. Both factors are individually representable and
   their product is not -- 2^30 dilation over a 3-wide kernel is 2^31 exactly,
   the engine's per-axis ceiling. Computed in [int] on a 32-bit backend this
   wraps negative and reaches [Op_config.Nonneg.of_int]'s assertion as an
   escaping exception rather than a typed error.

   Bounded in [Window_axis.output_extent], which every windowed axis in the
   engine goes through -- conv2d, conv2d.padding, convolution and both pooling
   ops -- so this fixture covers all of them at once. *)
let%expect_test "an effective kernel past the engine maximum is refused" =
  show "conv dilation 2^30:" (prog (conv ~dilation:"[1073741824,1]" ()));
  show "padding dilation 2^30:"
    (prog (conv_pad ~padding:"same" ~dilation:"[1073741824,1]" ()));
  show "padding valid, 2^30:"
    (prog (conv_pad ~padding:"valid" ~dilation:"[1073741824,1]" ()));
  [%expect
    {|
    conv dilation 2^30:        the effective kernel dilation * (kernel - 1) + 1 is 2147483649, over the engine maximum of 2147483648
    padding dilation 2^30:     the effective kernel dilation * (kernel - 1) + 1 is 2147483649, over the engine maximum of 2147483648
    padding valid, 2^30:       the effective kernel dilation * (kernel - 1) + 1 is 2147483649, over the engine maximum of 2147483648 |}]

(* ---- Native4D: which constructor these params select --------------------- *)

(* [domain_test.ml] already pins the dialect's grouping rule -- one group is
   [Conv2D], one input channel per group is [DepthwiseConv2D], anything else is
   refused -- but it pins it on HAND-BUILT params. Nothing there says the params
   THIS ARM builds land where that rule reads them, and that is the whole claim:
   a serialized depthwise conv is depthwise only if the importer put [groups]
   and the weight's per-group extent in the right places.

   The conversion is meaningful only on a CANONICAL graph (domain.mli): the arm
   emits the weight right-aligned from ATen's OIHW, which lands on D/H/W/C, so
   every unfolded conv weight has a non-unit D and the whole graph is outside
   the dialect before grouping is even reached. Hence the pipeline first, with
   [~fold:true] -- and payloads, since [Fold_const] declines every node whose
   payload is unbound and an unfolded permute is exactly the node that has to
   go. The values are irrelevant (nothing here reads a number); the SHAPES are
   what the fold propagates. *)
let zero_tensor shape =
  let data =
    Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout
      (Vec6.numel shape :> int)
  in
  Bigarray.Array1.fill data 0.;
  Tensor.Tensor
    { Tensor.shape; payload = { Payload.fmt = F32; quant = No_quant; data } }

let to4d label json =
  Format.printf "%-22s " label;
  match lower json with
  | Error e -> Format.printf "%a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok lowered -> (
      let g = lowered.Pt2_native_graph.graph in
      let constants =
        List.fold_left
          (fun acc id ->
            match Graph_ir.input_kind g id with
            | Graph_ir.Input.Input -> acc
            | Graph_ir.Input.Constant ->
                let sg =
                  Graph_ir.Tensor_id.Map.find id g.Graph_ir.Graph.tensors
                in
                Graph_ir.Tensor_id.Map.add id
                  (zero_tensor sg.Tensor_sig.shape)
                  acc)
          Graph_ir.Tensor_id.Map.empty g.Graph_ir.Graph.inputs
      in
      match
        Native_interp.transform_lowered ~constants lowered
          ~passes:[ Pipeline.canonical ~fold:true ]
      with
      | Error e ->
          Format.printf "%a@." Native_interp.pp_error (Err.Error.kind e)
      | Ok (Native_interp.Transformed t) -> (
          match Snapshot.create t.graph with
          | Error e ->
              Format.printf "%a@." Graph_view.pp_error (Err.Error.kind e)
          | Ok (Snapshot.Pack src) -> (
              match
                Native4d.Lower.convert ~constants:t.constants
                  ~constant_store:t.constant_store src
              with
              | Error e ->
                  Format.printf "outside the dialect: %a@." Native4d.Error.pp
                    (Err.Error.kind e)
              | Ok (Native4d.Lower.Pack r) ->
                  let dst = Native4d.Lower.graph r in
                  Format.printf "%s@."
                    (String.concat ", "
                       (List.map
                          (fun (n : Native4d.Graph.node) ->
                            Native4d.Op.name n.Graph_common.Node.op)
                          dst.Graph_common.Graph.nodes)))))

let%expect_test "the three group modes reach the three Native4D convolutions" =
  to4d "groups=1:" (prog (conv ~padding:"[1,1]" ()));
  to4d "depthwise:"
    (prog ~w_sizes:[ 4; 1; 3; 3 ] (conv ~groups:4 ~padding:"[1,1]" ()));
  to4d "groups=2:"
    (prog ~x_sizes:[ 1; 4; 8; 8 ] ~w_sizes:[ 8; 2; 3; 3 ] (conv ~groups:2 ()));
  [%expect
    {|
    groups=1:              Permute4, Conv2D, Permute4
    depthwise:             Permute4, DepthwiseConv2D, Permute4
    groups=2:              Permute4, GroupedConv2D, Permute4 |}]

(* ---- conv1d.default: the same overload one spatial axis down ------------- *)

(* Rank-3 twin of [conv] above: one spatial axis, so stride/padding/dilation
   are each a single int rather than an H/W pair, and the weight is
   [Cout,Cin/groups,K]. [prog] itself needs no changes -- it already takes
   [x_sizes]/[w_sizes] generically. *)
let conv1d ?(stride = "[1]") ?(padding = "[0]") ?(dilation = "[1]")
    ?(groups = 1) ?(bias = `Absent) () =
  let bias_arg =
    match bias with
    | `Absent -> ""
    | `None -> {|{"name":"bias","arg":{"as_none":true},"kind":1},|}
    | `Tensor -> jstr {|{"name":"bias","arg":%s,"kind":1},|} (as_tensor "b")
  in
  jstr
    {|{"target":"torch.ops.aten.conv1d.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1},%s{"name":"stride","arg":{"as_ints":%s},"kind":1},{"name":"padding","arg":{"as_ints":%s},"kind":1},{"name":"dilation","arg":{"as_ints":%s},"kind":1},{"name":"groups","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "w") bias_arg stride padding dilation groups
    (as_tensor "y")

let%expect_test "conv1d.default lowers with schema defaults" =
  dump "defaults:" (prog ~x_sizes:[ 1; 4; 8 ] ~w_sizes:[ 8; 4; 3 ] (conv1d ()));
  [%expect
    {|
    defaults:
    graph
    inputs: [t0 f32 [W=4 C=8] ->[n0], t1 f32 [H=8 W=4 C=3] ->[n1] constant]
    nodes:
      group g1 torch.ops.aten.conv1d.default:
        n0: [t2 f32 [W=8 C=4] ->[n2]] =
          permute x=t0 perm=[N<-H, H<-N, W<-C, C<-W]
        n1: [t3 f32 [N=8 T=1 D=1 H=1 W=3 C=4] ->[n2]] =
          permute x=t1 perm=[N<-H, H<-N, W<-C, C<-W]
        n2: [t4 f32 [W=6 C=8] ->[n3]] =
          conv1d
            x=t2 <-n0
            weight=t3 <-n1
            bias=none
            params={w={kernel=3; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=4;
                   groups=1}
        n3: [t5 f32 [W=8 C=6]] = permute x=t4 <-n2 perm=[N<-H, H<-N, W<-C, C<-W]
    outputs: [t5 f32 [W=8 C=6] <-n3] |}]

let%expect_test "conv1d reaches Conv2D in Native4D, H pinned to the unit window"
    =
  to4d "groups=1:"
    (prog ~x_sizes:[ 1; 4; 8 ] ~w_sizes:[ 8; 4; 3 ] (conv1d ~padding:"[1]" ()));
  to4d "depthwise:"
    (prog ~x_sizes:[ 1; 4; 8 ] ~w_sizes:[ 4; 1; 3 ]
       (conv1d ~groups:4 ~padding:"[1]" ()));
  [%expect
    {|
    groups=1:              Permute4, Conv2D, Permute4
    depthwise:             Permute4, DepthwiseConv2D, Permute4 |}]

(* ---- conv3d.default: the same overload, three spatial axes ---------------- *)

(* Rank-5 twin of [conv] above: three spatial axes (ATen's own D/H/W order),
   so stride/padding/dilation are each a 3-int list, and the weight is
   [Cout,Cin/groups,Kd,Kh,Kw]. [prog] itself needs no changes -- it already
   takes [x_sizes]/[w_sizes] generically. *)
let conv3d ?(stride = "[1,1,1]") ?(padding = "[0,0,0]") ?(dilation = "[1,1,1]")
    ?(groups = 1) ?(bias = `Absent) () =
  let bias_arg =
    match bias with
    | `Absent -> ""
    | `None -> {|{"name":"bias","arg":{"as_none":true},"kind":1},|}
    | `Tensor -> jstr {|{"name":"bias","arg":%s,"kind":1},|} (as_tensor "b")
  in
  jstr
    {|{"target":"torch.ops.aten.conv3d.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1},%s{"name":"stride","arg":{"as_ints":%s},"kind":1},{"name":"padding","arg":{"as_ints":%s},"kind":1},{"name":"dilation","arg":{"as_ints":%s},"kind":1},{"name":"groups","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "w") bias_arg stride padding dilation groups
    (as_tensor "y")

let%expect_test "conv3d.default lowers with schema defaults" =
  dump "defaults:"
    (prog ~x_sizes:[ 1; 4; 6; 6; 6 ] ~w_sizes:[ 8; 4; 3; 3; 3 ] (conv3d ()));
  [%expect
    {|
    defaults:
    graph
    inputs:
      [t0 f32 [D=4 H=6 W=6 C=6] ->[n0],
       t1 f32 [T=8 D=4 H=3 W=3 C=3] ->[n1] constant]
    nodes:
      group g1 torch.ops.aten.conv3d.default:
        n0: [t2 f32 [D=6 H=6 W=6 C=4] ->[n2]] =
          permute x=t0 perm=[N<-T, T<-N, D<-H, H<-W, W<-C, C<-D]
        n1: [t3 f32 [N=8 T=1 D=3 H=3 W=3 C=4] ->[n2]] =
          permute x=t1 perm=[N<-T, T<-N, D<-H, H<-W, W<-C, C<-D]
        n2: [t4 f32 [D=4 H=4 W=4 C=8] ->[n3]] =
          conv3d
            x=t2 <-n0
            weight=t3 <-n1
            bias=none
            params={d={kernel=3; stride=1; pad_before=0; pad_after=0; dilation=1};
                   h={kernel=3; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=3; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=4;
                   groups=1}
        n3: [t5 f32 [D=8 H=4 W=4 C=4]] =
          permute x=t4 <-n2 perm=[N<-T, T<-N, D<-C, H<-D, W<-H, C<-W]
    outputs: [t5 f32 [D=8 H=4 W=4 C=4] <-n3] |}]

(* Native4D forces T/D to extent 1 always; a real (non-unit) D is the
   ordinary case for this op (three genuinely spatial axes land on D/H/W, per
   conv_conv3d.ml's own comment), so this is an intrinsic-axis boundary, the
   same rejection [Unfold]/[Batched_matmul]'s multi-batch form get -- not a
   missing Native4D counterpart. *)
let%expect_test "conv3d is rejected at Native4D, an intrinsic D/H/W boundary" =
  to4d "groups=1:"
    (prog ~x_sizes:[ 1; 4; 6; 6; 6 ] ~w_sizes:[ 8; 4; 3; 3; 3 ]
       (conv3d ~padding:"[1,1,1]" ()));
  [%expect
    {| groups=1:              outside the dialect: node n0: axis T is outside the N/H/W/C dialect |}]
