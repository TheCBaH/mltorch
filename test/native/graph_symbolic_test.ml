(* Symbolic evaluation of native graphs: build the whole-graph stage DAG, print it
   (each stage's body shows the upstream signatures it loads), then chain-ground it
   and check the result equals Direct evaluation. See .ai/native_graph_design.md. *)

open Graph_ir

type output_count = { graph_name : string; count : int }

type error =
  [ `Build of Graph_builder.error
  | `Eval of Eval_direct.error
  | `Expected_single_output of output_count
  | `Missing_output_tensor of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Build e -> Graph_builder.pp_error ppf e
  | `Eval e -> Eval_direct.pp_error ppf e
  | `Expected_single_output { graph_name; count } ->
      Format.fprintf ppf "graph %S expected a single output, got %d" graph_name
        count
  | `Missing_output_tensor id ->
      Format.fprintf ppf "missing output tensor t%d" (Tensor_id.to_int id)

let pp_result pp_ok = Core.Pretty.core_result ~ok:pp_ok ~error:pp_error

let lift_build (r : ('a, Graph_builder.error) Core.result) :
    ('a, error) Core.result =
  Core.map_error (fun e -> `Build e) r

let lift_eval (r : ('a, Eval_direct.error) Core.result) :
    ('a, error) Core.result =
  Core.map_error (fun e -> `Eval e) r

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n
let chan c = Dim.to_int (Vec6.get c Axis.C)

let conv_axis ~kernel ~stride ~pad : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad;
    pad_after = Op_config.Nonneg.of_int pad;
    dilation = Op_config.Pos.of_int 1;
  }

let tensors_match shape a b =
  let ok = ref true in
  Vec6.iter shape (fun c ->
      if not (Float.equal (Tensor.read a c) (Tensor.read b c)) then ok := false);
  !ok

let output_id (g : graph) =
  match g.Graph.outputs with
  | [ id ] -> Core.return id
  | outputs ->
      Core.fail
        (`Expected_single_output
           { graph_name = g.Graph.name; count = List.length outputs })

let find_tensor env id =
  match Tensor_id.Map.find_opt id env with
  | Some tensor -> Core.return tensor
  | None -> Core.fail (`Missing_output_tensor id)

let compare_output g grounded direct =
  let open Core.Syntax in
  let* id = output_id g in
  let* grounded_out = find_tensor grounded id in
  let* direct_out = find_tensor direct id in
  let (Tensor.Tensor r) = grounded_out in
  Core.return (grounded_out, tensors_match r.shape grounded_out direct_out)

let pp_ground_result name ppf (tensor, matches) =
  Format.fprintf ppf "%s = %a@.ground matches direct: %b" name Tensor.pp tensor
    matches

let%expect_test "Symbolic graph: add -> relu stage DAG + ground matches Direct"
    =
  let result =
    let open Core.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"seq" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          let* t = add ~name:"sum" a b in
          relu ~name:"out" t)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a = Tensor.materialize (s1c 3) (fun c -> [| 1.; -5.; 2. |].(chan c)) in
    let b = Tensor.materialize (s1c 3) (fun c -> [| -3.; 1.; 2. |].(chan c)) in
    let inputs = List.combine g.Graph.inputs [ a; b ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: a, b
    sum = (a[0,0,0,0,0,C] + b[0,0,0,0,0,C])
    out = select((sum[N,T,D,H,W,C] < 0), 0, sum[N,T,D,H,W,C])
    outputs: out |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=3] {0, 0, 4}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: mul stage DAG + ground matches Direct" =
  let result =
    let open Core.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"prod" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          mul ~name:"out" a b)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let a = Tensor.materialize (s1c 3) (fun c -> [| 1.; -5.; 2. |].(chan c)) in
    let b = Tensor.materialize (s1c 3) (fun c -> [| -3.; 1.; 2. |].(chan c)) in
    let inputs = List.combine g.Graph.inputs [ a; b ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: a, b
    out = (a[0,0,0,0,0,C] * b[0,0,0,0,0,C])
    outputs: out |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=3] {-3, -5, 4}
    ground matches direct: true |}]

(* Conv decomposition symbolically: the conv stage loads the permute stage's
   signature, and the final permute loads the conv stage's — i.e. symbolic
   execution extends through the whole graph by tensor signature. *)
let conv_params =
  {
    Conv.Conv2d.h = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    w = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    in_channels = Dim.extent 2;
    groups = Op_config.Pos.of_int 1;
  }

let p_to_nhwc = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]
let p_to_nchw = Axis.[ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

let conv_padding_params =
  {
    Conv.Conv2d_padding.stride =
      { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
    padding = Conv.Conv2d_padding.Same;
    dilation = { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
    groups = Op_config.Pos.of_int 1;
  }

let convolution_params =
  {
    Conv.Convolution.stride =
      { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
    padding = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    dilation = { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
    transposed = false;
    output_padding =
      { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    groups = Op_config.Pos.of_int 1;
  }

let convolution_transposed_params =
  { convolution_params with Conv.Convolution.transposed = true }

let%expect_test
    "Symbolic graph: conv decomposition stage DAG + ground matches Direct" =
  let result =
    let open Core.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"conv_decomp" ~outputs:(fun (_, _, _, y) -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 3) ~name:"x_nchw" () in
          let* w = input ~shape:(s 1 1 1 2 2 2) ~name:"w" () in
          let* b = input ~shape:(s1c 1) ~name:"b" () in
          let* xh = permute ~name:"x_nhwc" p_to_nhwc x in
          let* yh =
            conv2d ~name:"y_nhwc" conv_params ~x:xh ~weight:w ~bias:b ()
          in
          let* y = permute ~name:"y_nchw" p_to_nchw yh in
          return (x, w, b, y))
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 2 3 3) (fun c ->
          let chan = Dim.to_int (Vec6.get c Axis.H) in
          let sph = Dim.to_int (Vec6.get c Axis.W) in
          let spw = Dim.to_int (Vec6.get c Axis.C) in
          float_of_int ((chan * 100) + (sph * 10) + spw))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 2) (fun _ -> 1.) in
    let b = Tensor.materialize (s1c 1) (fun _ -> 0.) in
    let inputs = List.combine g.Graph.inputs [ x; w; b ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: x_nchw, w, b
    x_nhwc = x_nchw[N,T,D,C,H,W]
    y_nhwc = (sum(r1=0..2: sum(r2=max(0,-1*1*H+0)..min(2,3+-1+-1*1*H+0+1): sum(r3=max(0,-1*1*W+0)..min(2,3+-1+-1*1*W+0+1): (x_nhwc[N,T,D,1*H+0+1*r2,1*W+0+1*r3,2*0+r1] * w[C,0,0,r2,r3,r1])))) + b[0,0,0,0,0,C])
    y_nchw = y_nhwc[N,T,D,W,C,H]
    outputs: y_nchw |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground y_nchw")) result;
  [%expect
    {|
    ground y_nchw = tensor f32 [W=2 C=2] {444, 452, 524, 532}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: conv2d_padding same ground matches Direct" =
  let result =
    let open Core.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"conv_padding" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 3 3 1) ~name:"x" () in
          let* w = input ~shape:(s 1 1 1 2 2 1) ~name:"w" () in
          conv2d_padding ~name:"y" conv_padding_params ~x ~weight:w ())
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 3 3 1) (fun c ->
          let h = Dim.to_int (Vec6.get c Axis.H) in
          let w = Dim.to_int (Vec6.get c Axis.W) in
          float_of_int ((h * 3) + w))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 1) (fun _ -> 1.) in
    let inputs = List.combine g.Graph.inputs [ x; w ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: x, w
    y = (sum(r1=0..1: sum(r2=max(0,-1*1*H+0)..min(2,3+-1+-1*1*H+0+1): sum(r3=max(0,-1*1*W+0)..min(2,3+-1+-1*1*W+0+1): (x[N,T,D,1*H+0+1*r2,1*W+0+1*r3,1*0+r1] * w[C,0,0,r2,r3,r1])))) + const0[0,0,0,0,0,C])
    outputs: y |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=3 W=3 C=1] {8, 12, 7, 20, 24, 13, 13, 15, ...}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: convolution ground matches Direct" =
  let result =
    let open Core.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"convolution" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 3 3 1) ~name:"x" () in
          let* w = input ~shape:(s 1 1 1 2 2 1) ~name:"w" () in
          convolution ~name:"y" convolution_params ~x ~weight:w ())
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 3 3 1) (fun c ->
          let h = Dim.to_int (Vec6.get c Axis.H) in
          let w = Dim.to_int (Vec6.get c Axis.W) in
          float_of_int ((h * 3) + w))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 1) (fun _ -> 1.) in
    let inputs = List.combine g.Graph.inputs [ x; w ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: x, w
    y = (sum(r1=0..1: sum(r2=max(0,-1*1*H+0)..min(2,3+-1+-1*1*H+0+1): sum(r3=max(0,-1*1*W+0)..min(2,3+-1+-1*1*W+0+1): (x[N,T,D,1*H+0+1*r2,1*W+0+1*r3,1*0+r1] * w[C,0,0,r2,r3,r1])))) + const0[0,0,0,0,0,C])
    outputs: y |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=2 W=2 C=1] {8, 12, 20, 24}
    ground matches direct: true |}]

let%expect_test "Symbolic graph: transposed convolution ground matches Direct" =
  let result =
    let open Core.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"convolution_transposed" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 2 2 1) ~name:"x" () in
          let* w = input ~shape:(s 1 1 1 2 2 1) ~name:"w" () in
          convolution ~name:"y" convolution_transposed_params ~x ~weight:w ())
    in
    let prog = Eval_symbolic.run g in
    let x =
      Tensor.materialize (s 1 1 1 2 2 1) (fun c ->
          let h = Dim.to_int (Vec6.get c Axis.H) in
          let w = Dim.to_int (Vec6.get c Axis.W) in
          float_of_int ((h * 2) + w + 1))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 1) (fun _ -> 1.) in
    let inputs = List.combine g.Graph.inputs [ x; w ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=3 W=3 C=1] {1, 3, 2, 4, 10, 6, 3, 7, ...}
    ground matches direct: true |}]
