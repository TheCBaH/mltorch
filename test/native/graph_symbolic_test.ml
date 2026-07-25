(* Symbolic evaluation of native graphs: build the whole-graph stage DAG, print it
   (each stage's body shows the upstream signatures it loads), then chain-ground it
   and check the result equals Direct evaluation. See .ai/native_graph_design.md. *)

open Graph_ir

type output_count = { count : int }

type error =
  [ `Build of Graph_builder.error
  | `Eval of Eval_direct.error
  | `Expected_single_output of output_count
  | `Missing_output_tensor of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Build e -> Graph_builder.pp_error ppf e
  | `Eval e -> Eval_direct.pp_error ppf e
  | `Expected_single_output { count } ->
      Format.fprintf ppf "graph expected a single output, got %d" count
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
      Core.fail (`Expected_single_output { count = List.length outputs })

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
    inputs: t0, t1
    t2 = (t0[0,0,0,0,0,C] + t1[0,0,0,0,0,C])
    t3 = select((t2[N,T,D,H,W,C] < 0), 0, t2[N,T,D,H,W,C])
    outputs: t3 |}];
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
    inputs: t0, t1
    t2 = (t0[0,0,0,0,0,C] * t1[0,0,0,0,0,C])
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=3] {-3, -5, 4}
    ground matches direct: true |}]

(* A [Discard] node emits NO stage (its producer "dead" still does), so the
   stage DAG lists sum + dead but nothing for the sink; grounding the kept
   output still matches Direct. *)
let%expect_test "Symbolic graph: Discard emits no stage; ground matches Direct"
    =
  let result =
    let open Core.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"discard" ~outputs:(fun (out, _) -> [ out ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          let* out = add ~name:"out" a b in
          let* dead = mul ~name:"dead" a b in
          let* () = discard dead in
          return (out, dead))
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
    inputs: t0, t1
    t2 = (t0[0,0,0,0,0,C] + t1[0,0,0,0,0,C])
    t3 = (t0[0,0,0,0,0,C] * t1[0,0,0,0,0,C])
    outputs: t2 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=3] {-2, -4, 4}
    ground matches direct: true |}]

(* Batch norm symbolically: the per-channel affine over running stats, grounded
   against Direct. *)
let%expect_test "Symbolic graph: batch_norm ground matches Direct" =
  let result =
    let open Core.Syntax in
    let vec2 a0 a1 =
      Tensor.materialize (s1c 2) (fun c -> [| a0; a1 |].(chan c))
    in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"bn" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 2 1 2) ~name:"x" () in
          let* w = input ~shape:(s1c 2) ~name:"w" () in
          let* b = input ~shape:(s1c 2) ~name:"b" () in
          let* rm = input ~shape:(s1c 2) ~name:"rm" () in
          let* rv = input ~shape:(s1c 2) ~name:"rv" () in
          batch_norm ~name:"y"
            { Norm.BatchNorm.channel = Axis.C; eps = 0. }
            ~x ~weight:w ~bias:b ~running_mean:rm ~running_var:rv ())
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 2 1 2) (fun c ->
          [| [| 1.; 3. |]; [| 5.; 7. |] |].(chan c).(Dim.to_int
                                                       (Vec6.get c Axis.H)))
    in
    let inputs =
      List.combine g.Graph.inputs
        [ x; vec2 2. 10.; vec2 1. (-1.); vec2 1. 5.; vec2 4. 4. ]
    in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0, t1, t2, t3, t4
    t5 = ((((t0[N,T,D,H,W,C] - t3[0,0,0,0,0,C]) * (1 / sqrt((t4[0,0,0,0,0,C] + 0)))) * t1[0,0,0,0,0,C]) + t2[0,0,0,0,0,C])
    outputs: t5 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [H=2 W=1 C=2] {1, -1, 3, 9}
    ground matches direct: true |}]

(* max_pool2d_with_indices has two outputs and the index output uses the
   value_of_index bridge + an argmax reduction; its stage DAG is large, so we
   assert Direct==Symbolic on both outputs rather than pin the program text. *)
let mp_params =
  {
    Pool.MaxPool2d.kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
    stride =
      Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
    pad =
      Op_config.Hw.
        { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
  }

let%expect_test "Symbolic graph: max_pool2d_with_indices ground matches Direct"
    =
  let result =
    let open Core.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mp" ~outputs:(fun (v, i) -> [ v; i ])
          @@
          let* x = input ~shape:(s 1 1 1 4 4 1) ~name:"x" () in
          max_pool2d_with_indices mp_params x)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 4 4 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 4)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let inputs = List.combine g.Graph.inputs [ x ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    Core.return
      (List.map
         (fun oid ->
           let d = Tensor_id.Map.find oid direct in
           let gr = Tensor_id.Map.find oid grounded in
           let (Tensor.Tensor r) = d in
           (d, tensors_match r.shape d gr))
         g.Graph.outputs)
  in
  (match result with
  | Ok outs ->
      List.iter
        (fun (t, m) ->
          Format.printf "%a  ground matches direct: %b@." Tensor.pp t m)
        outs
  | Error e -> Format.printf "%a@." pp_error e.Core.Error.kind);
  [%expect
    {|
    inputs: t0
    t1 = max_pool2d_value(t0; k=2x2 s=2x2 p=0x0; out=[N,T,D,H,W,C])
    t2 = max_pool2d_index(t0; k=2x2 s=2x2 p=0x0; out=[N,T,D,H,W,C])
    outputs: t1, t2
    tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15}  ground matches direct: true
    tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15}  ground matches direct: true |}]

(* Reshape symbolically: the index expression carries div/mod over the flat
   offset (the value_of_index-free delinearize path); ground must match Direct. *)
let%expect_test "Symbolic graph: reshape ground matches Direct" =
  let result =
    let open Core.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"reshape" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          reshape ~name:"out" { Reshape.Reshape.shape = s 1 1 1 1 1 6 } x)
    in
    let prog = Eval_symbolic.run g in
    Format.printf "%a@." Stage_program.pp prog;
    let x =
      Tensor.materialize (s 1 1 1 2 3 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 3)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let inputs = List.combine g.Graph.inputs [ x ] in
    let bind id = List.assoc id inputs in
    let grounded = Stage_program.ground prog ~bind in
    let* direct = lift_eval (Eval_direct.run g ~inputs) in
    compare_output g grounded direct
  in
  [%expect
    {|
    inputs: t0
    t1 = t0[floor_div(6*1*1*1*1*1*0+N+T+D+H+W+C,6)+-1*floor_div(6*1*1*1*1*1*0+N+T+D+H+W+C,6),floor_div(6*1*1*1*1*1*0+N+T+D+H+W+C,6)+-1*floor_div(6*1*1*1*1*1*0+N+T+D+H+W+C,6),floor_div(6*1*1*1*1*1*0+N+T+D+H+W+C,6)+-1*floor_div(6*1*1*1*1*1*0+N+T+D+H+W+C,6),floor_div(6*1*1*1*1*1*0+N+T+D+H+W+C,3)+-2*floor_div(floor_div(6*1*1*1*1*1*0+N+T+D+H+W+C,3),2),6*1*1*1*1*1*0+N+T+D+H+W+C+-3*floor_div(6*1*1*1*1*1*0+N+T+D+H+W+C,3),6*1*1*1*1*1*0+N+T+D+H+W+C+-1*6*1*1*1*1*1*0+N+T+D+H+W+C]
    outputs: t1 |}];
  Format.printf "%a@." (pp_result (pp_ground_result "ground")) result;
  [%expect
    {|
    ground = tensor f32 [C=6] {0, 1, 2, 3, 4, 5}
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
    inputs: t0, t1, t2
    t3 = t0[N,T,D,C,H,W]
    t4 = (sum(r1=0..2: sum(r2=max(0,-1*1*H+0)..min(2,3+-1+-1*1*H+0+1): sum(r3=max(0,-1*1*W+0)..min(2,3+-1+-1*1*W+0+1): (t3[N,T,D,1*H+0+1*r2,1*W+0+1*r3,2*0+r1] * t1[C,0,0,r2,r3,r1])))) + t2[0,0,0,0,0,C])
    t5 = t4[N,T,D,W,C,H]
    outputs: t5 |}];
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
    inputs: t0, t1
    t2 = (sum(r1=0..1: sum(r2=max(0,-1*1*H+0)..min(2,3+-1+-1*1*H+0+1): sum(r3=max(0,-1*1*W+0)..min(2,3+-1+-1*1*W+0+1): (t0[N,T,D,1*H+0+1*r2,1*W+0+1*r3,1*0+r1] * t1[C,0,0,r2,r3,r1])))) + t3[0,0,0,0,0,C])
    outputs: t2 |}];
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
    inputs: t0, t1
    t2 = (sum(r1=0..1: sum(r2=max(0,-1*1*H+0)..min(2,3+-1+-1*1*H+0+1): sum(r3=max(0,-1*1*W+0)..min(2,3+-1+-1*1*W+0+1): (t0[N,T,D,1*H+0+1*r2,1*W+0+1*r3,1*0+r1] * t1[C,0,0,r2,r3,r1])))) + t3[0,0,0,0,0,C])
    outputs: t2 |}];
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
