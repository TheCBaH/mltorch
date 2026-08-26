(* Direct evaluation of native graphs: simple sequences, the conv NCHW->NHWC
   decomposition, and a nested subgraph. Each test prints intermediate tensors (not
   just outputs) to show the whole computation. See .ai/native_graph_design.md. *)

open Graph_ir

type error =
  [ `Build of Graph_builder.error
  | `Eval of Eval_direct.error
  | `Shape of Shape_error.t
  | `Missing_named_tensor of string ]

let pp_error ppf : [< error ] -> unit = function
  | `Build e -> Graph_builder.pp_error ppf e
  | `Eval e -> Eval_direct.pp_error ppf e
  | `Shape e -> Shape_error.pp ppf e
  | `Missing_named_tensor name ->
      Format.fprintf ppf "missing named tensor %S" name

let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:pp_error

let lift_build (r : ('a, Graph_builder.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> `Build e) r

let lift_eval (r : ('a, Eval_direct.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> `Eval e) r

let lift_shape (r : ('a, Shape_error.t) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> `Shape e) r

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

let id_of_name (g : graph) name =
  let node_output i =
    match List.nth_opt g.Graph.nodes i with
    | Some { Node.outputs = id :: _; _ } -> Some id
    | _ -> None
  in
  let id =
    match name with
    | "sum" | "x_nhwc" -> node_output 0
    | "dead" | "y_nhwc" -> node_output 1
    | _ -> ( match g.Graph.outputs with id :: _ -> Some id | [] -> None)
  in
  match id with
  | Some id -> Err.return id
  | None -> Err.fail (`Missing_named_tensor name)

let tensor_of_name (g : graph) env name =
  let open Err.Syntax in
  let* id = id_of_name g name in
  match Tensor_id.Map.find_opt id env with
  | Some tensor -> Err.return tensor
  | None -> Err.fail (`Missing_named_tensor name)

let pp_named_tensor name ppf tensor =
  Format.fprintf ppf "%s = %a" name Tensor.pp tensor

let pp_named_tensor_pair name1 name2 ppf (tensor1, tensor2) =
  Format.fprintf ppf "%a@.%a" (pp_named_tensor name1) tensor1
    (pp_named_tensor name2) tensor2

let pp_conv_decomp ppf (x_nhwc, y_nhwc, y_nchw, matches) =
  Format.fprintf ppf "%a@.%a@.%a@.y_nhwc matches single conv: %b"
    (pp_named_tensor "x_nhwc") x_nhwc (pp_named_tensor "y_nhwc") y_nhwc
    (pp_named_tensor "y_nchw") y_nchw matches

let pp_bias_compare ppf (y_no_bias, matches) =
  Format.fprintf ppf "y (no bias) = %a@.matches explicit zero bias: %b"
    Tensor.pp y_no_bias matches

let pp_nested_compare ppf (nested_out, matches) =
  Format.fprintf ppf "nested_out = %a@.matches flat: %b" Tensor.pp nested_out
    matches

let pp_id_consistency ppf g =
  Tensor_id.Map.iter
    (fun tid sg ->
      Format.fprintf ppf "tid=%a sig_id=%a match=%b@." Tensor_id.pp tid
        Tensor_id.pp sg.Tensor_sig.id
        (Tensor_id.equal tid sg.Tensor_sig.id))
    g.Graph.tensors

let pp_ids ids =
  ids
  |> List.map (fun (t, s) ->
      Format.asprintf "(%a,%a)" Tensor_id.pp t Tensor_id.pp s)
  |> String.concat " "

let pp_deterministic_ids ppf (ids1, ids2, matches) =
  Format.fprintf ppf "g1 ids: %s@.g2 ids: %s@.match: %b" (pp_ids ids1)
    (pp_ids ids2) matches

let%expect_test "Direct graph: sequence add -> relu (with intermediate)" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"seq" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          let* b = input ~shape:(s1c 4) ~name:"b" () in
          let* t = add ~name:"sum" a b in
          relu ~name:"out" t)
    in
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| 1.; -5.; 2.; -8. |].(chan c))
    in
    let b =
      Tensor.materialize (s1c 4) (fun c -> [| -3.; 1.; 2.; 3. |].(chan c))
    in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    let* sum = tensor_of_name g env "sum" in
    let* out = tensor_of_name g env "out" in
    Err.return (sum, out)
  in
  Format.printf "%a@." (pp_result (pp_named_tensor_pair "sum" "out")) result;
  [%expect
    {|
    sum = tensor f32 [C=4] {-2, -4, 4, -5}
    out = tensor f32 [C=4] {0, 0, 4, 0} |}]

let%expect_test "Direct graph: captured constant is bound by tensor id" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"captured" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 3) ~name:"x" () in
          let* weight = constant ~shape:(s1c 3) ~name:"weight" () in
          add ~name:"out" x weight)
    in
    let x = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c + 1)) in
    let weight = Tensor.materialize (s1c 3) (fun _ -> 10.) in
    let* env =
      lift_eval
        (Eval_direct.run g
           ~inputs:[ (List.hd g.Graph.inputs, x) ]
           ~constants:[ (List.nth g.Graph.inputs 1, weight) ])
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  [%expect {| tensor f32 [C=3] {11, 12, 13} |}]

let%expect_test "Direct graph: add of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"add" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          add ~name:"out" a b)
    in
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun _ -> 10.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=3] {10, 11, 12} |}]

let%expect_test "Direct graph: concat of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"concat" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 2) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          concat ~name:"out" { Concat.Concat.axis = Axis.C } [ a; b ])
    in
    let a = Tensor.materialize (s1c 2) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun c -> float_of_int (10 + chan c)) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=5] {0, 1, 10, 11, 12} |}]

(* Stack inserts a new axis per operand, unlike Concat above which joins an
   existing one -- the fact worth pinning here is that a single [Stack] node
   appears, not N [Reshape]s plus a [Concat]. *)
let%expect_test "Direct graph: stack of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"stack" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          stack ~name:"out" { Concat.Stack.axis = Axis.W } [ a; b ])
    in
    Format.printf "%a@." Graph_ir.pp g;
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun c -> float_of_int (10 + chan c)) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [C=3] ->[n0], t1 f32 [C=3] ->[n0]]
    nodes:
      n0: [t2 f32 [W=2 C=3]] = stack xs=[t0, t1] params={axis=W}
    outputs: [t2 f32 [W=2 C=3] <-n0]
    out = tensor f32 [W=2 C=3] {0, 1, 2, 10, 11, 12} |}]

let%expect_test "Direct graph: div of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"div" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          div ~name:"out" a b)
    in
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c) +. 1.) in
    let b = Tensor.materialize (s1c 3) (fun _ -> 4.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=3] {0.25, 0.5, 0.75} |}]

let%expect_test "Direct graph: mul of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mul" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          mul ~name:"out" a b)
    in
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun _ -> 10.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=3] {0, 10, 20} |}]

let%expect_test "Direct graph: sqrt of an input" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"sqrt" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          sqrt ~name:"out" a)
    in
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| 0.; 1.; 4.; 2.25 |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=4] {0, 1, 2, 1.5} |}]

let%expect_test "Direct graph: sub of two inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"sub" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          sub ~name:"out" a b)
    in
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun _ -> 10.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=3] {-10, -9, -8} |}]

(* MobileNet-v3's hardsigmoid, as the exporter serialises it: the +3 and /6 are
   compile-time scalars in Tensor slots, so they lower to the scalar-parameter
   ops rather than to edges that would need binding. *)
let%expect_test "Direct graph: hardsigmoid chain (add_scalar/clamp/div_scalar)"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"hardsigmoid" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 5) ~name:"a" () in
          let* s = add_scalar 3. a in
          let* lo = clamp { Pointwise.Clamp.min = Some 0.; max = None } s in
          let* hi = clamp { Pointwise.Clamp.min = None; max = Some 6. } lo in
          div_scalar ~name:"out" 6. hi)
    in
    let a =
      Tensor.materialize (s1c 5) (fun c -> [| -4.; -3.; 0.; 3.; 4. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=5] {0, 0, 0.5, 1, 1} |}]

let%expect_test "Direct graph: hardtanh and clone" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"hardtanh_clone" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          let* h =
            hardtanh { Pointwise.Hardtanh.min_val = 0.; max_val = 6. } a
          in
          clone ~name:"out" h)
    in
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -3.; 0.; 2.5; 7. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=4] {0, 0, 2.5, 6} |}]

let%expect_test "Direct graph: silu" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"silu" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          silu ~name:"out" a)
    in
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {| out = tensor f32 [C=4] {-0.0148357, -0.18877, 0.31123, 5.98516} |}]

let%expect_test "Direct graph: sigmoid" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"sigmoid" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          sigmoid ~name:"out" a)
    in
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {| out = tensor f32 [C=4] {0.00247262, 0.377541, 0.622459, 0.997527} |}]

let%expect_test "Direct graph: gelu" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"gelu" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          gelu ~name:"out" Pointwise.Gelu.Exact a)
    in
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=4] {-5.94073e-09, -0.154269, 0.345731, 6} |}]

let%expect_test "Direct graph: mul_scalar" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mul_scalar" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          mul_scalar ~name:"out" 3. a)
    in
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c) +. 1.) in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=3] {3, 6, 9} |}]

let%expect_test "Direct graph: hardsigmoid" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"hardsigmoid" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          hardsigmoid ~name:"out" a)
    in
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=4] {0, 0.416667, 0.583333, 1} |}]

let%expect_test "Direct graph: hardswish" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"hardswish" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 4) ~name:"a" () in
          hardswish ~name:"out" a)
    in
    let a =
      Tensor.materialize (s1c 4) (fun c -> [| -6.; -0.5; 0.5; 6. |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=4] {-0, -0.208333, 0.291667, 6} |}]

(* The both-absent clamp is rejected where the node is built, so the error
   surfaces from the builder rather than from evaluation. *)
let%expect_test "Direct graph: clamp with no bounds fails to build" =
  let result =
    let open Err.Syntax in
    let+ g =
      lift_build
        Graph_builder.(
          build ~name:"clamp" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          clamp { Pointwise.Clamp.min = None; max = None } a)
    in
    Format.asprintf "%a" Graph_ir.pp g
  in
  Format.printf "%a@." (pp_result Format.pp_print_string) result;
  [%expect {| clamp: at least one of 'min' or 'max' must be given |}]

(* A [Discard] sink consumes a dead edge and produces no output. The graph still
   evaluates, the discarded producer ("dead") is still materialised in the env
   (so tools can inspect it), and the node prints with an empty output list. *)
let pp_discard ppf (g, out, dead) =
  Format.fprintf ppf "%a@.out = %a@.dead = %a" Graph_ir.pp g Tensor.pp out
    Tensor.pp dead

let%expect_test "Direct graph: Discard sink (dead edge still materialised)" =
  let result =
    let open Err.Syntax in
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
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun _ -> 10.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    let* out = tensor_of_name g env "out" in
    let* dead = tensor_of_name g env "dead" in
    Err.return (g, out, dead)
  in
  Format.printf "%a@." (pp_result pp_discard) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [C=3] ->[n0, n1], t1 f32 [C=3] ->[n0, n1]]
    nodes:
      n0: [t2 f32 [C=3]] = add a=t0 b=t1
      n1: [t3 f32 [C=3] ->[n2]] = mul a=t0 b=t1
      n2: [] = discard x=t3 <-n1
    outputs: [t2 f32 [C=3] <-n0]
    out = tensor f32 [C=3] {10, 11, 12}
    dead = tensor f32 [C=3] {0, 10, 20} |}]

(* Batch norm applies per-channel affine using running stats: with mean=[1,5],
   var=[4,4] (inv=0.5) and weight/bias [2,10]/[1,-1], y = (x-mean)*0.5*w+b. *)
let%expect_test "Direct graph: batch_norm per-channel over C" =
  let result =
    let open Err.Syntax in
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
    let x =
      Tensor.materialize (s 1 1 1 2 1 2) (fun c ->
          [| [| 1.; 3. |]; [| 5.; 7. |] |].(chan c).(Dim.to_int
                                                       (Vec6.get c Axis.H)))
    in
    let inputs =
      List.combine g.Graph.inputs
        [ x; vec2 2. 10.; vec2 1. (-1.); vec2 1. 5.; vec2 4. 4. ]
    in
    let* env = lift_eval (Eval_direct.run g ~inputs) in
    tensor_of_name g env "y"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "y")) result;
  [%expect {| y = tensor f32 [H=2 W=1 C=2] {1, -1, 3, 9} |}]

(* max_pool2d_with_indices produces TWO outputs — exercising the multi-output
   eval loop. value(h,w)=h*4+w; each 2x2/stride-2 window's max is its
   bottom-right corner, and the argmax index is that corner's flat position. *)
let mp_params =
  {
    Pool.MaxPool2d.ceil_mode = false;
    kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
    stride =
      Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
    pad =
      Op_config.Hw.
        { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
  }

let%expect_test "Direct graph: max_pool2d_with_indices (two outputs)" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mp" ~outputs:(fun (v, i) -> [ v; i ])
          @@
          let* x = input ~shape:(s 1 1 1 4 4 1) ~name:"x" () in
          max_pool2d_with_indices ~name:"vals" mp_params x)
    in
    let x =
      Tensor.materialize (s 1 1 1 4 4 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 4)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    match g.Graph.outputs with
    | [ vid; iid ] ->
        Err.return (Tensor_id.Map.find vid env, Tensor_id.Map.find iid env)
    | _ -> Err.fail (`Missing_named_tensor "two outputs")
  in
  Format.printf "%a@."
    (pp_result (pp_named_tensor_pair "values" "indices"))
    result;
  [%expect
    {|
    values = tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15}
    indices = tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15} |}]

(* Reshape reinterprets the same flat buffer under a new shape (contiguous). *)
let%expect_test "Direct graph: reshape [H=2 W=3 C=1] -> [C=6] (flatten)" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"reshape" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          reshape ~name:"out" { Reshape.Reshape.shape = s 1 1 1 1 1 6 } x)
    in
    let x =
      Tensor.materialize (s 1 1 1 2 3 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 3)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=6] {0, 1, 2, 3, 4, 5} |}]

(* Conv decomposition. The input is laid out NCHW: in the 6D frame its channel sits
   on H, spatial-H on W, spatial-W on C. Two permutes bracket a native (NHWC)
   conv: NCHW->NHWC moves the channel to C and the spatial axes to H/W; NHWC->NCHW
   is its inverse. *)
let conv_params =
  {
    Conv.Conv2d.h = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    w = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    in_channels = Dim.extent 2;
    groups = Op_config.Pos.of_int 1;
  }

let p_to_nhwc = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]
let p_to_nchw = Axis.[ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

let%expect_test
    "Direct graph: conv NCHW -> permute -> NHWC conv -> permute -> NCHW" =
  let result =
    let open Err.Syntax in
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
    let x =
      Tensor.materialize (s 1 1 1 2 3 3) (fun c ->
          let chan = Dim.to_int (Vec6.get c Axis.H) in
          let sph = Dim.to_int (Vec6.get c Axis.W) in
          let spw = Dim.to_int (Vec6.get c Axis.C) in
          float_of_int ((chan * 100) + (sph * 10) + spw))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 2) (fun _ -> 1.) in
    let b = Tensor.materialize (s1c 1) (fun _ -> 0.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x; w; b ]))
    in
    let* x_nhwc = tensor_of_name g env "x_nhwc" in
    let* y_nhwc = tensor_of_name g env "y_nhwc" in
    let* y_nchw = tensor_of_name g env "y_nchw" in
    let module Cv = Conv.Conv2d.Compute (Direct) in
    let (Tensor.Tensor r) = x_nhwc in
    let* ref_shape =
      lift_shape
        (Conv.Conv2d.output_shape ~x_shape:r.shape ~weight_shape:(s 1 1 1 2 2 2)
           conv_params)
    in
    let ref_y =
      Schedule.evaluate ref_shape
        (Cv.pixel conv_params ~x_shape:r.shape ~weight_shape:(s 1 1 1 2 2 2)
           ~x:x_nhwc ~weight:w ~bias:b)
    in
    Err.return (x_nhwc, y_nhwc, y_nchw, Tensor.equal_bits ref_y y_nhwc)
  in
  Format.printf "%a@." (pp_result pp_conv_decomp) result;
  [%expect
    {|
    x_nhwc = tensor f32 [H=3 W=3 C=2] {0, 100, 1, 101, 2, 102, 10, 110, ...}
    y_nhwc = tensor f32 [H=2 W=2 C=1] {444, 452, 524, 532}
    y_nchw = tensor f32 [W=2 C=2] {444, 452, 524, 532}
    y_nhwc matches single conv: true |}]

(* Conv with no bias: omit [?bias] so the IR carries [bias = None]; the evaluator
   fills a zeros bias. The result must equal the same conv with an explicit zero
   bias. *)
let%expect_test "Direct graph: conv with optional bias omitted (None -> zeros)"
    =
  let build_one ~with_bias =
    Graph_builder.(
      build ~name:"conv" ~outputs:(fun y -> [ y ])
      @@
      let* x = input ~shape:(s 1 1 1 3 3 2) ~name:"x" () in
      let* w = input ~shape:(s 1 1 1 2 2 2) ~name:"w" () in
      if with_bias then
        let* b = input ~shape:(s1c 1) ~name:"b" () in
        conv2d ~name:"y" conv_params ~x ~weight:w ~bias:b ()
      else conv2d ~name:"y" conv_params ~x ~weight:w ())
  in
  let result =
    let open Err.Syntax in
    let x =
      Tensor.materialize (s 1 1 1 3 3 2) (fun c ->
          float_of_int (Dim.to_int (Vec6.get c Axis.H) + chan c))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 2) (fun _ -> 1.) in
    let b = Tensor.materialize (s1c 1) (fun _ -> 0.) in
    let* g_no = lift_build (build_one ~with_bias:false) in
    let* g_yes = lift_build (build_one ~with_bias:true) in
    let* env_no =
      lift_eval
        (Eval_direct.run g_no ~inputs:(List.combine g_no.Graph.inputs [ x; w ]))
    in
    let* env_yes =
      lift_eval
        (Eval_direct.run g_yes
           ~inputs:(List.combine g_yes.Graph.inputs [ x; w; b ]))
    in
    let* y_no = tensor_of_name g_no env_no "y" in
    let* y_yes = tensor_of_name g_yes env_yes "y" in
    Err.return (y_no, Tensor.equal_bits y_no y_yes)
  in
  Format.printf "%a@." (pp_result pp_bias_compare) result;
  [%expect
    {|
    y (no bias) = tensor f32 [H=2 W=2 C=1] {8, 8, 16, 16}
    matches explicit zero bias: true |}]

let%expect_test "Direct graph: transposed convolution" =
  let params =
    {
      Conv.Convolution.stride =
        { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      padding = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
      dilation = { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      transposed = true;
      output_padding =
        { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
      groups = Op_config.Pos.of_int 1;
    }
  in
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"transposed" ~outputs:(fun y -> [ y ])
          @@
          let* x = input ~shape:(s 1 1 1 2 2 1) ~name:"x" () in
          let* w = input ~shape:(s 1 1 1 2 2 1) ~name:"w" () in
          convolution ~name:"y" params ~x ~weight:w ())
    in
    let x =
      Tensor.materialize (s 1 1 1 2 2 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 2)
            + Dim.to_int (Vec6.get c Axis.W)
            + 1))
    in
    let w = Tensor.materialize (s 1 1 1 2 2 1) (fun _ -> 1.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x; w ]))
    in
    tensor_of_name g env "y"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "y")) result;
  [%expect {| y = tensor f32 [H=3 W=3 C=1] {1, 3, 2, 4, 10, 6, 3, 7, ...} |}]

(* A structural group computes add -> relu in the parent graph's global SSA
   namespace. Its result must match the same computation built flat. *)
let%expect_test "Direct graph: grouped evaluation" =
  let result =
    let open Err.Syntax in
    let* nested =
      lift_build
        Graph_builder.(
          build ~name:"outer" ~outputs:(fun outs -> outs)
          @@
          let* x = input ~shape:(s1c 4) ~name:"x" () in
          let* y = input ~shape:(s1c 4) ~name:"y" () in
          group ~label:"add_relu"
            (let* t = add ~name:"sum" x y in
             let* r = relu ~name:"r" t in
             return [ r ]))
    in
    let* flat =
      lift_build
        Graph_builder.(
          build ~name:"flat" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 4) ~name:"x" () in
          let* y = input ~shape:(s1c 4) ~name:"y" () in
          let* t = add x y in
          relu ~name:"out" t)
    in
    let x =
      Tensor.materialize (s1c 4) (fun c -> [| 1.; -5.; 2.; -8. |].(chan c))
    in
    let y =
      Tensor.materialize (s1c 4) (fun c -> [| -3.; 1.; 2.; 3. |].(chan c))
    in
    let* env_n =
      lift_eval
        (Eval_direct.run nested
           ~inputs:(List.combine nested.Graph.inputs [ x; y ]))
    in
    let* env_f =
      lift_eval
        (Eval_direct.run flat ~inputs:(List.combine flat.Graph.inputs [ x; y ]))
    in
    let* out_n = tensor_of_name nested env_n "nested_out" in
    let* out_f = tensor_of_name flat env_f "out" in
    Err.return (out_n, Tensor.equal_bits out_n out_f)
  in
  Format.printf "%a@." (pp_result pp_nested_compare) result;
  [%expect
    {|
    nested_out = tensor f32 [C=4] {0, 0, 4, 0}
    matches flat: true |}]

(* Tensor_sig.id must equal the Tensor_id key in the graph's tensor map.
   This verifies the state-monad id generator produces consistent ids. *)
let%expect_test "Builder: Tensor_sig.id equals Tensor_id for all edges" =
  let result =
    lift_build
      Graph_builder.(
        build ~name:"check" ~outputs:(fun r -> [ r ])
        @@
        let* a = input ~shape:(s1c 2) ~name:"a" () in
        let* b = input ~shape:(s1c 2) ~name:"b" () in
        add ~name:"out" a b)
  in
  Format.printf "%a@." (pp_result pp_id_consistency) result;
  [%expect
    {|
    tid=t0 sig_id=t0 match=true
    tid=t1 sig_id=t1 match=true
    tid=t2 sig_id=t2 match=true |}]

(* Building the same graph twice must produce identical tensor ids: the id
   generator is local to [build], not a global counter. *)
let%expect_test "Builder: ids are deterministic across multiple build calls" =
  let build_add () =
    Graph_builder.(
      build ~name:"add" ~outputs:(fun r -> [ r ])
      @@
      let* a = input ~shape:(s1c 3) ~name:"a" () in
      let* b = input ~shape:(s1c 3) ~name:"b" () in
      add ~name:"out" a b)
  in
  let result =
    let open Err.Syntax in
    let* g1 = lift_build (build_add ()) in
    let* g2 = lift_build (build_add ()) in
    let ids_of g =
      Tensor_id.Map.bindings g.Graph.tensors
      |> List.map (fun (tid, sg) -> (tid, sg.Tensor_sig.id))
    in
    Err.return (ids_of g1, ids_of g2, ids_of g1 = ids_of g2)
  in
  Format.printf "%a@." (pp_result pp_deterministic_ids) result;
  [%expect
    {|
    g1 ids: (t0,t0) (t1,t1) (t2,t2)
    g2 ids: (t0,t0) (t1,t1) (t2,t2)
    match: true |}]

(* A node whose output ARITY comes from its input signature rather than from the
   op: three outputs here only because W's extent is 3. Every one is evaluated,
   so the variable-arity path through [Graph_shape] -> builder -> [Eval_direct]'s
   per-ordinal loop is exercised end to end rather than at output 0. *)
let%expect_test "Direct graph: unbind is a variable-arity node" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"ub" ~outputs:Fun.id
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          unbind ~name:"slice" { Split.Unbind.axis = Axis.W } x)
    in
    let x =
      Tensor.materialize (s 1 1 1 2 3 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    Err.return
      (List.map (fun oid -> Tensor_id.Map.find oid env) g.Graph.outputs)
  in
  (match result with
  | Ok outs ->
      List.iteri (fun i t -> Format.printf "out%d = %a@." i Tensor.pp t) outs
  | Error e -> Format.printf "%a@." pp_error (Err.Error.kind e));
  [%expect
    {|
    out0 = tensor f32 [W=2 C=1] {0, 10}
    out1 = tensor f32 [W=2 C=1] {1, 11}
    out2 = tensor f32 [W=2 C=1] {2, 12} |}]

(* Same variable-arity path as unbind above, except the arity comes from
   [params.sizes]'s length rather than the input's own extent, and every
   output KEEPS the split axis instead of dropping it. *)
let%expect_test "Direct graph: split_with_sizes divides one axis into windows" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"sws" ~outputs:Fun.id
          @@
          let* x = input ~shape:(s 1 1 1 2 5 1) ~name:"x" () in
          split_with_sizes ~name:"pieces"
            { Split.Split_with_sizes.axis = Axis.W; sizes = [ 2; 3 ] }
            x)
    in
    let x =
      Tensor.materialize (s 1 1 1 2 5 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    Err.return
      (List.map (fun oid -> Tensor_id.Map.find oid env) g.Graph.outputs)
  in
  (match result with
  | Ok outs ->
      List.iteri (fun i t -> Format.printf "out%d = %a@." i Tensor.pp t) outs
  | Error e -> Format.printf "%a@." pp_error (Err.Error.kind e));
  [%expect
    {|
    out0 = tensor f32 [H=2 W=2 C=1] {0, 1, 10, 11}
    out1 = tensor f32 [H=2 W=3 C=1] {2, 3, 4, 12, 13, 14} |}]

(* Through the builder with BOTH affine operands absent, so this exercises
   [Eval_op]'s fill path (identity weight=1, bias=0) end to end -- the same
   [H=2 W=1 C=4] two-group fixture [compute_test.ml]'s hand-computed test
   uses, so the two can be cross-checked against each other. *)
let%expect_test "Direct graph: group_norm splits C into groups, absent affine" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"gn" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 1 4) ~name:"x" () in
          group_norm ~name:"out"
            {
              Norm.GroupNorm.channel = Axis.C;
              groups = Op_config.Pos.of_int 2;
              eps = 0.;
            }
            ~x ())
    in
    let x =
      Tensor.materialize (s 1 1 1 2 1 4) (fun c ->
          float_of_int ((Dim.to_int (Vec6.get c Axis.H) * 10) + chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    out = tensor f32 [H=2 W=1 C=4] {-1.09454, -0.895533, -1.09454, -0.895533, 0.895533, 1.09454, 0.895533, 1.09454} |}]

(* Pad through the builder: the fill is narrowed to f32 there, and the shape
   rule sees the input edge, so a graph whose pad empties an axis cannot be
   built at all. *)
let%expect_test "Direct graph: pad reflect on W, constant fill on H" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"pad" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          pad ~name:"out"
            {
              Pad.Pad.pads = [ (Axis.W, { Pad.Pad.before = 1; after = 1 }) ];
              mode = Pad.Pad.Reflect;
            }
            x)
    in
    Format.printf "%a@." Graph_ir.pp g;
    let x =
      Tensor.materialize (s 1 1 1 2 3 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=3 C=1] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=5 C=1]] = pad x=t0 params={pads=[W:1,1] mode=reflect}
    outputs: [t1 f32 [H=2 W=5 C=1] <-n0]
    out = tensor f32 [H=2 W=5 C=1] {1, 0, 1, 2, 1, 11, 10, 11, ...} |}]

let%expect_test "Direct graph: a pad that empties an axis is not buildable" =
  let result =
    let open Err.Syntax in
    let+ g =
      lift_build
        Graph_builder.(
          build ~name:"pad" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          pad ~name:"out"
            {
              Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = -2; after = 0 }) ];
              mode = Pad.Pad.Constant 0.;
            }
            x)
    in
    Format.asprintf "%a" Graph_ir.pp g
  in
  Format.printf "%a@." (pp_result Format.pp_print_string) result;
  [%expect
    {| pad of axis H by (-2, 0) over extent 2 leaves 0 elements; the engine has no empty extent |}]

(* Slice through the builder. The bounds are canonical by contract, so the two
   graph-level facts worth pinning are that the shape rule sees the input edge
   (and so refuses an empty or out-of-range selection at BUILD time, where an
   importer's own check would already have run) and that the rank survives. *)
let%expect_test "Direct graph: slice narrows one axis and keeps the rank" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"slice" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 5 1) ~name:"x" () in
          slice ~name:"out"
            {
              Split.Slice.axis = Axis.W;
              start = 1;
              stop = 5;
              step = Op_config.Pos.of_int 2;
            }
            x)
    in
    Format.printf "%a@." Graph_ir.pp g;
    let x =
      Tensor.materialize (s 1 1 1 2 5 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=5 C=1] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=2 C=1]] =
        slice x=t0 params={axis=W start=1 stop=5 step=2}
    outputs: [t1 f32 [H=2 W=2 C=1] <-n0]
    out = tensor f32 [H=2 W=2 C=1] {1, 3, 11, 13} |}]

(* Select drops the axis it picks along, unlike Slice above, so the graph's
   output rank is one less than the input's -- the fact worth pinning here is
   that a single [Select] node appears, not a [Slice]+[Reshape]
   pair. *)
let%expect_test "Direct graph: select drops one axis" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"select" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 5 1) ~name:"x" () in
          select ~name:"out" { Split.Select.axis = Axis.W; index = 3 } x)
    in
    Format.printf "%a@." Graph_ir.pp g;
    let x =
      Tensor.materialize (s 1 1 1 2 5 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=5 C=1] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=1]] = select x=t0 params={axis=W index=3}
    outputs: [t1 f32 [W=2 C=1] <-n0]
    out = tensor f32 [W=2 C=1] {3, 13} |}]

let%expect_test "Direct graph: an empty slice is not buildable" =
  let result =
    let open Err.Syntax in
    let+ g =
      lift_build
        Graph_builder.(
          build ~name:"slice" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          slice ~name:"out"
            {
              Split.Slice.axis = Axis.W;
              start = 2;
              stop = 2;
              step = Op_config.Pos.of_int 1;
            }
            x)
    in
    Format.asprintf "%a" Graph_ir.pp g
  in
  Format.printf "%a@." (pp_result Format.pp_print_string) result;
  [%expect
    {| slice of axis W [2, 2) step 1 over extent 3 selects 0 elements; the engine has no empty extent |}]
