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

let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:pp_error

let lift_build (r : ('a, Graph_builder.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> `Build e) r

let lift_eval (r : ('a, Eval_direct.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> `Eval e) r

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n
let chan c = Dim.to_int (Vec6.get c Axis.C)
let p_to_nhwc = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]

let conv_axis ~kernel ~stride ~pad : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad;
    pad_after = Op_config.Nonneg.of_int pad;
    dilation = Op_config.Pos.of_int 1;
  }

let conv_params =
  {
    Conv.Conv2d.h = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    w = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    in_channels = Dim.extent 2;
    groups = Op_config.Pos.of_int 1;
  }

(* Shared across the split files below: the error/pp
   plumbing, [s]/[s1c]/[chan] shape helpers, [p_to_nhwc]/[conv_axis]/
   [conv_params] (used by both the conv and lowering tests), and
   [output_id]/[find_tensor]/[compare_output]/[pp_ground_result] for
   comparing symbolic-ground and Direct results. [p_to_nchw] and the other
   conv/param helpers stay local to graph_symbolic_conv_test.ml, and
   [mp_params] to the pool test, since nothing else uses them. *)

let output_id (g : graph) =
  match g.Graph.outputs with
  | [ id ] -> Err.return id
  | outputs ->
      Err.fail (`Expected_single_output { count = List.length outputs })

let find_tensor env id =
  match Tensor_id.Map.find_opt id env with
  | Some tensor -> Err.return tensor
  | None -> Err.fail (`Missing_output_tensor id)

let compare_output g grounded direct =
  let open Err.Syntax in
  let* id = output_id g in
  let* grounded_out = find_tensor grounded id in
  let* direct_out = find_tensor direct id in
  Err.return (grounded_out, Tensor.equal_bits grounded_out direct_out)

let pp_ground_result name ppf (tensor, matches) =
  Format.fprintf ppf "%s = %a@.ground matches direct: %b" name Tensor.pp tensor
    matches
