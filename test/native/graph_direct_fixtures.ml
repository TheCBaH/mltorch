(* Direct evaluation of native graphs: simple sequences, the conv NCHW->NHWC
   decomposition, and a nested subgraph. Each test prints intermediate tensors (not
   just outputs) to show the whole computation. See .ai/native_graph_design.md. *)

open Graph_ir

type error =
  [ `Build of Graph_builder.error
  | `Eval of Eval_direct.error
  | `Missing_named_tensor of string
  | `Shape of Shape_error.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Build e -> Graph_builder.pp_error ppf e
  | `Eval e -> Eval_direct.pp_error ppf e
  | `Missing_named_tensor name ->
      Format.fprintf ppf "missing named tensor %S" name
  | `Shape e -> Shape_error.pp ppf e

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

(* Shared across the split files below: the error/pp plumbing,
   [s]/[s1c]/[chan] shape helpers, and [id_of_name]/[tensor_of_name]/
   [pp_named_tensor]/[pp_named_tensor_pair] for naming and printing a
   tensor by its graph-builder name. [pp_named_tensor_pair] is used by both
   the "sequence add -> relu" test and the max_pool2d_with_indices test, so
   it travels with the rest rather than staying local. [conv_axis] and the
   other conv/param helpers stay local to graph_conv_test.ml; the six
   single-use pp_* helpers (pp_conv_decomp, pp_bias_compare,
   pp_nested_compare, pp_id_consistency, pp_ids, pp_deterministic_ids,
   pp_discard) travel with the one test each of them prints for. *)

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
