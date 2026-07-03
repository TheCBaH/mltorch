(* Graph interpreter for exported `.pt2` models. Walks the ExportedProgram graph
   node by node, decoding each node's named arguments and dispatching to the
   bound ATen ops (Aten_c.Aten_operations). Tensors flow as raw atc_tensor
   handles; SSA value names are kept in an immutable environment that is threaded
   through the node fold.

   Argument decoding, output binding, and the environment live in Interp_decode;
   per-node dispatch (Interp_dispatch.dispatch) is generated from the ATen schema
   by bin/aten_ops_gen. This module is just the driver around them.

   Only the parameters/buffers actually referenced by some node are loaded (see
   [referenced_tensor_names]), so unused buffers (e.g. the int64
   num_batches_tracked, which no node reads) never reach the float32-only
   bridge. *)

open Ctypes
open Pytorch_types
open Schema_runtime
open Interp_decode
open Core.Syntax
module O = Aten_c.Aten_operations
module TG = Aten_types_generated

let tget = Aten_operation_description.tensors_get

type error =
  [ Pt2_archive.error
  | Interp_decode.error
  | `Aten_runtime_failure of string * int
  | `Unhandled_op of string
  | `Unexpected_graph_output of string
  | `Topk_read_failed ]

let pp_error ppf : [< error ] -> unit = function
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | #Interp_decode.error as e -> Interp_decode.pp_error ppf e
  | `Aten_runtime_failure (op, st) ->
      Fmt.pf ppf "ATen op %s failed with status %d" op st
  | `Unhandled_op target -> Fmt.pf ppf "unhandled op %S" target
  | `Unexpected_graph_output summary ->
      Fmt.pf ppf "expected a single tensor graph output, got %s" summary
  | `Topk_read_failed ->
      Fmt.string ppf "failed to read topk outputs back from ATen"

(* --- driver --- *)

(* Every value name used as a tensor argument anywhere in the graph. Parameter
   names appearing here are exactly the ones to load; names that turn out to be
   node results or user inputs are filtered out against the params map. *)
let referenced_tensor_names (g : Graph.t) =
  List.concat_map
    (fun (node : Node.t) ->
      List.concat_map
        (fun (na : NamedArgument.t) ->
          match na.arg with
          | Argument.Tensor ta -> [ ta.TensorArgument.name ]
          | Argument.Optional_tensor (OptionalTensorArgument.Tensor ta) ->
              [ ta.TensorArgument.name ]
          | Argument.Tensors tas ->
              List.map (fun (ta : TensorArgument.t) -> ta.name) tas
          | _ -> [])
        node.Node.inputs)
    g.Graph.nodes

let graph_output_summary outputs =
  match outputs with
  | [ output ] -> Interp_decode.argument_kind_name output
  | outputs -> Printf.sprintf "%d outputs" (List.length outputs)

let run archive (image : Pt2_tensor.t) =
  let prog = Pt2_archive.program archive in
  let g = prog.ExportedProgram.graph_module.GraphModule.graph in
  let sign = prog.ExportedProgram.graph_module.GraphModule.signature in
  (* graph input name -> weight config name *)
  let params =
    List.fold_left
      (fun params (spec : InputSpec.t) ->
        match spec with
        | InputSpec.Parameter p ->
            String_map.add p.InputToParameterSpec.arg.TensorArgument.name
              p.InputToParameterSpec.parameter_name params
        | InputSpec.Buffer b ->
            String_map.add b.InputToBufferSpec.arg.TensorArgument.name
              b.InputToBufferSpec.buffer_name params
        | _ -> params)
      String_map.empty sign.GraphSignature.input_specs
  in
  (* User inputs are bound to the supplied image. *)
  let env =
    List.fold_left
      (fun env (spec : InputSpec.t) ->
        match spec with
        | InputSpec.User_input { UserInputSpec.arg = Argument.Tensor ta; _ } ->
            String_map.add ta.TensorArgument.name (Pt2_aten.to_tensor image) env
        | _ -> env)
      String_map.empty sign.GraphSignature.input_specs
  in
  (* Load exactly the parameters/buffers referenced by some node, so unused
     buffers (e.g. the int64 num_batches_tracked) never reach the float32
     bridge. *)
  let* env =
    Core.List.fold_left
      (fun env name ->
        if String_map.mem name env then Core.return env
        else
          match String_map.find_opt name params with
          | Some cfg_name ->
              let* weight =
                Pt2_archive.load_weight archive cfg_name
                |> Core.map_error (fun e -> (e :> error))
              in
              Core.return (String_map.add name (Pt2_aten.to_tensor weight) env)
          | None -> Core.return env)
      env
      (referenced_tensor_names g)
  in
  let* env =
    Core.List.fold_left
      (fun env node ->
        Interp_dispatch.dispatch env node
        |> Core.map_error (fun e -> (e :> error)))
      env g.Graph.nodes
  in
  match g.Graph.outputs with
  | [ Argument.Tensor ta ] -> resolve env ta.TensorArgument.name
  | outputs ->
      Core.fail (`Unexpected_graph_output (graph_output_summary outputs))

(* Index of the maximum element of the flattened logits, computed by ATen
   (at::argmax with dim=None) and read back as an int. *)
let none_int = from_voidp int64_t null

let argmax logits =
  Core.return (Aten_tensor.item_int (O.argmax logits none_int false))

(* The [k] highest-scoring (class index, probability) pairs of [logits]
   ([1; classes]), descending: softmax over the last dim, then at::topk. *)
let top_predictions logits k =
  let probs = O._softmax logits 1L false in
  let out = make TG.tensors2_struct in
  let st = O.topk probs (Int64.of_int k) (-1L) true true (addr out) in
  if st <> 0 then Core.fail (`Aten_runtime_failure ("topk", st))
  else
    match
      ( Aten_tensor.data Aten_dtype.float32 (tget out TG.tensors2_v0),
        Aten_tensor.data Aten_dtype.int64 (tget out TG.tensors2_v1) )
    with
    | Some vs, Some idx ->
        Core.return
          (List.init (Bigarray.Array1.dim idx) (fun i ->
               (Int64.to_int idx.{i}, vs.{i})))
    | _ -> Core.fail `Topk_read_failed
