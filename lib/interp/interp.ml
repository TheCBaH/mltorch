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
open Err.Syntax
module O = Aten_c.Aten_operations
module TG = Aten_types_generated

let tget = Aten_operation_description.tensors_get

type error =
  [ `Aten_runtime_failure of string * int
  | `Input_metadata of string * string
  | `Input_names of string list * string list
  | `Metadata_mismatch of string
  | Interp_decode.error
  | Pt2_archive.error
  | `Topk_read_failed
  | `Unexpected_graph_input of string
  | `Unexpected_graph_output of string
  | `Unhandled_op of string ]

let pp_error ppf : [< error ] -> unit = function
  | `Aten_runtime_failure (op, st) ->
      Fmt.pf ppf "ATen op %s failed with status %d" op st
  | `Input_metadata (name, field) ->
      Fmt.pf ppf "ATen input %s metadata differs: %s" name field
  | `Input_names (expected, actual) ->
      Fmt.pf ppf "ATen input names: expected [%s], got [%s]"
        (String.concat "; " expected)
        (String.concat "; " actual)
  | `Metadata_mismatch field ->
      Fmt.pf ppf "ATen tensor metadata differs: %s" field
  | #Interp_decode.error as e -> Interp_decode.pp_error ppf e
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | `Topk_read_failed ->
      Fmt.string ppf "failed to read topk outputs back from ATen"
  | `Unexpected_graph_input kind ->
      Fmt.pf ppf "unsupported graph input: %s" kind
  | `Unexpected_graph_output summary ->
      Fmt.pf ppf "expected a single tensor graph output, got %s" summary
  | `Unhandled_op target -> Fmt.pf ppf "unhandled op %S" target

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
          | Argument.Optional_tensors tas ->
              List.filter_map
                (function
                  | OptionalTensorArgument.None _ -> None
                  | OptionalTensorArgument.Tensor ta ->
                      Some ta.TensorArgument.name)
                tas
          | Argument.Tensors tas ->
              List.map (fun (ta : TensorArgument.t) -> ta.name) tas
          | _ -> [])
        node.Node.inputs)
    g.Graph.nodes

let graph_output_summary outputs =
  match outputs with
  | [ output ] -> Interp_decode.argument_kind_name output
  | outputs -> Printf.sprintf "%d outputs" (List.length outputs)

(* The graph and its I/O signature out of an opened archive — the starting
   point shared by [run] and any other node-by-node walk (e.g.
   Native_graph). *)
let graph_and_signature archive =
  let prog = Pt2_archive.program archive in
  ( prog.ExportedProgram.graph_module.GraphModule.graph,
    prog.ExportedProgram.graph_module.GraphModule.signature )

(* graph input name -> captured payload name, for every tensor source the
   inference runtime can bind before execution. *)
let param_names (sign : GraphSignature.t) =
  List.fold_left
    (fun params (spec : InputSpec.t) ->
      match spec with
      | InputSpec.Parameter p ->
          String_map.add p.InputToParameterSpec.arg.TensorArgument.name
            p.InputToParameterSpec.parameter_name params
      | InputSpec.Buffer b ->
          String_map.add b.InputToBufferSpec.arg.TensorArgument.name
            b.InputToBufferSpec.buffer_name params
      | InputSpec.Tensor_constant c ->
          String_map.add c.InputToTensorConstantSpec.arg.TensorArgument.name
            c.InputToTensorConstantSpec.tensor_constant_name params
      | _ -> params)
    String_map.empty sign.GraphSignature.input_specs

(* Load exactly the captured tensors referenced by some node of [g] into
   [env], so unused buffers (e.g. the int64 num_batches_tracked) never reach
   the float32 bridge. Names [env] already binds (e.g. user inputs) are left
   untouched. *)
let load_referenced_params archive (g : Graph.t) (sign : GraphSignature.t) env =
  let params = param_names sign in
  Err.List.fold_left
    (fun env name ->
      if String_map.mem name env then Err.return env
      else
        match String_map.find_opt name params with
        | Some cfg_name ->
            let* weight =
              Pt2_archive.load_captured_tensor archive cfg_name
              |> Err.map_error (fun e -> (e :> error))
            in
            Err.return (String_map.add name (Pt2_aten.to_tensor weight) env)
        | None -> Err.return env)
    env
    (referenced_tensor_names g)

(* This zero-output debug assertion is checked against the live C++ tensor's
   metadata. It is never silently skipped and performs no tensor arithmetic. *)
let assert_tensor_metadata env node =
  let* tensor = tensor_arg env node "a" in
  let check field yes =
    if yes then Err.return () else Err.fail (`Metadata_mismatch field)
  in
  let optional_dims name actual =
    match find_arg node name with
    | None | Some (Argument.None _) -> Err.return ()
    | Some _ ->
        let* expected = ints_arg node name in
        check name (expected = Array.to_list actual)
  in
  let* () = optional_dims "size" (Aten_tensor.shape tensor) in
  let* () = optional_dims "stride" (Aten_tensor.strides tensor) in
  let* () =
    match find_arg node "dtype" with
    | None | Some (Argument.None _) -> Err.return ()
    | Some (Argument.Scalar_type dtype) -> (
        let expected =
          match dtype with
          | ScalarType.BOOL -> Some Aten_scalar_type.Bool
          | BYTE -> Some Byte
          | CHAR -> Some Char
          | DOUBLE -> Some Double
          | FLOAT -> Some Float
          | HALF -> Some Half
          | INT -> Some Int
          | LONG -> Some Long
          | SHORT -> Some Short
          | _ -> None
        in
        match expected with
        | Some dtype -> check "dtype" (dtype = Aten_tensor.scalar_type tensor)
        | None -> Err.fail (`Unsupported_scalar_type_arg "dtype"))
    | Some arg -> wrong_kind "dtype" `Scalar_type_opt arg
  in
  let* device = device_opt_arg node "device" in
  let* () = check "device" (device = None || Aten_tensor.is_cpu tensor) in
  let* () =
    match find_arg node "layout" with
    | None | Some (Argument.None _) | Some (Argument.Layout Layout.Strided) ->
        Err.return ()
    | Some (Argument.Layout _) -> Err.fail (`Unsupported_layout_arg "layout")
    | Some arg -> wrong_kind "layout" `Layout_opt arg
  in
  let+ () = check "outputs" (node.Node.outputs = []) in
  env

let run_named archive ~inputs =
  let g, sign = graph_and_signature archive in
  let* () =
    if
      List.for_all
        (function OutputSpec.User_output _ -> true | _ -> false)
        sign.GraphSignature.output_specs
    then Err.return ()
    else Err.fail (`Unexpected_graph_output "non-user output specification")
  in
  let* named =
    Err.List.map
      (function
        | InputSpec.User_input { UserInputSpec.arg = Argument.Tensor ta; _ } ->
            Err.return (Some ta.TensorArgument.name)
        | InputSpec.User_input { UserInputSpec.arg; _ } ->
            Err.fail (`Unexpected_graph_input (argument_kind_name arg))
        | _ -> Err.return None)
      sign.GraphSignature.input_specs
  in
  let names = List.filter_map Fun.id named in
  let actual = List.map fst inputs in
  let* () =
    if List.sort String.compare names = List.sort String.compare actual then
      Err.return ()
    else Err.fail (`Input_names (names, actual))
  in
  let* () =
    Err.List.iter
      (fun (name, (tensor : Pt2_tensor.t)) ->
        match String_map.find_opt name g.Graph.tensor_values with
        | None -> Err.fail (`Input_metadata (name, "missing"))
        | Some meta ->
            let check field yes =
              if yes then Err.return ()
              else Err.fail (`Input_metadata (name, field))
            in
            let* () =
              check "dtype"
                (Err.payload (Pt2_dtype.of_scalar_type meta.TensorMeta.dtype)
                = Ok tensor.dtype)
            in
            check "shape"
              (meta.sizes = List.map (fun n -> SymInt.Int n) tensor.sizes))
      inputs
  in
  let env =
    List.fold_left
      (fun env (name, tensor) ->
        String_map.add name (Pt2_aten.to_tensor tensor) env)
      String_map.empty inputs
  in
  let* env = load_referenced_params archive g sign env in
  let* env =
    Err.List.fold_left
      (fun env node ->
        (if node.Node.target = "torch.ops.aten._assert_tensor_metadata.default"
         then assert_tensor_metadata env node
         else Interp_dispatch.dispatch env node)
        |> Err.map_error (fun e -> (e :> error)))
      env g.Graph.nodes
  in
  Err.List.map
    (function
      | Argument.Tensor ta -> resolve env ta.TensorArgument.name
      | output ->
          Err.fail (`Unexpected_graph_output (graph_output_summary [ output ])))
    g.Graph.outputs

let run archive (image : Pt2_tensor.t) =
  let _, sign = graph_and_signature archive in
  let inputs =
    List.filter_map
      (function
        | InputSpec.User_input { UserInputSpec.arg = Argument.Tensor ta; _ } ->
            Some (ta.TensorArgument.name, image)
        | _ -> None)
      sign.GraphSignature.input_specs
  in
  let* outputs = run_named archive ~inputs in
  match outputs with
  | [ tensor ] -> Err.return tensor
  | outputs ->
      Err.fail
        (`Unexpected_graph_output
           (Printf.sprintf "%d outputs" (List.length outputs)))

(* Index of the maximum element of the flattened logits, computed by ATen
   (at::argmax with dim=None) and read back as an int. *)
let none_int = from_voidp int64_t null

(* [item_int] reads through the shim and never manages, so the result has to be
   managed here or the handle lives until the process exits. *)
let argmax logits =
  Err.return
    (Aten_tensor.item_int
       (O.argmax logits none_int false
       |> Aten_tensor.check |> Aten_tensor.manage))

(* The [k] highest-scoring (class index, probability) pairs of [logits]
   ([1; classes]), descending: softmax over the last dim, then at::topk. *)
let top_predictions logits k =
  (* Own every handle before reading it: [Aten_tensor.data]'s finaliser only
     anchors an already-registered handle for the Bigarray's lifetime, it never
     frees, so an unmanaged op result here leaks. Materialize after managing —
     topk's outputs are dense today, so the fast path returns them unchanged and
     "manage exactly once" still holds; were that to change, the original stays
     managed and the helper independently manages its clone. *)
  let probs =
    O._softmax logits 1L false |> Aten_tensor.check |> Aten_tensor.manage
  in
  let out = make TG.tensors2_struct in
  let st = O.topk probs (Int64.of_int k) (-1L) true true (addr out) in
  if st <> 0 then Err.fail (`Aten_runtime_failure ("topk", st))
  else
    let own v =
      tget out v |> Aten_tensor.check |> Aten_tensor.manage
      |> Aten_tensor.materialize_for_raw_read
    in
    match
      ( Aten_tensor.data Aten_dtype.float32 (own TG.tensors2_v0),
        Aten_tensor.data Aten_dtype.int64 (own TG.tensors2_v1) )
    with
    | Some vs, Some idx ->
        Err.return
          (List.init (Bigarray.Array1.dim idx) (fun i ->
               (Int64.to_int idx.{i}, vs.{i})))
    | _ -> Err.fail `Topk_read_failed
