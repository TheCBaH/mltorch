(* One global SSA environment; structural groups do not affect evaluation. *)

open Graph_ir
module E = Eval_op.Make (Direct)

type context = Operand | Sig_shape
type missing_tensor = { context : context; id : Tensor_id.t }
type arity_mismatch = { expected : int; actual : int }

type error =
  [ Graph_shape.error
  | `Missing_tensor of missing_tensor
  | `Output_arity_mismatch of arity_mismatch
  | `Missing_input of Tensor_id.t
  | `Missing_constant of Tensor_id.t ]

type hooks =
  | Hooks : { on_start : node -> 'a; on_end : node -> 'a -> unit } -> hooks

let pp_context ppf = function
  | Operand -> Format.pp_print_string ppf "operand"
  | Sig_shape -> Format.pp_print_string ppf "shape lookup"

let pp_error ppf : [< error ] -> unit = function
  | #Graph_shape.error as e -> Graph_shape.pp_error ppf e
  | `Missing_tensor { context; id } ->
      Format.fprintf ppf "missing %a tensor t%d" pp_context context
        (Tensor_id.to_int id)
  | `Output_arity_mismatch { expected; actual } ->
      Format.fprintf ppf
        "node output arity mismatch: %d output shapes for %d output ids"
        expected actual
  | `Missing_input id ->
      Format.fprintf ppf "missing input tensor t%d" (Tensor_id.to_int id)
  | `Missing_constant id ->
      Format.fprintf ppf "missing constant tensor t%d" (Tensor_id.to_int id)

let find_tensor map id ~context =
  match Tensor_id.Map.find_opt id map with
  | Some x -> Core.return x
  | None -> Core.fail (`Missing_tensor { context; id })

let widen (r : ('a, [< error ]) Core.result) : ('a, error) Core.result =
  (r :> ('a, error) Core.result)

let sig_shape (g : graph) r =
  let open Core.Syntax in
  let+ sg = find_tensor g.Graph.tensors r ~context:Sig_shape in
  sg.Tensor_sig.shape

let input_ids g =
  List.filter (fun id -> Graph_ir.input_kind g id = Input.Input) g.Graph.inputs

(* An exported program can retain captured state that no lowered operation
   consumes (for example BatchNorm's int64 num_batches_tracked).  Constants are
   therefore required only when they occur as an actual graph operand. *)
let constant_is_used g id =
  List.exists
    (fun node -> List.mem id (Graph_ir.operands node.Node.op))
    g.Graph.nodes

let bind_constants g constants env =
  Core.List.fold_left
    (fun env id ->
      match Graph_ir.input_kind g id with
      | Input.Input -> Core.return env
      | Input.Constant when not (constant_is_used g id) -> Core.return env
      | Input.Constant -> (
          match List.assoc_opt id constants with
          | Some tensor -> Core.return (Tensor_id.Map.add id tensor env)
          | None -> Core.fail (`Missing_constant id)))
    env g.Graph.inputs

let rec run_graph ?hooks ~constants (g : graph)
    (env : Tensor.packed Tensor_id.Map.t) :
    (Tensor.packed Tensor_id.Map.t, error) Core.result =
  let open Core.Syntax in
  let* env = bind_constants g constants env in
  Core.List.fold_left
    (fun env node ->
      match hooks with
      | None -> eval_node g env node
      | Some (Hooks h) ->
          let state = h.on_start node in
          let* env = eval_node g env node in
          h.on_end node state;
          Core.return env)
    env g.Graph.nodes

and eval_node (g : graph) (env : Tensor.packed Tensor_id.Map.t) (node : node) :
    (Tensor.packed Tensor_id.Map.t, error) Core.result =
  let open Core.Syntax in
  let op = node.Node.op in
  let operand r = find_tensor env r ~context:Operand in
  let shape_of r = sig_shape g r in
  let fill v shape = Tensor.materialize shape (fun _ -> v) in
  let* shapes =
    widen
      (Graph_shape.output_shape op ~sig_of:(fun r ->
           match Tensor_id.Map.find_opt r g.Graph.tensors with
           | Some sg -> Core.return sg
           | None -> Core.fail (`Missing_tensor_sig r)))
  in
  let* operand_env =
    Core.List.fold_left
      (fun acc r ->
        let+ t = operand r in
        Tensor_id.Map.add r t acc)
      Tensor_id.Map.empty (Graph_ir.operands op)
  in
  let* shape_env =
    Core.List.fold_left
      (fun acc r ->
        let+ sh = shape_of r in
        Tensor_id.Map.add r sh acc)
      Tensor_id.Map.empty (Graph_ir.operands op)
  in
  (* One materialisation per output edge: [Graph_shape] and [Node.outputs]
         agree in length by construction (single-output ops give one of each; a
         [Discard]-style zero-output op gives none, so the fold is empty). *)
  if List.compare_lengths node.Node.outputs shapes <> 0 then
    Core.fail
      (`Output_arity_mismatch
         {
           expected = List.length shapes;
           actual = List.length node.Node.outputs;
         })
  else
    let outs =
      List.mapi
        (fun output (oid, out_shape) -> (output, oid, out_shape))
        (List.combine node.Node.outputs shapes)
    in
    Core.List.fold_left
      (fun env (output, oid, out_shape) ->
        let result =
          Schedule.evaluate out_shape
            (E.pixel op ~output
               ~operand:(fun r -> Tensor_id.Map.find r operand_env)
               ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
               ~fill)
        in
        Core.return (Tensor_id.Map.add oid result env))
      env outs

let run ?hooks ?(constants = []) (g : graph)
    ~(inputs : (Tensor_id.t * Tensor.packed) list) =
  let provided =
    List.fold_left
      (fun e (id, t) -> Tensor_id.Map.add id t e)
      Tensor_id.Map.empty inputs
  in
  let open Core.Syntax in
  let* env0 =
    Core.List.fold_left
      (fun env id ->
        match Tensor_id.Map.find_opt id provided with
        | Some tensor -> Core.return (Tensor_id.Map.add id tensor env)
        | None -> Core.fail (`Missing_input id))
      Tensor_id.Map.empty (input_ids g)
  in
  run_graph ?hooks ~constants g env0
