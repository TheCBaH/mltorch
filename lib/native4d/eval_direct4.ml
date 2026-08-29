(* Direct evaluation of a Native4D graph. The twin of [Eval_direct], and
   structurally identical to it: one global SSA environment, one materialisation
   per output edge, structural groups ignored.

   It is a twin rather than a shared driver because [Eval_direct] is typed on
   [Graph_ir.graph] and the two records differ. Stage 4 makes the record shared;
   the driver could follow, but the arithmetic that matters — [Eval_op4]'s
   delegation to Native's [Compute (S)] — is already shared, which is the reuse
   the design asks for. *)

module E = Eval_op4.Make (Direct)

type context = Operand | Sig_shape
type missing_tensor = { context : context; id : Tensor_id.t }
type arity_mismatch = { expected : int; actual : int }

type error =
  [ Graph_shape4.error
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Missing_tensor of missing_tensor
  | `Output_arity_mismatch of arity_mismatch ]

let pp_context ppf = function
  | Operand -> Fmt.string ppf "operand"
  | Sig_shape -> Fmt.string ppf "shape lookup"

let pp_error ppf : [< error ] -> unit = function
  | #Graph_shape4.error as e -> Graph_shape4.pp_error ppf e
  | `Missing_constant id ->
      Fmt.pf ppf "missing constant tensor %a" Tensor_id.pp id
  | `Missing_input id -> Fmt.pf ppf "missing input tensor %a" Tensor_id.pp id
  | `Missing_tensor { context; id } ->
      Fmt.pf ppf "missing %a tensor %a" pp_context context Tensor_id.pp id
  | `Output_arity_mismatch { expected; actual } ->
      Fmt.pf ppf
        "node output arity mismatch: %d output shapes for %d output ids"
        expected actual

let find_tensor map id ~context =
  Tensor_id.Map.find_opt id map
  |> Err.of_option (`Missing_tensor { context; id })

let widen (r : ('a, [< error ]) Err.t) : ('a, error) Err.t =
  (r :> ('a, error) Err.t)

let sig_shape (g : Graph.graph) r =
  let open Err.Syntax in
  let+ sg = find_tensor g.Graph.Graph.tensors r ~context:Sig_shape in
  sg.Tensor_sig.shape

let input_ids (g : Graph.graph) =
  List.filter
    (fun id -> Graph.input_kind g id = Graph_ir.Input.Input)
    g.Graph.Graph.inputs

(* As in Native: an exported program can retain captured state no lowered
   operation consumes, so a constant is required only where it is an operand. *)
let constant_is_used (g : Graph.graph) id =
  List.exists
    (fun (n : Graph.node) -> List.mem id (Op.operands n.Graph.Node.op))
    g.Graph.Graph.nodes

let bind_constants (g : Graph.graph) constants env =
  Err.List.fold_left
    (fun env id ->
      match Graph.input_kind g id with
      | Graph_ir.Input.Constant when not (constant_is_used g id) ->
          Err.return env
      | Graph_ir.Input.Constant -> (
          match List.assoc_opt id constants with
          | None -> Err.fail (`Missing_constant id)
          | Some tensor -> Err.return (Tensor_id.Map.add id tensor env))
      | Graph_ir.Input.Input -> Err.return env)
    env g.Graph.Graph.inputs

let eval_node (g : Graph.graph) env (node : Graph.node) =
  let open Err.Syntax in
  let op = node.Graph.Node.op in
  let fill v shape = Tensor.materialize shape (fun _ -> v) in
  let* shapes =
    widen
      (Graph_shape4.output_shape op ~sig_of:(fun r ->
           Tensor_id.Map.find_opt r g.Graph.Graph.tensors
           |> Err.of_option (`Missing_tensor_sig r)))
  in
  let* operand_env =
    Err.List.fold_left
      (fun acc r ->
        let+ t = find_tensor env r ~context:Operand in
        Tensor_id.Map.add r t acc)
      Tensor_id.Map.empty (Op.operands op)
  in
  let* shape_env =
    Err.List.fold_left
      (fun acc r ->
        let+ sh = sig_shape g r in
        Tensor_id.Map.add r sh acc)
      Tensor_id.Map.empty (Op.operands op)
  in
  let* pairs =
    Err.List.map2
      ~unequal_lengths:(fun actual expected ->
        `Output_arity_mismatch { expected; actual })
      (fun oid out_shape -> Err.return (oid, out_shape))
      node.Graph.Node.outputs shapes
  in
  Err.List.fold_left
    (fun env (output, oid, out_shape) ->
      let result =
        match op with
        | Op.Unbind { Ops4.Unbind.params; x } ->
            Tensor.unbind
              (Tensor_id.Map.find x operand_env)
              ~axis:(Axis4.to_axis params.axis)
              ~output ~shape:(Shape4.to_vec6 out_shape)
        | _ ->
            Schedule.evaluate (Shape4.to_vec6 out_shape)
              (E.pixel op ~output
                 ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                 ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                 ~fill)
      in
      Err.return (Tensor_id.Map.add oid result env))
    env
    (List.mapi (fun output (oid, out_shape) -> (output, oid, out_shape)) pairs)

let run ?(constants = []) (g : Graph.graph)
    ~(inputs : (Tensor_id.t * Tensor.packed) list) =
  let provided =
    List.fold_left
      (fun e (id, t) -> Tensor_id.Map.add id t e)
      Tensor_id.Map.empty inputs
  in
  let open Err.Syntax in
  let* env0 =
    Err.List.fold_left
      (fun env id ->
        match Tensor_id.Map.find_opt id provided with
        | None -> Err.fail (`Missing_input id)
        | Some tensor -> Err.return (Tensor_id.Map.add id tensor env))
      Tensor_id.Map.empty (input_ids g)
  in
  let* env = bind_constants g constants env0 in
  Err.List.fold_left (eval_node g) env g.Graph.Graph.nodes
