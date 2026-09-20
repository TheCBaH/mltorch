(* See lower_relabel.mli. *)

open Graph_ir

type t = {
  members : Node_id.t list;
  internal : Tensor_id.t list;
  fresh : Tensor_id.t list;
  nodes : node list;
  sigs : Tensor_sig.t list;
}

let ext shape axis = Dim.to_int (Vec6.get shape axis)

(* A tensor is relabelled when its only non-unit axis among N/T/D is D. *)
let candidate shape =
  ext shape Axis.D > 1 && ext shape Axis.T = 1 && ext shape Axis.N = 1

let dialect shape = ext shape Axis.T = 1 && ext shape Axis.D = 1

let relabel_shape shape =
  Vec6.set
    (Vec6.set shape Axis.N (Dim.extent (ext shape Axis.D)))
    Axis.D (Dim.extent 1)

let relabel_axis = function Axis.D -> Axis.N | a -> a

(* The transposition (N D): a permutation must stay a bijection, so it swaps
   where [relabel_axis], which names a reduced or normalized axis, only moves. *)
let swap_axis = function Axis.D -> Axis.N | Axis.N -> Axis.D | a -> a

exception Abort

(* A reduction over [dims] with D read as N. The reduced tensor must itself be
   in the frame, and [dims] must not name the unit N or T. With [keepdim] false
   the survivors re-pack inward, which moves the extent on D somewhere the
   relabelled tensor's does not follow unless D is one of the axes reduced
   (only the unit N and T lie outside it), so that case is refused. *)
let reduce_dims ~x_shape (p : Reduce.Dims_keepdim.t) =
  if
    (not (candidate x_shape))
    || List.exists (function Axis.N | Axis.T -> true | _ -> false) p.dims
    || not (p.keepdim || List.mem Axis.D p.dims)
  then raise Abort
  else Reduce.Dims_keepdim.map_dims relabel_axis p

(* The op with D read as N, or [Abort] when it is not one whose meaning survives
   the move. [shape_of] reads the SOURCE graph, [rename] maps a source tensor to
   the id its relabelled twin has, and [out_shape ()] is the op's source output
   shape. An operand or result outside the group keeps its shape, so a
   permutation is conjugated only on the sides that were relabelled. *)
let relabel_op ~shape_of ~rename ~out_shape (op : op) =
  let shape_or_same s = if candidate s then relabel_shape s else s in
  let x_shape () = shape_of (List.hd (Graph_ir.operands op)) in
  match Graph_ir.map_operands rename op with
  | ( Add _ | Add_scalar _ | Batched_matmul _ | Clone _ | Div _ | Div_scalar _
    | Gelu _ | Mul _ | Mul_scalar _ | Relu _ | Sigmoid _ | Silu _ | Sub _ ) as
    op ->
      op
  | Amax { Reduce.Amax.params; x } ->
      Amax { Reduce.Amax.params = reduce_dims ~x_shape:(x_shape ()) params; x }
  | Expand { Pointwise.Expand.params; x } ->
      Expand
        {
          Pointwise.Expand.params =
            { size = shape_or_same params.Pointwise.Expand.size };
          x;
        }
  | Mean { Reduce.Mean.params; x } ->
      Mean { Reduce.Mean.params = reduce_dims ~x_shape:(x_shape ()) params; x }
  | Permute { Permute.Permute.perm; x } ->
      let swap shape a = if candidate shape then swap_axis a else a in
      let x_shape = x_shape () and out_shape = out_shape () in
      Permute
        {
          Permute.Permute.perm =
            Permute.Permute.of_fn (fun a ->
                swap x_shape (Permute.Permute.lookup perm (swap out_shape a)));
          x;
        }
  | Reshape { Reshape.Reshape.params; x } ->
      Reshape
        {
          Reshape.Reshape.params =
            {
              Reshape.Reshape.shape = shape_or_same params.Reshape.Reshape.shape;
            };
          x;
        }
  | Softmax { Reduce.Softmax.params; x } ->
      Softmax
        {
          Reduce.Softmax.params =
            { axis = relabel_axis params.Reduce.Softmax.axis };
          x;
        }
  | Stack { Concat.Stack.params = { axis = Axis.D }; xs } ->
      Concat { Concat.Concat.params = { axis = Axis.N }; xs }
  | Sum { Reduce.Sum.params; x } ->
      Sum { Reduce.Sum.params = reduce_dims ~x_shape:(x_shape ()) params; x }
  | _ -> raise Abort

let component view seed =
  let visited = ref Node_id.Set.empty in
  let tensors = ref Tensor_id.Set.empty in
  let shape_of id =
    match Graph_view.sig_of view id with
    | Some sg -> sg.Tensor_sig.shape
    | None -> raise Abort
  in
  let rec tensor id =
    if not (Tensor_id.Set.mem id !tensors) then begin
      tensors := Tensor_id.Set.add id !tensors;
      (* A relabelled tensor is one value; its producer and every reader move
         with it, and a graph input or output has no such freedom. *)
      if Graph_view.is_graph_output view id then raise Abort;
      (match Graph_view.def view id with
      | Some n -> node n
      | None -> raise Abort);
      List.iter node (Graph_view.uses view id)
    end
  and node (n : Graph_ir.node) =
    if not (Node_id.Set.mem n.Node.id !visited) then begin
      visited := Node_id.Set.add n.Node.id !visited;
      List.iter
        (fun id -> if candidate (shape_of id) then tensor id)
        (Graph_ir.operands n.Node.op @ n.Node.outputs)
    end
  in
  tensor seed;
  (!visited, !tensors)

let find view ~watermark =
  let g = Graph_view.graph view in
  let next = ref watermark in
  let claimed = ref Tensor_id.Set.empty in
  List.filter_map
    (fun (n : Graph_ir.node) ->
      List.find_map
        (fun id ->
          match Graph_view.sig_of view id with
          | Some sg
            when candidate sg.Tensor_sig.shape
                 && not (Tensor_id.Set.mem id !claimed) -> (
              try
                let visited, tensors = component view id in
                claimed := Tensor_id.Set.union !claimed tensors;
                let members =
                  List.filter
                    (fun (m : Graph_ir.node) ->
                      Node_id.Set.mem m.Node.id visited)
                    g.Graph.nodes
                in
                let fresh_of =
                  List.map
                    (fun id ->
                      let f = Tensor_id.of_int !next in
                      incr next;
                      (id, f))
                    (Tensor_id.Set.elements tensors)
                in
                let rename id =
                  Option.value (List.assoc_opt id fresh_of) ~default:id
                in
                let shape_of id =
                  match Graph_view.sig_of view id with
                  | Some sg -> sg.Tensor_sig.shape
                  | None -> raise Abort
                in
                let nodes =
                  List.map
                    (fun (m : Graph_ir.node) ->
                      {
                        m with
                        Node.op =
                          relabel_op ~shape_of ~rename
                            ~out_shape:(fun () ->
                              shape_of (List.hd m.Node.outputs))
                            m.Node.op;
                        outputs = List.map rename m.Node.outputs;
                      })
                    members
                in
                (* Every tensor a member touches must now be in the frame. *)
                List.iter
                  (fun (m : Graph_ir.node) ->
                    List.iter
                      (fun id ->
                        let s = shape_of id in
                        if not (candidate s || dialect s) then raise Abort)
                      (Graph_ir.operands m.Node.op @ m.Node.outputs))
                  members;
                let sigs =
                  List.map
                    (fun (old_id, f) ->
                      match Graph_view.sig_of view old_id with
                      | Some sg ->
                          {
                            sg with
                            Tensor_sig.id = f;
                            shape = relabel_shape sg.Tensor_sig.shape;
                          }
                      | None -> raise Abort)
                    fresh_of
                in
                Some
                  {
                    members =
                      List.map (fun (m : Graph_ir.node) -> m.Node.id) members;
                    internal = List.map fst fresh_of;
                    fresh = List.map snd fresh_of;
                    nodes;
                    sigs;
                  }
              with Abort -> None)
          | _ -> None)
        n.Node.outputs)
    g.Graph.nodes

let apply (g : Graph_ir.graph) groups =
  let replaced =
    List.concat_map
      (fun r -> List.map (fun (n : Graph_ir.node) -> (n.Node.id, n)) r.nodes)
      groups
  in
  let internal = List.concat_map (fun r -> r.internal) groups in
  {
    g with
    Graph.nodes =
      List.map
        (fun (n : Graph_ir.node) ->
          Option.value (List.assoc_opt n.Node.id replaced) ~default:n)
        g.Graph.nodes;
    tensors =
      List.fold_left
        (fun m (sg : Tensor_sig.t) -> Tensor_id.Map.add sg.Tensor_sig.id sg m)
        (List.fold_left
           (fun m id -> Tensor_id.Map.remove id m)
           g.Graph.tensors internal)
        (List.concat_map (fun r -> r.sigs) groups);
  }
