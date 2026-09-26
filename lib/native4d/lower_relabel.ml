(* See lower_relabel.mli. *)

open Graph_ir

type t = {
  members : Node_id.t list;
  internal : Tensor_id.t list;
  fresh : Tensor_id.t list;
  extra : Tensor_id.t list;
  extra_nodes : Node_id.t list;
  nodes : (Node_id.t * node list) list;
  sigs : Tensor_sig.t list;
}

exception Abort

let extent shape axis = Vec6.get shape axis
let is_one e = Dim.equal e Dim.one

(* A tensor is relabelled when it carries an extent on T or D: N, T and D are
   adjacent in the frame's order, so fusing them into N keeps its flat layout. *)
let candidate shape =
  (not (is_one (extent shape Axis.T))) || not (is_one (extent shape Axis.D))

(* The one shape that reading D as N (rather than fusing) is exact for: its
   only non-unit axis among N/T/D is D. Ops that name D are defined for it. *)
let d_only shape =
  (not (is_one (extent shape Axis.D)))
  && is_one (extent shape Axis.T)
  && is_one (extent shape Axis.N)

let dialect shape = is_one (extent shape Axis.T) && is_one (extent shape Axis.D)
let simple shape = d_only shape || dialect shape

(* N*T*D fused into N. A shape whose fused extent does not fit is refused
   ([Abort]), like any op whose meaning does not survive the fusion. *)
let fuse_ntd shape =
  match
    Extent_product.bounded
      [ extent shape Axis.N; extent shape Axis.T; extent shape Axis.D ]
  with
  | Some n -> n
  | None -> raise Abort

let relabel_shape shape =
  Vec6.set
    (Vec6.set (Vec6.set shape Axis.N (fuse_ntd shape)) Axis.T Dim.one)
    Axis.D Dim.one

let relabel_axis = function Axis.D -> Axis.N | a -> a

(* The transposition (N D): a permutation must stay a bijection, so it swaps
   where [relabel_axis], which names a reduced or normalized axis, only moves. *)
let swap_axis = function Axis.D -> Axis.N | Axis.N -> Axis.D | a -> a

(* A reduction over [dims] with D read as N. The reduced tensor must itself be
   in the frame, and [dims] must not name the unit N or T. With [keepdim] false
   the survivors re-pack inward, which moves the extent on D somewhere the
   relabelled tensor's does not follow unless D is one of the axes reduced
   (only the unit N and T lie outside it), so that case is refused. *)
let reduce_dims ~x_shape (p : Reduce.Dims_keepdim.t) =
  if
    (not (candidate x_shape))
    || List.exists (function Axis.N | Axis.T -> true | _ -> false) p.dims
    || (List.mem Axis.D p.dims && not (d_only x_shape))
    || not (p.keepdim || List.mem Axis.D p.dims)
  then raise Abort
  else Reduce.Dims_keepdim.map_dims relabel_axis p

(* The op with N, T and D fused into N, or [Abort] when it is not one whose
   meaning survives the fusion. [shape_of] reads the SOURCE graph, [rename] maps
   a source tensor to the id its relabelled twin has, and [out_shape ()] is the
   op's source output shape. Names of D (a reduction, a softmax, a stack) are
   defined only for a tensor that is [d_only], where fusing is reading D as N.
   An operand or result outside the group keeps its shape, so a permutation is
   conjugated only on the sides that were relabelled. *)
let relabel_op ~shape_of ~rename ~out_shape (op : op) =
  let x_shape () = shape_of (List.hd (Graph_ir.operands op)) in
  match Graph_ir.map_operands rename op with
  | ( Add _ | Add_scalar _ | Avg_pool2d _ | Batched_matmul _ | Clone _ | Div _
    | Div_scalar _ | Gelu _ | Max_pool2d _ | Mul _ | Mul_scalar _ | Relu _
    | Sigmoid _ | Silu _ | Sub _ ) as op ->
      op
  | Amax { Reduce.Amax.params; x } ->
      Amax { Reduce.Amax.params = reduce_dims ~x_shape:(x_shape ()) params; x }
  | (Conv2d _ | Linear _) as op -> op
  | Expand { Pointwise.Expand.params; x } ->
      Expand
        {
          Pointwise.Expand.params =
            { size = relabel_shape params.Pointwise.Expand.size };
          x;
        }
  | Mean { Reduce.Mean.params; x } ->
      Mean { Reduce.Mean.params = reduce_dims ~x_shape:(x_shape ()) params; x }
  | Permute { Permute.Permute.perm; x } ->
      let x_shape = x_shape () and out_shape = out_shape () in
      if not (simple x_shape && simple out_shape) then raise Abort;
      let swap shape a = if d_only shape then swap_axis a else a in
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
              Reshape.Reshape.shape = relabel_shape params.Reshape.Reshape.shape;
            };
          x;
        }
  | Sdpa _ as op -> op
  | Softmax { Reduce.Softmax.params; x } ->
      let axis = params.Reduce.Softmax.axis in
      (match axis with
      | Axis.N | Axis.T -> raise Abort
      | Axis.D -> if not (d_only (x_shape ())) then raise Abort
      | Axis.H | Axis.W | Axis.C -> ());
      Softmax { Reduce.Softmax.params = { axis = relabel_axis axis }; x }
  | Split_with_sizes { Split.Split_with_sizes.params; x } ->
      let axis =
        match params.Split.Split_with_sizes.axis with
        | Axis.N | Axis.T -> raise Abort
        | Axis.D ->
            if not (d_only (x_shape ())) then raise Abort;
            Axis.N
        | (Axis.H | Axis.W | Axis.C) as a -> a
      in
      Split_with_sizes
        { Split.Split_with_sizes.params = { params with axis }; x }
  | Stack { Concat.Stack.params = { axis = Axis.D }; xs } ->
      if not (d_only (out_shape ())) then raise Abort;
      Concat { Concat.Concat.params = { axis = Axis.N }; xs }
  | Sum { Reduce.Sum.params; x } ->
      Sum { Reduce.Sum.params = reduce_dims ~x_shape:(x_shape ()) params; x }
  | Unbind { Split.Unbind.params = { axis }; x } ->
      (* The outputs are the fused axis cut into equal runs, which is
         contiguous only when nothing outside [axis] is larger than one. *)
      let s = x_shape () in
      let e = extent s in
      let count, run =
        match axis with
        | Axis.N ->
            ( e Axis.N,
              match Extent_product.bounded [ e Axis.T; e Axis.D ] with
              | Some run -> run
              | None -> raise Abort )
        | Axis.T when is_one (e Axis.N) -> (e Axis.T, e Axis.D)
        | Axis.D when is_one (e Axis.N) && is_one (e Axis.T) ->
            (e Axis.D, Dim.one)
        | _ -> raise Abort
      in
      if not (candidate s) then raise Abort;
      Split_with_sizes
        {
          Split.Split_with_sizes.params =
            {
              axis = Axis.N;
              sizes = List.init (count :> int) (fun _ -> (run :> int));
            };
          x;
        }
  | _ -> raise Abort

(* A permutation that moves N, T or D beyond swapping D with N is a data
   movement of the FLAT layout, planned like any other by [Wide_permute] from
   the source's own six-axis shape to the fused shapes on either side. Each
   step is an ordinary Native reshape or permute, so the ordinary lowering still
   runs on the result. *)
let plan_permute ~perm ~x_shape ~out_shape =
  let fuse shape =
    match Shape4.of_vec6 (relabel_shape shape) with
    | Ok s -> s
    | Error _ -> raise Abort
  in
  match
    Wide_permute.plan ~source:x_shape ~x:(fuse x_shape) ~y:(fuse out_shape)
      [ Wide_permute.Permute perm ]
  with
  | Some steps -> steps
  | None -> raise Abort

let native_of_step ~x = function
  | Wide_permute.Reshape4 shape ->
      Reshape { Reshape.Reshape.params = { shape = Shape4.to_vec6 shape }; x }
  | Wide_permute.Permute4 (perm, _) ->
      let of4 a =
        match Axis4.of_axis a with
        | Some a4 -> Axis4.to_axis (List.assoc a4 perm)
        | None -> a
      in
      Permute { Permute.Permute.perm = Permute.Permute.of_fn of4; x }

let step_shape = function
  | Wide_permute.Reshape4 shape | Wide_permute.Permute4 (_, shape) ->
      Shape4.to_vec6 shape

(* [x] reduced over H, W or C with the survivors re-packed inward moves the
   fused axis, so it is reduced keeping its dims and reshaped to the packed
   result. *)
let keepdim_of op ~x_shape =
  let on_frame (p : Reduce.Dims_keepdim.t) =
    (not p.keepdim) && candidate x_shape
    && List.for_all
         (function Axis.C | Axis.H | Axis.W -> true | _ -> false)
         p.dims
  in
  let keep (p : Reduce.Dims_keepdim.t) = { p with keepdim = true } in
  match op with
  | Amax { Reduce.Amax.params; x } when on_frame params ->
      Some (Amax { Reduce.Amax.params = keep params; x }, params.dims)
  | Mean { Reduce.Mean.params; x } when on_frame params ->
      Some (Mean { Reduce.Mean.params = keep params; x }, params.dims)
  | Sum { Reduce.Sum.params; x } when on_frame params ->
      Some (Sum { Reduce.Sum.params = keep params; x }, params.dims)
  | _ -> None

(* The nodes standing in for [n], its own id on the last, and the signatures of
   the tensors between them. [steps] are the ops in order, each built from its
   operand and paired with the shape it produces (unread for the last). *)
let chain ~rename ~sig_of ~fresh_tensor ~fresh_node (n : Graph_ir.node) ~x steps
    =
  let out_id = List.hd n.Node.outputs in
  let out_sig =
    match sig_of out_id with Some sg -> sg | None -> raise Abort
  in
  let count = List.length steps in
  let nodes, sigs, _ =
    List.fold_left
      (fun (nodes, sigs, (i, cur)) (make, shape) ->
        let last = i = count - 1 in
        let out = if last then rename out_id else fresh_tensor () in
        let id = if last then n.Node.id else fresh_node () in
        let sigs =
          if last then sigs
          else { out_sig with Tensor_sig.id = out; shape } :: sigs
        in
        ( { Node.id; op = make cur; outputs = [ out ] } :: nodes,
          sigs,
          (i + 1, out) ))
      ([], [], (0, x))
      steps
  in
  (List.rev nodes, sigs)

let relabel_node ~shape_of ~sig_of ~rename ~fresh_tensor ~fresh_node
    (n : Graph_ir.node) =
  let out_id = List.hd n.Node.outputs in
  let out_shape () = shape_of out_id in
  let chain = chain ~rename ~sig_of ~fresh_tensor ~fresh_node n in
  let keepdim =
    match n.Node.op with
    | (Amax _ | Mean _ | Sum _) as op ->
        keepdim_of op ~x_shape:(shape_of (List.hd (Graph_ir.operands op)))
    | _ -> None
  in
  match (n.Node.op, keepdim) with
  | _, Some (op, dims) ->
      let x = List.hd (Graph_ir.operands op) in
      let kept =
        List.fold_left
          (fun s a -> Vec6.set s a (Dim.extent 1))
          (relabel_shape (shape_of x))
          dims
      in
      chain ~x:(rename x)
        [
          ((fun x -> Graph_ir.map_operands (fun _ -> x) op), kept);
          ( (fun x ->
              Reshape
                {
                  Reshape.Reshape.params =
                    { shape = relabel_shape (out_shape ()) };
                  x;
                }),
            out_shape () );
        ]
  | ( Unbind
        {
          Split.Unbind.params = { axis = (Axis.C | Axis.H | Axis.W) as axis };
          x;
        },
      _ )
    when candidate (shape_of x) ->
      (* Dropping an inner axis shifts N, T and D inward, so each output is a
         unit slice reshaped to the packed result. *)
      let out_sig =
        match sig_of out_id with Some sg -> sg | None -> raise Abort
      in
      let x_fused = relabel_shape (shape_of x) in
      let sliced = Vec6.set x_fused axis (Dim.extent 1) in
      let slices = List.map (fun _ -> fresh_tensor ()) n.Node.outputs in
      let split =
        {
          Node.id = fresh_node ();
          op =
            Split_with_sizes
              {
                Split.Split_with_sizes.params =
                  { axis; sizes = List.map (fun _ -> 1) slices };
                x = rename x;
              };
          outputs = slices;
        }
      in
      let count = List.length slices in
      let reshapes =
        List.mapi
          (fun i (slice, out) ->
            {
              Node.id = (if i = count - 1 then n.Node.id else fresh_node ());
              op =
                Reshape
                  {
                    Reshape.Reshape.params =
                      { shape = relabel_shape (out_shape ()) };
                    x = slice;
                  };
              outputs = [ rename out ];
            })
          (List.combine slices n.Node.outputs)
      in
      ( split :: reshapes,
        List.map
          (fun id -> { out_sig with Tensor_sig.id; shape = sliced })
          slices )
  | Permute { Permute.Permute.perm; x }, _
    when not (simple (shape_of x) && simple (out_shape ())) ->
      let steps =
        plan_permute ~perm ~x_shape:(shape_of x) ~out_shape:(out_shape ())
      in
      chain ~x:(rename x)
        (match steps with
        | [] -> [ ((fun x -> Clone { Pointwise.Clone.x }), out_shape ()) ]
        | steps ->
            List.map
              (fun step -> ((fun x -> native_of_step ~x step), step_shape step))
              steps)
  | op, _ ->
      ( [
          {
            n with
            Node.op = relabel_op ~shape_of ~rename ~out_shape op;
            outputs = List.map rename n.Node.outputs;
          };
        ],
        [] )

let component view ~avoid seed =
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
      if Graph_view.is_graph_output view id || Tensor_id.Set.mem id avoid then
        raise Abort;
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

let find view ~watermark ~avoid =
  let g = Graph_view.graph view in
  let next = ref watermark in
  let next_node = ref (Id_supply.next_node (Id_supply.of_graph g)) in
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
                let visited, tensors = component view ~avoid id in
                claimed := Tensor_id.Set.union !claimed tensors;
                let members =
                  List.filter
                    (fun (m : Graph_ir.node) ->
                      Node_id.Set.mem m.Node.id visited)
                    g.Graph.nodes
                in
                let fresh_tensor () =
                  let f, n = Tensor_id.Next.alloc !next in
                  next := n;
                  f
                in
                let fresh_node () =
                  let f, n = Node_id.Next.alloc !next_node in
                  next_node := n;
                  f
                in
                let fresh_of =
                  List.map
                    (fun id -> (id, fresh_tensor ()))
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
                let replaced =
                  List.map
                    (fun (m : Graph_ir.node) ->
                      let nodes, sigs =
                        relabel_node ~shape_of ~sig_of:(Graph_view.sig_of view)
                          ~rename ~fresh_tensor ~fresh_node m
                      in
                      (m.Node.id, nodes, sigs))
                    members
                in
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
                let extra_sigs =
                  List.concat_map (fun (_, _, s) -> s) replaced
                in
                Some
                  {
                    members =
                      List.map (fun (m : Graph_ir.node) -> m.Node.id) members;
                    internal = List.map fst fresh_of;
                    fresh = List.map snd fresh_of;
                    extra =
                      List.map
                        (fun (sg : Tensor_sig.t) -> sg.Tensor_sig.id)
                        extra_sigs;
                    extra_nodes =
                      List.concat_map
                        (fun (id, nodes, _) ->
                          List.filter_map
                            (fun (m : Graph_ir.node) ->
                              if Node_id.equal m.Node.id id then None
                              else Some m.Node.id)
                            nodes)
                        replaced;
                    nodes =
                      List.map (fun (id, nodes, _) -> (id, nodes)) replaced;
                    sigs = sigs @ extra_sigs;
                  }
              with Abort -> None)
          | _ -> None)
        n.Node.outputs)
    g.Graph.nodes

let apply (g : Graph_ir.graph) groups =
  let replaced = List.concat_map (fun r -> r.nodes) groups in
  let internal = List.concat_map (fun r -> r.internal) groups in
  let rec regroup (grp : Graph_common.Group.t) =
    {
      grp with
      Graph_common.Group.items =
        List.concat_map
          (function
            | Graph_common.Group.Group sub ->
                [ Graph_common.Group.Group (regroup sub) ]
            | Graph_common.Group.Node id -> (
                match List.assoc_opt id replaced with
                | Some nodes ->
                    List.map
                      (fun (m : Graph_ir.node) ->
                        Graph_common.Group.Node m.Node.id)
                      nodes
                | None -> [ Graph_common.Group.Node id ]))
          grp.Graph_common.Group.items;
    }
  in
  {
    g with
    Graph.nodes =
      List.concat_map
        (fun (n : Graph_ir.node) ->
          Option.value (List.assoc_opt n.Node.id replaced) ~default:[ n ])
        g.Graph.nodes;
    root = regroup g.Graph.root;
    tensors =
      List.fold_left
        (fun m (sg : Tensor_sig.t) -> Tensor_id.Map.add sg.Tensor_sig.id sg m)
        (List.fold_left
           (fun m id -> Tensor_id.Map.remove id m)
           g.Graph.tensors internal)
        (List.concat_map (fun r -> r.sigs) groups);
  }
