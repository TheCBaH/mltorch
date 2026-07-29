(* See cell_origin.mli. *)

open Graph_ir
module Origin = Ground_expr.Origin

type t = { dst : Tensor_id.t -> Origin.t; src : Tensor_id.t -> Origin.t }

(* Compared by name, not structurally: [packed_fmt] is an existential over a
   GADT. [quant] is a first-order variant, so [=] is exact there. *)
let fmt_name (Payload.Fmt f) = Payload.fmt_name f

let tensor_ids (g : graph) =
  Tensor_id.Map.fold
    (fun id _ acc -> Tensor_id.Set.add id acc)
    g.Graph.tensors Tensor_id.Set.empty

(* One variable per cluster, in cluster order, restricted on each side to the
   ids that really are graph inputs THERE. A cluster pairing an input with a node
   output contributes a variable on one side only, and the two sides then
   disagree, which is the correct answer. *)
let input_vars ~src ~dst clusters =
  let inputs (g : graph) = Tensor_id.Set.of_list g.Graph.inputs in
  let src_inputs = inputs (Snapshot.graph src)
  and dst_inputs = inputs (Snapshot.graph dst) in
  let _, s, d =
    List.fold_left
      (fun (i, s, d) (c : ('src, 'dst) Correspondence.Cluster.t) ->
        let v = Input_var.of_int i in
        let add keep set m =
          Correspondence.Set.fold
            (fun e m ->
              let id = Correspondence.raw e in
              if Tensor_id.Set.mem id keep then Tensor_id.Map.add id v m else m)
            set m
        in
        (i + 1, add src_inputs c.src s, add dst_inputs c.dst d))
      (0, Tensor_id.Map.empty, Tensor_id.Map.empty)
      clusters
  in
  (s, d)

let classify ~src ~dst clusters =
  let src_view = Snapshot.view src and dst_view = Snapshot.view dst in
  let src_var, dst_var = input_vars ~src ~dst clusters in
  let sig_key view id =
    Option.map
      (fun (sg : Tensor_sig.t) -> (sg.shape, fmt_name sg.fmt, sg.quant))
      (Graph_view.sig_of view id)
  in
  let output_index (n : node) id =
    let rec go i = function
      | [] -> -1
      | x :: rest -> if Tensor_id.equal x id then i else go (i + 1) rest
    in
    go 0 n.Node.outputs
  in
  (* Operands are erased to one anchor so this compares the CONSTRUCTOR and its
     non-tensor parameters only; operand identity is the separate check below.
     [Rewrite.same_definition] uses the same idiom against a representative. *)
  let anchor = Tensor_id.of_int 0 in
  let shape_of_op (op : op) = Graph_ir.map_operands (fun _ -> anchor) op in
  let memo = ref Tensor_id.Map.empty in
  let rec origins visiting id =
    match Tensor_id.Map.find_opt id !memo with
    | Some o -> o
    | None ->
        (* Re-entering an id still in progress means the two graphs disagree
           about its depth — one defines it upstream of an edge the other
           defines downstream of it. Refusing [Shared] there is the conservative
           direction and is what keeps this total. Deliberately not memoized:
           the answer holds only under this recursion. *)
        if Tensor_id.Set.mem id visiting then (Origin.Src id, Origin.Dst id)
        else
          let visiting = Tensor_id.Set.add id visiting in
          let o = decide visiting id in
          memo := Tensor_id.Map.add id o !memo;
          o
  and decide visiting id =
    match
      (Tensor_id.Map.find_opt id src_var, Tensor_id.Map.find_opt id dst_var)
    with
    | Some a, Some b -> (Origin.Input a, Origin.Input b)
    | Some a, None -> (Origin.Input a, Origin.Dst id)
    | None, Some b -> (Origin.Src id, Origin.Input b)
    | None, None -> (
        match (Graph_view.def src_view id, Graph_view.def dst_view id) with
        | Some ns, Some nd when same visiting ns nd id ->
            (Origin.Shared id, Origin.Shared id)
        | _ -> (Origin.Src id, Origin.Dst id))
  and same visiting (ns : node) (nd : node) id =
    let a = Graph_ir.operands ns.Node.op and b = Graph_ir.operands nd.Node.op in
    shape_of_op ns.Node.op = shape_of_op nd.Node.op
    && List.length a = List.length b
    && List.for_all2
         (fun x y ->
           Origin.equal (fst (origins visiting x)) (snd (origins visiting y)))
         a b
    (* [Max_pool2d_with_indices] produces semantically distinct slots —
       [Output_transfer] calls slot 0 continuous and the rest discontinuous — so
       one raw id used in different slots must not compare equal. *)
    && output_index ns id = output_index nd id
    (* A cheap conservative guard only. The real format check is
       [Graph_map.create]'s, where it is decidable and where the lens needs it. *)
    && sig_key src_view id = sig_key dst_view id
  in
  (* Eagerly, in ascending id order, so the memo fills the same way whoever asks
     first and a printed counterexample is reproducible. *)
  Tensor_id.Set.iter
    (fun id -> ignore (origins Tensor_id.Set.empty id))
    (Tensor_id.Set.union
       (tensor_ids (Snapshot.graph src))
       (tensor_ids (Snapshot.graph dst)));
  let lookup id =
    Option.value
      (Tensor_id.Map.find_opt id !memo)
      ~default:(Origin.Src id, Origin.Dst id)
  in
  { dst = (fun id -> snd (lookup id)); src = (fun id -> fst (lookup id)) }
