(* See domain.mli. *)

open Graph_ir
open Core.Syntax

(* --- the four-axis shape rule ------------------------------------------------

   Which tensors it covers is the load-bearing part, and the two obvious
   phrasings are both wrong. "Every tensor" rejects a graph over a captured
   weight nobody reads. "Every tensor reachable from the outputs" accepts an
   unused non-4D graph INPUT — and DCE will not remove one: [Rewrite.apply]
   keeps an unused user input deliberately, "because the graph's signature is
   externally meaningful" (rewrite.ml). The lowerer would then have no good
   move, since copying it yields a graph the dialect rejects and omitting it
   silently narrows the interface.

   So: every effective [Input.Input], read or not, plus every tensor reachable
   from [Graph.outputs]. An unread CONSTANT falls outside both and is accepted —
   the lowerer omits it, which is the same seam [Rewrite.apply] already cuts
   along ("a constant nobody reads is gone"). See .ai/native4d_design.md §4.2. *)

let reachable view =
  let g = Graph_view.graph view in
  let rec go seen = function
    | [] -> seen
    | id :: rest ->
        if Tensor_id.Set.mem id seen then go seen rest
        else
          let seen = Tensor_id.Set.add id seen in
          go seen
            (match Graph_view.def view id with
            | None -> rest (* a graph input: nothing upstream *)
            | Some n -> Graph_ir.operands n.Node.op @ rest)
  in
  go Tensor_id.Set.empty g.Graph.outputs

let live_tensors view =
  let g = Graph_view.graph view in
  List.fold_left
    (fun acc id ->
      if Graph_ir.input_kind g id = Input.Input then Tensor_id.Set.add id acc
      else acc)
    (reachable view) g.Graph.inputs

(* An absent signature is not this module's business: [Graph_view] owns that
   check, and a tensor it never recorded cannot be shape-tested here. *)
let check_shapes view =
  Core.List.iter
    (fun id ->
      match Graph_view.sig_of view id with
      | None -> Core.return ()
      | Some sg ->
          let shape = sg.Tensor_sig.shape in
          let unit_axis axis = Dim.to_int (Vec6.get shape axis) = 1 in
          if unit_axis Axis.T && unit_axis Axis.D then Core.return ()
          else Core.fail (`Non_four_dimensional_tensor (id, shape)))
    (Tensor_id.Set.elements (live_tensors view))

(* --- per-node predicates ----------------------------------------------------

   Four ops name axes, and the predicate differs for each. [Mean] and [Rms_norm]
   carry an axis SUBSET, so naming T or D is possible and is what gets rejected.
   [Permute] carries a full six-axis bijection — [Permute.output_shape] requires
   every axis of [Axis.all] exactly once on each side — so it ALWAYS names T and
   D, and "names no axis outside N/H/W/C" would reject every permutation in
   existence; the implementable condition is that it FIXES T and D. [Batch_norm]
   names one axis, which must be C, because the §7.6 lowering to a depthwise 1x1
   convolution scales per channel and cannot normalize over N, H or W. *)

let check_dims node dims =
  Core.List.iter
    (fun axis ->
      match axis with
      | Axis.T | Axis.D -> Core.fail (`Axis_outside_dialect (node, axis))
      | Axis.N | Axis.H | Axis.W | Axis.C -> Core.return ())
    dims

let check_perm node perm =
  Core.List.iter
    (fun axis ->
      if Axis.equal (Permute.Permute.lookup perm axis) axis then Core.return ()
      else Core.fail (`Axis_outside_dialect (node, axis)))
    [ Axis.T; Axis.D ]

(* [weight] is laid out [Cout, 1, 1, Kh, Kw, Cin/groups] (conv.ml), so the C
   extent IS the per-group input channel count. Depthwise is exactly "one input
   channel per group"; combined with groups = in_channels that is the only other
   grouping the dialect represents. A weight with no recorded signature cannot
   be classified, so it is left to [Graph_view]. *)
let check_groups view node ~weight ~groups =
  let groups = (groups : Op_config.Pos.t :> int) in
  if groups = 1 then Core.return ()
  else
    match Graph_view.sig_of view weight with
    | None -> Core.return ()
    | Some sg ->
        if Dim.to_int (Vec6.get sg.Tensor_sig.shape Axis.C) = 1 then
          Core.return ()
        else Core.fail (`Unsupported_grouped_conv (node, groups))

let check_transposed node ~groups =
  let groups = (groups : Op_config.Pos.t :> int) in
  if groups = 1 then Core.return ()
  else Core.fail (`Unsupported_grouped_transposed_conv (node, groups))

let check_bmm view node ~input =
  match Graph_view.sig_of view input with
  | None -> Core.return ()
  | Some sg ->
      let batch = Vec6.get sg.Tensor_sig.shape Axis.H in
      if Dim.to_int batch = 1 then Core.return ()
      else Core.fail (`Unsupported_bmm_batch (node, batch))

(* The two running statistics are required operands; [weight] and [bias] are
   optional, and absent means the identity, which is not a dynamic parameter.
   So the rule is: the statistics, plus every optional that is PRESENT, must be
   an effective [Input.Constant]. A parameter that is still a node output — an
   unfolded relayout permute, say — is dynamic as far as this check goes, which
   is why constant folding runs before conversion. *)
let check_batch_norm view node (bn : Norm.BatchNorm.t) =
  let* () =
    let channel = bn.Norm.BatchNorm.params.Norm.BatchNorm.channel in
    if Axis.equal channel Axis.C then Core.return ()
    else Core.fail (`Axis_outside_dialect (node, channel))
  in
  Core.List.iter
    (fun id ->
      if Graph_view.is_constant view id then Core.return ()
      else Core.fail (`Dynamic_batch_norm node))
    (bn.Norm.BatchNorm.running_mean :: bn.Norm.BatchNorm.running_var
    :: List.filter_map Fun.id
         [ bn.Norm.BatchNorm.weight; bn.Norm.BatchNorm.bias ])

(* "Live" is not "used": a [Discard] sink exists precisely to record that an
   edge is dead while keeping the op's full ATen arity, so an index output
   consumed only by [Discard] is dead. That case is still rejected here — the
   dialect has no [Max_pool2d_with_indices] at all — but as an unsupported op,
   because stage 1's [Drop_pool_indices] removes it, whereas a genuinely live
   index needs an argmax-pool operation the dialect does not have and will not
   grow in this milestone. *)
let index_is_live view id =
  Graph_view.is_graph_output view id
  || List.exists
       (fun (n : node) -> match n.Node.op with Discard _ -> false | _ -> true)
       (Graph_view.uses view id)

(* Exhaustive, with NO default arm, following [Output_transfer.classify]: a
   defaulting match would silently admit the next Native op into a dialect that
   cannot represent it. Adding a Native op means deciding its answer here. *)
let check_node view (n : node) =
  let node = n.Node.id in
  let unsupported () = Core.fail (`Unsupported_op (node, n.Node.op)) in
  match n.Node.op with
  (* Direct counterparts, or legalizations that constrain nothing here: their
     tensors are covered by the shape rule above. *)
  | Add _ | Add_scalar _ | Avg_pool2d _ | Clamp _ | Clone _ | Div _
  | Div_scalar _ | Hardtanh _ | Linear _ | Max_pool2d _ | Mul _ | Relu _
  | Reshape _ | Sqrt _ | Sub _ ->
      Core.return ()
  | Batch_norm bn -> check_batch_norm view node bn
  | Bmm { Matmul.Bmm.input; _ } -> check_bmm view node ~input
  | Conv2d { Conv.Conv2d.params; weight; _ } ->
      check_groups view node ~weight ~groups:params.Conv.Conv2d.groups
  | Conv2d_padding { Conv.Conv2d_padding.params; weight; _ } ->
      check_groups view node ~weight ~groups:params.Conv.Conv2d_padding.groups
  | Convolution { Conv.Convolution.params; weight; _ } ->
      if params.Conv.Convolution.transposed then
        check_transposed node ~groups:params.Conv.Convolution.groups
      else check_groups view node ~weight ~groups:params.Conv.Convolution.groups
  | Mean { Reduce.Mean.params; _ } -> check_dims node params.Reduce.Mean.dims
  | Permute { Permute.Permute.perm; _ } -> check_perm node perm
  | Rms_norm { Norm.RmsNorm.params; _ } ->
      check_dims node params.Norm.RmsNorm.dims
  (* Two ops the dialect does not have. Both are removed by the canonical
     pipeline rather than legalized, so reaching them means the pipeline did not
     run — except for a live index, which nothing can remove. *)
  | Max_pool2d_with_indices _ -> (
      match n.Node.outputs with
      | [ _; indices ] when index_is_live view indices ->
          Core.fail (`Live_max_pool_indices (node, indices))
      | _ -> unsupported ())
  | Discard _ -> unsupported ()

(* Node predicates FIRST, then the shape rule. The two overlap — a permutation
   that moves C onto D necessarily produces a tensor with extent on D, so either
   check would reject it — and the op-level diagnostic is the actionable one:
   "axis D is outside the dialect" names the node to fix, where "tensor t1 has
   extent on D" names its consequence. Running shapes first would report the
   consequence for every axis violation an op causes.

   Order within the node pass is [Graph.nodes] order, which [Graph_view] has
   already checked is topological, so the first rejection is the earliest node
   that cannot be legalized. *)
let check view =
  let* () =
    Core.List.iter (check_node view) (Graph_ir.nodes (Graph_view.graph view))
  in
  check_shapes view
