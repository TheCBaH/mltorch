(* See domain.mli. *)

open Graph_ir
open Err.Syntax

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
          (* A graph input is produced by no node, so it pushes nothing. *)
          let upstream =
            Graph_view.def view id
            |> Option.fold ~none:[] ~some:(fun n -> Graph_ir.operands n.Node.op)
          in
          go seen (upstream @ rest)
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
  Err.List.iter
    (fun id ->
      match Graph_view.sig_of view id with
      | None -> Err.return ()
      | Some sg ->
          let shape = sg.Tensor_sig.shape in
          let unit_axis axis = Dim.to_int (Vec6.get shape axis) = 1 in
          if unit_axis Axis.T && unit_axis Axis.D then Err.return ()
          else Err.fail (`Non_four_dimensional_tensor (id, shape)))
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
  Err.List.iter
    (fun axis ->
      match axis with
      | Axis.T | Axis.D -> Err.fail (`Axis_outside_dialect (node, axis))
      | Axis.N | Axis.H | Axis.W | Axis.C -> Err.return ())
    dims

let check_perm node perm =
  Err.List.iter
    (fun axis ->
      if Axis.equal (Permute.Permute.lookup perm axis) axis then Err.return ()
      else Err.fail (`Axis_outside_dialect (node, axis)))
    [ Axis.T; Axis.D ]

(* Forward grouped convolution always legalizes now: [Conv2d] (groups=1),
   [Depthwise_conv2d] (one input channel per group), or [Grouped_conv2d] (the
   general form, `.ai/native4d_design.md` §8) between them cover every count.
   Only the TRANSPOSED direction still lacks a general counterpart. *)
let check_transposed node ~groups =
  let groups = (groups : Op_config.Pos.t :> int) in
  if groups = 1 then Err.return ()
  else Err.fail (`Unsupported_grouped_transposed_conv (node, groups))

(* [Batched_matmul]'s batch axes are N/T/D/H, all four of which
   [output_shape] requires to agree OR broadcast (one side extent 1) between
   [input] and [mat2] -- not "D and H" as an earlier reading of this arm had
   it. Of those, D is the one this dialect cannot name, so the only real
   restriction is on the OUTPUT's D extent; N and H are already dialect axes
   and carry the corpus's actual batch (heads on H, `mvitv2_tiny`). Checking
   [input] alone is no longer enough now that [output_shape] broadcasts: an
   [input] with D=1 broadcast against a [mat2] with D>1 has an output D>1
   that [input]'s own extent does not show, so both operands' D must be read
   and the broadcast result (whichever is not 1) is what the dialect actually
   has to represent. [Bmm] needs no such check at all: it legalizes to
   [Batched_matmul] unchanged, and its own [H] batch axis is exactly this
   dialect's [H], unrestricted at any extent. *)
let check_batched_matmul view node ~input ~mat2 =
  match (Graph_view.sig_of view input, Graph_view.sig_of view mat2) with
  | None, _ | _, None -> Err.return ()
  | Some input_sg, Some mat2_sg ->
      let input_d = Vec6.get input_sg.Tensor_sig.shape Axis.D in
      let mat2_d = Vec6.get mat2_sg.Tensor_sig.shape Axis.D in
      let out_d = if Dim.to_int input_d = 1 then mat2_d else input_d in
      if Dim.to_int out_d = 1 then Err.return ()
      else Err.fail (`Batched_matmul_batch_axis node)

(* [Sdpa]'s batch axis is D alone (heads are on H), so unlike
   [Batched_matmul] this really is a genuine restriction: D = 1 is the whole
   admissible case, not merely the reachable slice of a wider one. *)
let check_sdpa view node ~query =
  match Graph_view.sig_of view query with
  | None -> Err.return ()
  | Some sg ->
      let batch = Vec6.get sg.Tensor_sig.shape Axis.D in
      if Dim.to_int batch = 1 then Err.return ()
      else Err.fail (`Sdpa_batch_axis node)

(* The two running statistics are required operands; [weight] and [bias] are
   optional, and absent means the identity, which is not a dynamic parameter.
   So the rule is: the statistics, plus every optional that is PRESENT, must be
   an effective [Input.Constant]. A parameter that is still a node output — an
   unfolded relayout permute, say — is dynamic as far as this check goes, which
   is why constant folding runs before conversion. *)
let check_batch_norm view node (bn : Norm.BatchNorm.t) =
  let* () =
    let channel = bn.Norm.BatchNorm.params.Norm.BatchNorm.channel in
    if Axis.equal channel Axis.C then Err.return ()
    else Err.fail (`Axis_outside_dialect (node, channel))
  in
  (* Every parameter must be as long as the axis it scales. The lowerer reads
     one value per channel to precompute the depthwise weight, so a shorter
     vector reads out of bounds — and an out-of-bounds read raises, where this
     module's whole contract is to answer with a typed error.

     Nothing upstream rejects it because nothing VALIDATES it:
     [Norm.BatchNorm.output_shape] is a function of the input shape alone, so
     the parameters' extents are never compared against the normalized axis.
     Native's evaluation does not survive such a graph either — [Compute] reads
     each parameter at the OUTPUT's channel index and [Tensor.read] is strict —
     so this check does not make Native4D stricter than Native, it makes the
     failure arrive as a typed error instead of an exception. *)
  let* () =
    match Graph_view.sig_of view bn.Norm.BatchNorm.x with
    | None -> Err.return ()
    | Some x_sig ->
        let channels = Vec6.get x_sig.Tensor_sig.shape Axis.C in
        Err.List.iter
          (fun id ->
            match Graph_view.sig_of view id with
            | None -> Err.return ()
            | Some sg ->
                let extent = Vec6.get sg.Tensor_sig.shape Axis.C in
                if Dim.equal extent channels then Err.return ()
                else Err.fail (`Batch_norm_extent (node, id, extent, channels)))
          (bn.Norm.BatchNorm.running_mean :: bn.Norm.BatchNorm.running_var
          :: List.filter_map Fun.id
               [ bn.Norm.BatchNorm.weight; bn.Norm.BatchNorm.bias ])
  in
  Err.List.iter
    (fun id ->
      if Graph_view.is_constant view id then Err.return ()
      else Err.fail (`Dynamic_batch_norm node))
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
  let unsupported () = Err.fail (`Unsupported_op (node, n.Node.op)) in
  match n.Node.op with
  (* Direct counterparts, or legalizations that constrain nothing here: their
     tensors are covered by the shape rule above. *)
  | Add _ | Add_scalar _ | Adaptive_avg_pool2d _ | Adaptive_max_pool2d _
  | Avg_pool2d _ | Bmm _ | Clamp _ | Clone _ | Conv1d _ | Conv2d _
  | Conv2d_padding _ | Div _ | Div_scalar _ | Expand _ | Gelu _ | Hardsigmoid _
  | Hardswish _ | Hardtanh _ | Leaky_relu _ | Linear _ | Lstm _ | Max_pool2d _
  | Mul _ | Mul_scalar _ | Pow _ | Relu _ | Repeat _ | Reshape _ | Rsub_scalar _
  | Sigmoid _ | Silu _ | Sqrt _ | Sub _ | To_copy _ | Upsample_bilinear2d _
  | Upsample_nearest2d _ ->
      Err.return ()
  | Arange _ | Eye _ | Zeros _ -> Err.return ()
  | Batch_norm bn -> check_batch_norm view node bn
  | Batch_norm_no_stats { Norm.BatchNormNoStats.params; _ } ->
      if Axis.equal params.channel Axis.C then Err.return ()
      else Err.fail (`Axis_outside_dialect (node, params.channel))
  (* [Group_norm4] now exists, so [Group_norm] gets the same channel-must-be-C
     restriction [Batch_norm_no_stats] gets, for the same reason: every
     importer sets it there today (`Norm.GroupNorm`'s own doc comment), and
     the dialect has no need to accept a configuration the corpus never
     produces. [groups] is not checked here -- any divisor of the channel
     extent is representable, so it is [check_shapes]'s business, not this
     arm's. *)
  | Group_norm { Norm.GroupNorm.params; _ } ->
      if Axis.equal params.channel Axis.C then Err.return ()
      else Err.fail (`Axis_outside_dialect (node, params.channel))
  | Layer_norm { Norm.LayerNorm.params; _ } ->
      check_dims node params.Norm.LayerNorm.dims
  (* Its batch axes are N/T/D/H, all four of which [output_shape] requires to
     agree or broadcast between [input] and [mat2] -- D is the axis this
     dialect cannot name; N and H are dialect axes and already carry the
     corpus's real batch (heads on H, `mvitv2_tiny`), so an output D of 1 is
     the only admissible case. [Bmm] needs no such check: it legalizes to
     [Batched_matmul] unchanged (the "Direct counterparts" arm above), and
     its own batch axis is [H], which this dialect already names at any
     extent. *)
  | Batched_matmul { Matmul.Batched_matmul.input; mat2 } ->
      check_batched_matmul view node ~input ~mat2
  | Convolution { Conv.Convolution.params; _ } ->
      if params.Conv.Convolution.transposed then
        check_transposed node ~groups:params.Conv.Convolution.groups
      else Err.return ()
  | Amax { Reduce.Amax.params; _ } -> check_dims node params.Reduce.Amax.dims
  | Mean { Reduce.Mean.params; _ } -> check_dims node params.Reduce.Mean.dims
  | Sum { Reduce.Sum.params; _ } -> check_dims node params.Reduce.Sum.dims
  | Vector_norm { Reduce.Vector_norm.params; _ } ->
      check_dims node params.Reduce.Vector_norm.dims
  (* Only the PADDED axes are gated. An entry naming T or D is what the dialect
     cannot say; an unpadded T or D extent is not this arm's business, and
     [check_shapes] plus the lowerer's [Shape4.of_vec6] already cover it. *)
  | Pad { Pad.Pad.params; _ } ->
      check_dims node (List.map fst params.Pad.Pad.pads)
  | Permute { Permute.Permute.perm; _ } -> check_perm node perm
  | Slice { Split.Slice.params; _ } -> check_dims node [ params.axis ]
  | Rms_norm { Norm.RmsNorm.params; _ } ->
      check_dims node params.Norm.RmsNorm.dims
  (* Not [check_dims]: the batch axis is D, unconditionally -- heads are on H,
     so unlike [Batched_matmul]'s N/T/D/H spread this is a genuine
     restriction, checked the same conditional way as every other batch-axis
     arm here. A Region-authored graph at D = 1 needs no per-head
     decomposition: it delegates to the same [Region_program] Native uses. *)
  | Sdpa { Attention.Sdpa.query; _ } -> check_sdpa view node ~query
  (* Three ops the dialect does not have (the adaptive pair joining
     [Max_pool2d_with_indices] for the same reason -- see [Adaptive_max_pool2d]
     just above, which IS a direct counterpart precisely because
     [Drop_pool_indices] narrows the dead-index case to it before this ever
     runs). Both are removed by the canonical pipeline rather than legalized,
     so reaching them means the pipeline did not run — except for a live
     index, which nothing can remove. *)
  | Max_pool2d_with_indices _ | Adaptive_max_pool2d_with_indices _ -> (
      match n.Node.outputs with
      | [ _; indices ] when index_is_live view indices ->
          Err.fail (`Live_max_pool_indices (node, indices))
      | _ -> unsupported ())
  | Discard _ -> unsupported ()
  (* [Concat4] now exists, so [Concat] gets the same [check_dims]-style axis
     rejection [Select]/[Slice]/[Stack]/[Unbind] get: the JOINED axis is the
     one the dialect must be able to name, and the rest of the domain -- every
     operand already being four-axis -- is [check_shapes]'s business, not this
     arm's. *)
  | Concat { Concat.Concat.params; _ } -> check_dims node [ params.axis ]
  (* [Split_with_sizes4] now exists, so [Split_with_sizes] gets the same
     [check_dims]-style axis rejection [Concat]/[Select]/[Slice]/[Unbind] get:
     the SPLIT axis is the one the dialect must be able to name, and the rest
     of the domain is [check_shapes]'s business, not this arm's. *)
  | Split_with_sizes { Split.Split_with_sizes.params; _ } ->
      check_dims node [ params.axis ]
  (* [Select4] now exists, so [Select] gets the same [check_dims]-style axis
     rejection [Concat]/[Slice]/[Split_with_sizes]/[Unbind] get: the SELECTED
     axis is the one the dialect must be able to name. Whether the axis it
     DROPS re-enters the dialect is a fact about [Select]'s inferred output,
     not about this arm -- the same "node predicates first" split the comment
     below [check view] states; [check_shapes] and the lowerer's own
     [Shape4.of_vec6] cover it. *)
  | Select { Split.Select.params; _ } -> check_dims node [ params.axis ]
  (* [Stack4] now exists, so [Stack] gets the same [check_dims]-style axis
     rejection [Concat]/[Select]/[Slice]/[Split_with_sizes]/[Unbind] get: the
     INSERTED axis is the one the dialect must be able to name. Whether the
     axis it pushes outward re-enters the dialect is a fact about [Stack]'s
     inferred output, not about this arm -- the same split [Select]'s own
     comment states; [check_shapes] and the lowerer's own [Shape4.of_vec6]
     cover it. *)
  | Stack { Concat.Stack.params; _ } -> check_dims node [ params.axis ]
  (* [RepeatInterleave4] now exists, so [RepeatInterleave] gets the same
     [check_dims]-style axis rejection [Select]/[Slice]/[Stack] get -- see
     [Ops4.RepeatInterleave4]'s own comment on why this is a consistency
     choice rather than a load-bearing one for this particular op (it
     MULTIPLIES its named axis, unlike those, which COLLAPSE it). *)
  | RepeatInterleave { Repeat.RepeatInterleave.params; _ } ->
      check_dims node [ params.axis ]
  (* [Select_scatter4] now exists, so [Select_scatter] gets the same
     [check_dims]-style axis rejection [Select] gets: the WRITTEN axis is
     the one the dialect must be able to name. Unlike [Select], the output
     shape is [self_shape] unchanged (no drop, no repack), so there is no
     post-hoc [Shape4.of_vec6] re-check to defer to here -- if [self] is
     already four-axis, so is the output, regardless of which axis this op
     names. *)
  | Select_scatter { Split.Select_scatter.params; _ } ->
      check_dims node [ params.axis ]
  (* [Softmax4] now exists, so [Softmax] gets the same [check_dims]-style
     axis rejection [Select]/[Select_scatter]/[Slice]/[Stack]/
     [RepeatInterleave] get: the REDUCED axis is the one the dialect must be
     able to name. Softmax never changes shape, so unlike [Amax]/[Mean]/
     [Vector_norm] there is no separate keepdim-vs-drop distinction here --
     one counterpart, one axis check. See .ai/matmul_softmax_design.md §3. *)
  | Softmax { Reduce.Softmax.params; _ } -> check_dims node [ params.axis ]
  (* [Cumsum4] now exists, so [Cumsum] gets the same [check_dims]-style axis
     rejection [Softmax] gets just above: the WALKED axis is the one the
     dialect must be able to name. Cumsum never changes shape either, so the
     same "one counterpart, one axis check" reasoning applies. *)
  | Cumsum { Reduce.Cumsum.params; _ } -> check_dims node [ params.axis ]
  (* [IndexTensor4] now exists, so [Index_tensor] gets the same [check_dims]-
     style axis rejection [Select]/[Select_scatter]/[Stack]/[RepeatInterleave]
     get: the GATHERED axis is the one the dialect must be able to name. Not
     load-bearing the same way [Select_scatter]'s is not: the output is
     [self_shape] with that one axis's extent changed (no drop, no repack),
     so there is no separate shape-consequence rejection to demonstrate. *)
  | Index_tensor { Index_tensor.Index_tensor.params; _ } ->
      check_dims node [ params.Index_tensor.Index_tensor.axis ]
  (* The axis is checked HERE, on the Native [Axis.t], and converted to
     [Axis4.t] only in the lowerer. That ordering is what lets the diagnostic
     name the rejected axis: converting first would leave nothing to report but
     "conversion failed".
     The rest of the domain — every inferred slice still being four-axis — is
     not restated here. [check_shapes] covers the outputs that are live, and the
     lowerer's own [Shape4.of_vec6] covers the rest, so an unbind that shifts a
     non-unit N onto T is refused without this arm knowing the rule. *)
  | Unbind { Split.Unbind.params; _ } -> check_dims node [ params.axis ]
  (* Not a missing counterpart: [Unfold]'s own output_shape (unfold.ml)
     shifts EVERY carried-through axis one step toward N (`source_of`), so
     whatever real (non-unit) content sat on the input's H axis lands on the
     output's D axis, and C's own input content lands on W -- both real
     dialect-incompatible moves for any input whose H/C actually hold data,
     which is the ordinary case (an unfolded spatial or channel axis is
     rarely unit). Native4D has no way to represent that shift without
     naming T/D as real axes, so this is the same intrinsic boundary
     [Batched_matmul]'s multi-batch form and [Sdpa]'s own D axis are --
     `.ai/native4d_design.md` §8 territory, not a gap to close later. *)
  | Unfold _ -> unsupported ()
  (* Not a missing counterpart: unlike [Conv1d]/[Conv2d], whose ATen schemas
     genuinely have at most two spatial axes (fitting the H/W the dialect
     already names), [Conv3d]'s three spatial axes land on Native's D/H/W
     (conv_conv3d.ml) -- and the dialect's own N/H/W/C frame forces D and T
     to extent 1 always. A real (non-unit) D is the ordinary case for this
     op, so admitting it would require extending the dialect itself, the
     same intrinsic-axis boundary [Batched_matmul]'s multi-batch form,
     [Sdpa]'s own D axis, and [Unfold] above are. *)
  | Conv3d _ -> unsupported ()

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
    Err.List.iter (check_node view) (Graph_ir.nodes (Graph_view.graph view))
  in
  check_shapes view
