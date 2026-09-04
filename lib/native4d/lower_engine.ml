(* The Native4D legalization engine: the walk accumulator ([acc]), the
   Native->Native4D parameter-translation helpers, and [lower_node], the
   per-source-node dispatch. Split out of lower.ml under the tracked
   file-size ceiling; lower.ml keeps the public surface ([t], [convert],
   [evaluate], ...) and calls into [lower_node] here for the per-node work.
   See lower.mli for the module this feeds. *)

(* Bound BEFORE [open Graph_ir], which shadows [Graph] with the Native one. *)
module G4 = Graph
open Graph_ir
open Err.Syntax

(* ---- what the walk accumulates -------------------------------------------- *)

type acc = {
  nodes : G4.node list; (* reversed *)
  tensors : Tensor_sig.t Tensor_id.Map.t;
  subst : Tensor_id.t Tensor_id.Map.t; (* clone removal, source-side rewiring *)
  next_tid : int;
  next_nid : int;
  created : Tensor_id.t list; (* fresh destination edges *)
  deleted : Tensor_id.t list; (* source edges with no destination *)
  claims : (Tensor_id.t * Correspondence.relation) list;
      (* weaker than Identical *)
  node_pairs : (Node_id.t * Node_id.t list) list;
  provenance : (Tensor_id.t list * Tensor_id.t) list;
  constants : Tensor.packed Tensor_id.Map.t;
  fresh_constants : Tensor_id.t list; (* new captured state, in creation order *)
}

let resolve acc id =
  Option.value (Tensor_id.Map.find_opt id acc.subst) ~default:id

let fresh_tensor acc shape =
  let id = Tensor_id.of_int acc.next_tid in
  let sg =
    Tensor_sig.create ~id ~name:"" ~shape:(Shape4.to_vec6 shape)
      ~fmt:(Payload.Fmt Payload.F32) ()
  in
  ( id,
    {
      acc with
      next_tid = acc.next_tid + 1;
      tensors = Tensor_id.Map.add id sg acc.tensors;
      created = id :: acc.created;
    } )

(* A fresh CONSTANT is not just a signature: it is captured model state, so it
   has to join [Graph.inputs] with kind [Constant] as well. Omitting that leaves
   it defined by no node and declared no input, which validation rejects — the
   symptom being an operand with no definition. *)
let fresh_constant acc shape payload =
  let id, acc = fresh_tensor acc shape in
  ( id,
    {
      acc with
      fresh_constants = id :: acc.fresh_constants;
      constants = Tensor_id.Map.add id payload acc.constants;
    } )

(* A destination node taking over [outputs]; [from] is the source node it came
   from, which is what the node map records.

   NODE ids follow the same policy as tensor ids, and for the same reason. The
   first destination node of a source node KEEPS that node's id; a second one
   (Mean keepdim=false) takes a fresh id above the source watermark.
   Allocating densely from zero instead would make destination node 0 a
   different node from source node 0 whenever anything was removed — the raw-id
   collision the design forbids for edges, reappearing for nodes. *)
let emit acc ~from op outputs =
  let already =
    List.exists (fun (s, _) -> Node_id.equal s from) acc.node_pairs
  in
  let nid = if already then Node_id.of_int acc.next_nid else from in
  let acc =
    {
      acc with
      next_nid = (if already then acc.next_nid + 1 else acc.next_nid);
      nodes = { G4.Node.id = nid; op; outputs } :: acc.nodes;
    }
  in
  let node_pairs =
    List.map
      (fun (s, ds) -> if Node_id.equal s from then (s, nid :: ds) else (s, ds))
      acc.node_pairs
  in
  let node_pairs =
    if List.exists (fun (s, _) -> Node_id.equal s from) node_pairs then
      node_pairs
    else (from, [ nid ]) :: node_pairs
  in
  { acc with node_pairs }

(* ---- parameter translation ------------------------------------------------ *)

let conv_params (p : Conv.Conv2d.params) : Ops4.Conv_params.t =
  { h = p.h; w = p.w; in_channels = p.in_channels }

(* Grouping becomes a constructor. §7.2/§8: one group is [Conv2D], one input
   channel per group is [DepthwiseConv2D], and every other count is
   [GroupedConv2D] — the general form, which needs [groups] itself since
   neither of the other two constructors carries it. *)
let forward_conv ~node:_ ~params ~x ~weight ~bias ~weight_shape =
  let groups = (params.Conv.Conv2d.groups :> int) in
  let payload =
    { Ops4.Conv_payload.params = conv_params params; x; weight; bias }
  in
  if groups = 1 then Err.return (Op.Conv2d payload)
  else if Dim.to_int (Vec6.get weight_shape Axis.C) = 1 then
    Err.return (Op.Depthwise_conv2d payload)
  else
    let grouped_params =
      {
        Ops4.Grouped_conv_params.h = params.Conv.Conv2d.h;
        w = params.Conv.Conv2d.w;
        in_channels = params.Conv.Conv2d.in_channels;
        groups = params.Conv.Conv2d.groups;
      }
    in
    Err.return
      (Op.Grouped_conv2d
         { Ops4.Grouped_conv_payload.params = grouped_params; x; weight; bias })

let perm4_of_native ~node (perm : Permute.Permute.perm) =
  Err.List.map
    (fun out ->
      let in_axis = Permute.Permute.lookup perm (Axis4.to_axis out) in
      match Axis4.of_axis in_axis with
      | Some a -> Err.return (out, a)
      | None -> Err.fail (`Axis_outside_dialect (node, in_axis)))
    Axis4.all

let dims4 ~node dims =
  Err.List.map
    (fun axis ->
      Axis4.of_axis axis |> Err.of_option (`Axis_outside_dialect (node, axis)))
    dims

let shape4 ~id shape =
  match Shape4.of_vec6 shape with
  | Ok s -> Err.return s
  | Error _ -> Err.fail (`Non_four_dimensional_tensor (id, shape))

(* ---- batch norm ----------------------------------------------------------- *)

(* §7.6, and the one legalization that is EQUIVALENT rather than identical.

   A standalone inference batch norm with constant parameters is per-channel
   affine: out = (x - mean) * weight / sqrt(var + eps) + bias. Precomputing one
   scale and one offset per channel turns it into a 1x1 depthwise convolution —
   but it re-associates the arithmetic, so the f32 rounding points move, and the
   claim is [Equivalent]. Reporting it [Identical] would have the verifier
   assert bit-equality the computation does not deliver.

   Needs the payloads AT CONVERSION TIME, which is why absence is an error here
   and nowhere else. *)
let bn_channel_values ~node ~channels ~constants ids =
  Err.List.map
    (fun id ->
      match Tensor_id.Map.find_opt id constants with
      | None -> Err.fail (`Missing_constant_payload (node, id))
      | Some t ->
          Err.return (fun c ->
              (* Parameters are laid out on C; every other axis is unit, so the
                 read is at (0,…,0,c). *)
              Tensor.read_at t (fun axis ->
                  if Axis.equal axis Axis.C then Dim.index c else Dim.index 0)))
    ids
  |> fun r ->
  let+ fs = r in
  ignore channels;
  fs

let batch_norm_weights acc ~node ~channels ~eps (bn : Norm.BatchNorm.t) =
  let required = [ bn.running_mean; bn.running_var ] in
  let optional = List.filter_map Fun.id [ bn.weight; bn.bias ] in
  let* readers =
    bn_channel_values ~node ~channels ~constants:acc.constants
      (required @ optional)
  in
  let mean, var, rest =
    match readers with
    | m :: v :: rest -> (m, v, rest)
    | _ -> assert false (* two required operands, always present *)
  in
  let gamma, beta =
    match (bn.weight, bn.bias, rest) with
    | Some _, Some _, [ g; b ] -> (g, b)
    | Some _, None, [ g ] -> (g, fun _ -> 0.)
    | None, Some _, [ b ] -> ((fun _ -> 1.), b)
    | _ -> ((fun _ -> 1.), fun _ -> 0.)
  in
  let scale c = gamma c /. Float.sqrt (var c +. eps) in
  let offset c = beta c -. (mean c *. scale c) in
  (* Depthwise 1x1 weight is [Cout,1,1,1,1,1]; bias is [1,1,1,1,1,Cout]. *)
  let w_shape = Shape4.of_ints ~n:channels ~h:1 ~w:1 ~c:1 in
  let b_shape = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:channels in
  let w =
    Tensor.materialize (Shape4.to_vec6 w_shape) (fun coord ->
        scale (Dim.to_int (Vec6.get coord Axis.N)))
  in
  let b =
    Tensor.materialize (Shape4.to_vec6 b_shape) (fun coord ->
        offset (Dim.to_int (Vec6.get coord Axis.C)))
  in
  let wid, acc = fresh_constant acc w_shape w in
  let bid, acc = fresh_constant acc b_shape b in
  Err.return (wid, bid, acc)

let unit_conv_params ~in_channels : Ops4.Conv_params.t =
  {
    h = Conv.Conv2d.unit_window;
    w = Conv.Conv2d.unit_window;
    in_channels = Dim.extent in_channels;
  }

(* ---- one source node ------------------------------------------------------ *)

(* Every arm resolves its operands through [subst] first, which is how clone
   removal reaches consumers: the clone contributes no node and instead records
   "my output means my input", so every later reference rewires. *)
let lower_node ~view acc (n : node) =
  let node = n.Node.id in
  let op_of = resolve acc in
  let sig_of id =
    match Graph_view.sig_of view id with
    | Some sg -> Err.return sg.Tensor_sig.shape
    | None ->
        Err.fail
          (`Non_four_dimensional_tensor
             (id, Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1))
  in
  (* An INVARIANT, not a diagnostic. [Graph_view] has already checked every
     node's arity against shape inference, and [Domain.check] has already run,
     so a single-output arm meeting anything else means those two disagree with
     this match — our bug, not a model we cannot represent.

     It must not be `Unsupported_op`: [Me_classify.native4d] maps that to
     [Unavailable Outside_dialect_domain], which tells a user to change their
     model to work around a lowering defect. Same treatment as
     [Native_interp.add_env] gives its own module invariants.

     Lazy, unlike the [List.hd] it replaces: that was computed before the match
     and so would have raised on a zero-output [Discard] node before reaching
     the arm that rejects one. *)
  let single () =
    match n.Node.outputs with
    | [ o ] -> o
    | outputs ->
        invalid_arg
          (Format.asprintf
             "Native4d.Lower: %a is a single-output op but declares %d outputs"
             Node_id.pp node (List.length outputs))
  in
  let simple op = Err.return (emit acc ~from:node op [ single () ]) in
  (* Shared by [Mean]/[Amax]/[Sum]/[Vector_norm]'s arms below: all four are
     the SAME shape (`.ai/native4d_design.md` §1's reduction rule is not
     specific to any one of them, and all four reuse [Reduce.Dims_keepdim]'s
     own [output_shape] directly, not a per-op copy). [keepdim=true] is the
     dialect's own [X_keepdims] node unchanged; [keepdim=false] is corrected
     by C1 into that same node plus a [Reshape4], and the reshape target
     re-enters the dialect only when the packed Native output shape stays
     four-axis. Only [keepdims_op] (which [Ops4] constructor to build) varies
     per op. *)
  let lower_keepdim_reduction ~(orig_dims : Axis.t list) ~keepdim
      ~(keepdims_op : Axis4.t list -> Tensor_id.t -> Op.t) ~x =
    let* dims = dims4 ~node orig_dims in
    if keepdim then
      Err.return
        (emit acc ~from:node (keepdims_op dims (op_of x)) [ single () ])
    else
      let* x_shape = sig_of x in
      let* kept =
        match
          Reduce.Dims_keepdim.output_shape ~x_shape
            { Reduce.Dims_keepdim.dims = orig_dims; keepdim = true }
        with
        | Ok s -> Err.return s
        | Error _ -> Err.fail (`Unsupported_op (node, n.Node.op))
      in
      let* packed = sig_of (single ()) in
      let* kept4 = shape4 ~id:(single ()) kept in
      let* packed4 = shape4 ~id:(single ()) packed in
      let mid, acc = fresh_tensor acc kept4 in
      let acc = emit acc ~from:node (keepdims_op dims (op_of x)) [ mid ] in
      let acc =
        { acc with provenance = ([ op_of x ], mid) :: acc.provenance }
      in
      Err.return
        (emit acc ~from:node
           (Op.Reshape4 { Ops4.Reshape4.params = { shape = packed4 }; x = mid })
           [ single () ])
  in
  match n.Node.op with
  (* §7.1 direct counterparts. The payload records are Native's own, reused
     unchanged — they name no axis and carry no shape. *)
  | Add { Pointwise.Bin.a; b } ->
      simple (Op.Add { Pointwise.Bin.a = op_of a; b = op_of b })
  | Sub { Pointwise.Bin.a; b } ->
      simple (Op.Sub { Pointwise.Bin.a = op_of a; b = op_of b })
  | Mul { Pointwise.Bin.a; b } ->
      simple (Op.Mul { Pointwise.Bin.a = op_of a; b = op_of b })
  | Div { Pointwise.Bin.a; b } ->
      simple (Op.Div { Pointwise.Bin.a = op_of a; b = op_of b })
  | Add_scalar { Pointwise.Scalar_bin.x; scalar } ->
      simple (Op.Add_scalar { Pointwise.Scalar_bin.x = op_of x; scalar })
  | Div_scalar { Pointwise.Scalar_bin.x; scalar } ->
      simple (Op.Div_scalar { Pointwise.Scalar_bin.x = op_of x; scalar })
  | Mul_scalar { Pointwise.Scalar_bin.x; scalar } ->
      simple (Op.Mul_scalar { Pointwise.Scalar_bin.x = op_of x; scalar })
  | Pow { Pointwise.Scalar_bin.x; scalar } ->
      simple (Op.Pow { Pointwise.Scalar_bin.x = op_of x; scalar })
  | Rsub_scalar { Pointwise.Rsub_scalar.params; x } ->
      simple (Op.Rsub_scalar { Pointwise.Rsub_scalar.params; x = op_of x })
  | Clamp { Pointwise.Clamp.params; x } ->
      simple (Op.Clamp { Pointwise.Clamp.params; x = op_of x })
  | Gelu { Pointwise.Gelu.x; approximate } ->
      simple (Op.Gelu { Pointwise.Gelu.x = op_of x; approximate })
  | Hardsigmoid { Pointwise.Hardsigmoid.x } ->
      simple (Op.Hardsigmoid { Pointwise.Hardsigmoid.x = op_of x })
  | Hardswish { Pointwise.Hardswish.x } ->
      simple (Op.Hardswish { Pointwise.Hardswish.x = op_of x })
  | Hardtanh { Pointwise.Hardtanh.params; x } ->
      simple (Op.Hardtanh { Pointwise.Hardtanh.params; x = op_of x })
  | Leaky_relu { Pointwise.Leaky_relu.params; x } ->
      simple (Op.Leaky_relu { Pointwise.Leaky_relu.params; x = op_of x })
  | Arange { Factory.Arange.params } ->
      let* source = sig_of (single ()) in
      let+ _ = shape4 ~id:(single ()) source in
      emit acc ~from:node
        (Op.Arange4
           {
             Ops4.Arange4.params =
               {
                 start = params.start;
                 stop = params.stop;
                 step = params.step;
                 fmt = params.fmt;
               };
           })
        [ single () ]
  | Zeros { Factory.Zeros.params } ->
      let* shape = sig_of (single ()) in
      let+ shape = shape4 ~id:(single ()) shape in
      emit acc ~from:node
        (Op.Zeros4 { Ops4.Zeros4.params = { shape; fmt = params.fmt } })
        [ single () ]
  | Eye { Factory.Eye.params } ->
      let* shape = sig_of (single ()) in
      let+ shape = shape4 ~id:(single ()) shape in
      emit acc ~from:node
        (Op.Eye4 { Ops4.Eye4.params = { shape; fmt = params.fmt } })
        [ single () ]
  | Relu { Pointwise.Relu.x } -> simple (Op.Relu { Pointwise.Relu.x = op_of x })
  (* A direct counterpart, the same shape [Expand]'s own arm has: [repeats]
     converts through [shape4] exactly like [Expand]'s [size] does, and
     needs no post-hoc output re-check the way [Select]/[Stack] do, because
     [Repeat] neither drops nor inserts an axis -- every axis keeps its own
     identity (repeat.ml's own doc comment), so a T/D-unit [repeats] and a
     T/D-unit input [x] compose to a T/D-unit output automatically. *)
  | Repeat { Repeat.Repeat.params; x } ->
      let* repeats = shape4 ~id:(single ()) params.Repeat.Repeat.repeats in
      simple (Op.Repeat4 { Ops4.Repeat4.params = { repeats }; x = op_of x })
  (* A direct counterpart with only the axis KEY converted, the same reason
     [Select]'s own arm converts its axis -- so [Domain]'s
     [Axis_outside_dialect] can still name the rejected Native axis. No
     post-hoc output re-check needed, the same reasoning [Repeat4]'s arm
     above gives: [RepeatInterleave] multiplies its named axis's extent in
     place rather than dropping or inserting one, so nothing shifts into
     T/D. *)
  | RepeatInterleave { Repeat.RepeatInterleave.params; x } ->
      let* axis4 = dims4 ~node [ params.Repeat.RepeatInterleave.axis ] in
      simple
        (Op.RepeatInterleave4
           {
             Ops4.RepeatInterleave4.params =
               {
                 axis = List.hd axis4;
                 repeats = params.Repeat.RepeatInterleave.repeats;
               };
             x = op_of x;
           })
  | Sigmoid { Pointwise.Sigmoid.x } ->
      simple (Op.Sigmoid { Pointwise.Sigmoid.x = op_of x })
  | Silu { Pointwise.Silu.x } -> simple (Op.Silu { Pointwise.Silu.x = op_of x })
  | Sqrt { Pointwise.Sqrt.x } -> simple (Op.Sqrt { Pointwise.Sqrt.x = op_of x })
  | To_copy { Pointwise.To_copy.target; x } ->
      simple (Op.To_copy { Pointwise.To_copy.target; x = op_of x })
  | Avg_pool2d { Pool.AvgPool2d.params; x } ->
      simple (Op.Avg_pool2d { Pool.AvgPool2d.params; x = op_of x })
  | Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x } ->
      simple
        (Op.Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x = op_of x })
  | Adaptive_max_pool2d { Pool.AdaptiveMaxPool2d.params; x } ->
      simple
        (Op.Adaptive_max_pool2d { Pool.AdaptiveMaxPool2d.params; x = op_of x })
  | Max_pool2d { Pool.MaxPool2d.params; x } ->
      simple (Op.Max_pool2d { Pool.MaxPool2d.params; x = op_of x })
  (* Direct counterpart, like [Max_pool2d] above: [Resize.Bilinear2d.params]
     names no axis and carries no shape, so it crosses unchanged. *)
  | Upsample_bilinear2d { Resize.Bilinear2d.params; x } ->
      simple (Op.Upsample_bilinear2d { Resize.Bilinear2d.params; x = op_of x })
  (* Same reasoning as [Upsample_bilinear2d] just above: [Resize.Nearest2d.params]
     names no axis and carries no shape either. *)
  | Upsample_nearest2d { Resize.Nearest2d.params; x } ->
      simple (Op.Upsample_nearest2d { Resize.Nearest2d.params; x = op_of x })
  (* The axes were gated on the NATIVE [Axis.t] by [Domain.check_dims] before
     this walk started, so [dims4] here only converts what is already known to
     be inside the dialect -- which is what lets the diagnostic name the
     rejected axis instead of reporting "conversion failed". *)
  | Layer_norm { Norm.LayerNorm.params; x; weight; bias } ->
      let+ dims = dims4 ~node params.Norm.LayerNorm.dims in
      emit acc ~from:node
        (Op.Layer_norm
           {
             Ops4.Layer_norm.params = { dims; eps = params.Norm.LayerNorm.eps };
             x = op_of x;
             weight = Option.map op_of weight;
             bias = Option.map op_of bias;
           })
        [ single () ]
  | Rms_norm { Norm.RmsNorm.params; x; weight } ->
      let+ dims = dims4 ~node params.Norm.RmsNorm.dims in
      emit acc ~from:node
        (Op.Rms_norm
           {
             Ops4.Rms_norm.params = { dims; eps = params.Norm.RmsNorm.eps };
             x = op_of x;
             weight = Option.map op_of weight;
           })
        [ single () ]
  | Batch_norm_no_stats { Norm.BatchNormNoStats.params; x; weight; bias } ->
      let outputs =
        match n.Node.outputs with
        | [ _; _; _ ] as outputs -> outputs
        | outputs ->
            invalid_arg
              (Format.asprintf
                 "Native4d.Lower: %a is a three-output op but declares %d \
                  outputs"
                 Node_id.pp node (List.length outputs))
      in
      let* channel =
        Axis4.of_axis params.Norm.BatchNormNoStats.channel
        |> Err.of_option
             (`Axis_outside_dialect (node, params.Norm.BatchNormNoStats.channel))
      in
      Err.return
        (emit acc ~from:node
           (Op.Batch_norm_no_stats
              {
                Ops4.Batch_norm_no_stats.params =
                  { channel; eps = params.Norm.BatchNormNoStats.eps };
                x = op_of x;
                weight = Option.map op_of weight;
                bias = Option.map op_of bias;
              })
           outputs)
  (* Same channel-conversion shape as [Batch_norm_no_stats] just above --
     [Domain.check_node] has already gated it to C -- but single-output like
     [Layer_norm]/[Rms_norm], and [groups] crosses unchanged: it names no
     axis, so there is nothing for this arm to convert. *)
  | Group_norm { Norm.GroupNorm.params; x; weight; bias } ->
      let* channel =
        Axis4.of_axis params.Norm.GroupNorm.channel
        |> Err.of_option
             (`Axis_outside_dialect (node, params.Norm.GroupNorm.channel))
      in
      simple
        (Op.Group_norm4
           {
             Ops4.Group_norm4.params =
               {
                 channel;
                 groups = params.Norm.GroupNorm.groups;
                 eps = params.Norm.GroupNorm.eps;
               };
             x = op_of x;
             weight = Option.map op_of weight;
             bias = Option.map op_of bias;
           })
  (* §7.1: [Clone] is removed and its output tied to its input. No node, no
     fresh id — just a substitution, and a pair cluster recording that the two
     edges are the same value. *)
  | Clone { Pointwise.Clone.x } ->
      Err.return
        {
          acc with
          subst = Tensor_id.Map.add (single ()) (op_of x) acc.subst;
          (* Drop the signature too. Left in place the edge would exist in the
             destination universe, and naming it only as a cluster SOURCE would
             then be [Unpaired_src] — an id present in both graphs has to be
             named on both sides. *)
          tensors = Tensor_id.Map.remove (single ()) acc.tensors;
        }
  | Permute { Permute.Permute.perm; x } ->
      let+ perm = perm4_of_native ~node perm in
      emit acc ~from:node
        (Op.Permute4 { Ops4.Permute4.perm; x = op_of x })
        [ single () ]
  | Reshape { Reshape.Reshape.params; x } ->
      let* shape = shape4 ~id:(single ()) params.Reshape.Reshape.shape in
      simple (Op.Reshape4 { Ops4.Reshape4.params = { shape }; x = op_of x })
  (* Same shape as [Reshape] just above: the target is typed [Shape4.t], so a
     broadcast that fans an axis onto T or D is refused HERE, by [shape4],
     rather than by a domain-check arm -- [Domain.check_node] admits every
     [Expand] unconditionally, for the reason its own comment gives. *)
  | Expand { Pointwise.Expand.params; x } ->
      let* size = shape4 ~id:(single ()) params.Pointwise.Expand.size in
      simple (Op.Expand4 { Ops4.Expand4.params = { size }; x = op_of x })
  (* A direct counterpart with only the axis KEYS converted: the signed amounts
     and the mode cross unchanged, and [Eval_op4] runs the very same
     [Pad.Pad.Compute] functor over f32 values on both sides, so no value
     changes and the claim is [Identical].

     [Domain.check_node] has already refused T and D, so this conversion cannot
     fail — it is still written as a conversion rather than asserted away, for
     the reason the [Unbind] arm below gives: the domain check and this match
     disagreeing is a bug worth reporting, not worth raising on.

     The pairs are rebuilt in ONE traversal rather than mapping the axes and
     re-pairing them, so there is no second list whose length could drift from
     the entries it is paired with. *)
  (* A direct counterpart with only the axis converted: the bounds are already
     canonical against the same extent on both sides, and [Eval_op4] runs the
     very same [Split.Slice.Compute], so no value changes and the claim is
     [Identical]. T and D have already been refused by [Domain.check_node], for
     the reason the [Unbind] arm below gives. *)
  | Slice { Split.Slice.params; x } ->
      let* axis = dims4 ~node [ params.axis ] in
      simple
        (Op.Slice4
           {
             Ops4.Slice4.params =
               {
                 axis = List.hd axis;
                 start = params.start;
                 stop = params.stop;
                 step = params.step;
               };
             x = op_of x;
           })
  (* A direct counterpart with only the axis converted: every operand's shape
     already agrees off the joined axis (Native's own shape rule), and
     [Eval_op4] runs the very same [Concat.Concat.Compute], so no value
     changes and the claim is [Identical]. T and D have already been refused
     by [Domain.check_node], for the reason the [Unbind] arm gives. *)
  | Concat { Concat.Concat.params; xs } ->
      let* axis = dims4 ~node [ params.axis ] in
      simple
        (Op.Concat4
           {
             Ops4.Concat4.params = { axis = List.hd axis };
             xs = List.map op_of xs;
           })
  | Pad { Pad.Pad.params; x } ->
      let* pads =
        Err.List.map
          (fun (axis, entry) ->
            let+ axis4 =
              Axis4.of_axis axis
              |> Err.of_option (`Axis_outside_dialect (node, axis))
            in
            (axis4, entry))
          params.Pad.Pad.pads
      in
      simple
        (Op.Pad4
           {
             Ops4.Pad4.params = { pads; mode = params.Pad.Pad.mode };
             x = op_of x;
           })
  (* §7.3. The Native weight is already [Out,1,1,1,1,In] — literally a 1x1
     convolution weight — so this is a params-only rewrite with no data
     movement, and the spatial singleton loops add no arithmetic, leaving the
     input-channel reduction order unchanged. Identical. *)
  | Linear { Linear.Linear.params; x; weight; bias } ->
      simple
        (Op.Conv2d
           {
             Ops4.Conv_payload.params =
               unit_conv_params
                 ~in_channels:(Dim.to_int params.Linear.Linear.in_features);
             x = op_of x;
             weight = op_of weight;
             bias = Option.map op_of bias;
           })
  (* [Conv1d]'s own H window is always [Conv2d.unit_window] by construction
     (conv_conv1d.ml), so translating through [Conv.Conv1d.to_conv2d_params]
     and reusing [forward_conv] unchanged is the SAME "map onto an existing
     op after translating parameters" legalization [Linear] gets above --
     Native4D gains no new op or payload for it, just another source of a
     [Conv2D]/[DepthwiseConv2D]/[GroupedConv2D] the dialect already has. *)
  | Conv1d { Conv.Conv1d.params; x; weight; bias } ->
      let* weight_shape = sig_of weight in
      let* op =
        forward_conv ~node
          ~params:(Conv.Conv1d.to_conv2d_params params)
          ~x:(op_of x) ~weight:(op_of weight) ~bias:(Option.map op_of bias)
          ~weight_shape
      in
      simple op
  | Conv2d { Conv.Conv2d.params; x; weight; bias } ->
      let* weight_shape = sig_of weight in
      let* op =
        forward_conv ~node ~params ~x:(op_of x) ~weight:(op_of weight)
          ~bias:(Option.map op_of bias) ~weight_shape
      in
      simple op
  | Conv2d_padding { Conv.Conv2d_padding.params; x; weight; bias } ->
      let* weight_shape = sig_of weight in
      (* "same"/"valid" resolve to explicit windows first, through Native's own
         translation — restating it here would be a second definition free to
         drift from the one the compute uses. *)
      let* params =
        Err.map_error
          (fun _ -> `Unsupported_op (node, n.Node.op))
          (Conv.Conv2d_padding.to_conv2d_params ~weight_shape params)
      in
      let* op =
        forward_conv ~node ~params ~x:(op_of x) ~weight:(op_of weight)
          ~bias:(Option.map op_of bias) ~weight_shape
      in
      simple op
  | Convolution { Conv.Convolution.params; x; weight; bias } ->
      let* weight_shape = sig_of weight in
      if params.Conv.Convolution.transposed then
        let groups = (params.Conv.Convolution.groups :> int) in
        if groups <> 1 then
          Err.fail (`Unsupported_grouped_transposed_conv (node, groups))
        else
          simple
            (Op.Transposed_conv2d
               {
                 Ops4.Transposed_conv2d.params =
                   {
                     stride = params.Conv.Convolution.stride;
                     padding = params.Conv.Convolution.padding;
                     dilation = params.Conv.Convolution.dilation;
                     output_padding = params.Conv.Convolution.output_padding;
                   };
                 x = op_of x;
                 weight = op_of weight;
                 bias = Option.map op_of bias;
               })
      else
        let* params =
          Err.map_error
            (fun _ -> `Unsupported_op (node, n.Node.op))
            (Conv.Convolution.to_conv2d_params ~weight_shape params)
        in
        let* op =
          forward_conv ~node ~params ~x:(op_of x) ~weight:(op_of weight)
            ~bias:(Option.map op_of bias) ~weight_shape
        in
        simple op
  (* §7.5, and the [Amax]/[Sum]/[Vector_norm] arms below it, all corrected by
     C1 through [lower_keepdim_reduction] -- see its own comment above. The
     keepdim=false reshape target is the PACKED Native output shape, which
     for a reduction over H,W puts the batch extent on D, so it re-enters the
     dialect only when that extent is 1; [shape4] is where that is caught. *)
  | Mean { Reduce.Mean.params; x } ->
      lower_keepdim_reduction ~orig_dims:params.Reduce.Mean.dims
        ~keepdim:params.Reduce.Mean.keepdim
        ~keepdims_op:(fun dims x ->
          Op.Mean_keepdims { Ops4.Mean_keepdims.params = { dims }; x })
        ~x
  | Amax { Reduce.Amax.params; x } ->
      lower_keepdim_reduction ~orig_dims:params.Reduce.Amax.dims
        ~keepdim:params.Reduce.Amax.keepdim
        ~keepdims_op:(fun dims x ->
          Op.Max_keepdims { Ops4.Max_keepdims.params = { dims }; x })
        ~x
  | Sum { Reduce.Sum.params; x } ->
      lower_keepdim_reduction ~orig_dims:params.Reduce.Sum.dims
        ~keepdim:params.Reduce.Sum.keepdim
        ~keepdims_op:(fun dims x ->
          Op.Sum_keepdims { Ops4.Sum_keepdims.params = { dims }; x })
        ~x
  | Vector_norm { Reduce.Vector_norm.params; x } ->
      lower_keepdim_reduction ~orig_dims:params.Reduce.Vector_norm.dims
        ~keepdim:params.Reduce.Vector_norm.keepdim
        ~keepdims_op:(fun dims x ->
          Op.Vector_norm_keepdims
            { Ops4.Vector_norm_keepdims.params = { dims }; x })
        ~x
  (* §7.4. [Bmm]'s shape is exactly [Batched_matmul]'s at N=T=D=1: both
     read [input]/[mat2] at the same coordinates once [mat2]'s N/T/D/H are
     read off the OUTPUT (as [Batched_matmul.Compute] does) rather than
     hard-coded to index 0 (as [Bmm.Compute] does) -- at extent 1 those agree
     bit-for-bit, so the claim stays [Identical] at any batch [H], not just
     [H = 1]. This retires the previous 1x1-convolution legalization (Permute4
     + Conv2d, sound only at batch 1 and only for an f32-exact [mat2] format,
     since it MATERIALIZED [mat2] through the permute where both ops here read
     it directly): a direct counterpart, one node, no relayout, no format
     restriction, no batch restriction beyond [Domain]'s existing D = 1.
     Native's own [Batched_matmul] payload names no axis and carries no shape
     and so crosses unchanged; [Bmm]'s payload has the same two fields and
     crosses the same way. *)
  | Bmm { Matmul.Bmm.input; mat2 }
  | Batched_matmul { Matmul.Batched_matmul.input; mat2 } ->
      simple
        (Op.Batched_matmul
           { Matmul.Batched_matmul.input = op_of input; mat2 = op_of mat2 })
  (* §7.6, the only Equivalent legalization. *)
  | Batch_norm bn ->
      let* x_shape = sig_of bn.Norm.BatchNorm.x in
      let channels = Dim.to_int (Vec6.get x_shape Axis.C) in
      let eps = bn.Norm.BatchNorm.params.Norm.BatchNorm.eps in
      let* wid, bid, acc = batch_norm_weights acc ~node ~channels ~eps bn in
      let acc =
        {
          acc with
          claims = (single (), Correspondence.Equivalent) :: acc.claims;
          (* Provenance is a factual claim about what was computed from what,
             so it follows the arithmetic: scale = gamma / sqrt(var + eps)
             reads the variance and the optional weight but NOT the mean;
             offset = beta - mean * scale reads all of them. *)
          provenance =
            ( bn.Norm.BatchNorm.running_var
              :: Option.to_list bn.Norm.BatchNorm.weight,
              wid )
            :: ( [
                   bn.Norm.BatchNorm.running_mean; bn.Norm.BatchNorm.running_var;
                 ]
                 @ List.filter_map Fun.id
                     [ bn.Norm.BatchNorm.weight; bn.Norm.BatchNorm.bias ],
                 bid )
            :: acc.provenance;
        }
      in
      Err.return
        (emit acc ~from:node
           (Op.Depthwise_conv2d
              {
                Ops4.Conv_payload.params =
                  unit_conv_params ~in_channels:channels;
                x = op_of bn.Norm.BatchNorm.x;
                weight = wid;
                bias = Some bid;
              })
           [ single () ])
  (* The dialect's only multi-output node. The COMPLETE ordered output list is
     carried over unchanged, which is the whole of the correspondence work:
     under the id policy an edge whose value is preserved keeps its source id
     and so appears in no cluster, making every slice implicitly [Identical].
     Reordering or dropping one would be silent here — [Graph_map]'s output
     check is positional over the GRAPH's signature, not over a node's, so a
     swap inside this list is caught by [Map_verify] comparing the per-ordinal
     stages, and nowhere earlier.

     The axis converts here rather than in [Domain.check_node], so the domain's
     [Axis_outside_dialect] diagnostic can still name the rejected Native axis;
     by this point T/D have already been refused and [dims4] cannot fail. *)
  | Unbind { Split.Unbind.params; x } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      (* Every slice, checked HERE. [Domain.check_shapes] only inspects tensors
         that are live, so an unbind whose slices are all dead reaches this arm
         unvalidated; without this it would fail later inside
         [Snapshot4.create] as [`View _], which [Me_classify.native4d] calls
         Fatal — reporting a graph outside the dialect as our own defect, and
         naming no node. [shape4] gives the accurate row and the tensor id. *)
      let+ () =
        Err.List.iter
          (fun o ->
            let* shape = sig_of o in
            let+ (_ : Shape4.t) = shape4 ~id:o shape in
            ())
          n.Node.outputs
      in
      emit acc ~from:node
        (Op.Unbind
           { Ops4.Unbind.params = { axis = List.hd axis4 }; x = op_of x })
        n.Node.outputs
  (* [Unbind]'s rank-preserving sibling: the axis converts here for the same
     reason (so [Domain]'s [Axis_outside_dialect] can still name the rejected
     Native axis), and every output is checked here for the same reason (a
     dead slice would otherwise reach [Snapshot4.create] unvalidated). [sizes]
     crosses unchanged -- Native has already bounded its length and proved it
     sums to the axis extent, so there is nothing left for this arm to
     recheck. *)
  | Split_with_sizes { Split.Split_with_sizes.params; x } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      let+ () =
        Err.List.iter
          (fun o ->
            let* shape = sig_of o in
            let+ (_ : Shape4.t) = shape4 ~id:o shape in
            ())
          n.Node.outputs
      in
      emit acc ~from:node
        (Op.Split_with_sizes4
           {
             Ops4.Split_with_sizes4.params =
               {
                 axis = List.hd axis4;
                 sizes = params.Split.Split_with_sizes.sizes;
               };
             x = op_of x;
           })
        n.Node.outputs
  (* The axis converts here for the same reason [Unbind]'s does (so
     [Domain]'s [Axis_outside_dialect] can still name the rejected Native
     axis); the single output is checked here for the same reason [Unbind]'s
     are -- [Select] drops its axis, so the packed result re-enters the
     dialect only when [Shape4.of_vec6] accepts it, and a dead output would
     otherwise reach [Snapshot4.create] unvalidated. [index] crosses
     unchanged -- Native has already resolved it against the axis extent, so
     there is nothing left for this arm to recheck. *)
  | Select { Split.Select.params; x } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      let* () =
        let o = single () in
        let* shape = sig_of o in
        let+ (_ : Shape4.t) = shape4 ~id:o shape in
        ()
      in
      simple
        (Op.Select4
           {
             Ops4.Select4.params =
               { axis = List.hd axis4; index = params.Split.Select.index };
             x = op_of x;
           })
  (* The axis converts here for the same reason [Select]'s does. No post-hoc
     output re-check, unlike [Select]'s: this op's output shape is [self]'s
     OWN shape unchanged (no drop, no repack -- [Split.Select_scatter]'s own
     [output_shape] returns [self_shape] verbatim), so if [self] is already
     four-axis the output automatically is too, whichever axis this op
     names. *)
  | Select_scatter { Split.Select_scatter.params; self; src } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      simple
        (Op.Select_scatter4
           {
             Ops4.Select_scatter4.params =
               {
                 axis = List.hd axis4;
                 index = params.Split.Select_scatter.index;
               };
             self = op_of self;
             src = op_of src;
           })
  (* [Concat]'s variadic-operand handling above, plus [Select]'s post-hoc
     output check: [Stack] INSERTS an axis rather than keeping every one the
     way [Concat] does, so -- the same reason [Select]'s arm re-validates its
     single output -- the packed result re-enters the dialect only when
     [Shape4.of_vec6] accepts it. The axis converts here for the same reason
     [Select]'s does: so [Domain]'s [Axis_outside_dialect] can still name the
     rejected Native axis. *)
  | Stack { Concat.Stack.params; xs } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      let* () =
        let o = single () in
        let* shape = sig_of o in
        let+ (_ : Shape4.t) = shape4 ~id:o shape in
        ()
      in
      simple
        (Op.Stack4
           {
             Ops4.Stack4.params = { axis = List.hd axis4 };
             xs = List.map op_of xs;
           })
  (* The axis converts here for the same reason [Select]'s/[Stack]'s does. No
     post-hoc output re-check, unlike theirs: this op's output shape is [x]'s
     OWN shape unchanged ([Reduce.Softmax.output_shape] returns [x_shape]
     verbatim, the same fact [Select_scatter]'s arm above relies on), so if
     [x] is already four-axis the output automatically is too, whichever axis
     this op reduces over. *)
  | Softmax { Reduce.Softmax.params; x } ->
      let* axis4 = dims4 ~node [ params.axis ] in
      simple
        (Op.Softmax4
           { Ops4.Softmax4.params = { axis = List.hd axis4 }; x = op_of x })
  (* Direct counterpart, once [Domain.check] has proved D = 1: [Attention.Sdpa.t]
     names no axis and carries no shape, so it crosses unchanged, and
     [Region_computation4]'s [native_op] routes it back through the exact same
     [Region_program] Native uses -- no second numeric kernel. *)
  | Sdpa { Attention.Sdpa.params; query; key; value; mask } ->
      simple
        (Op.Sdpa
           {
             Attention.Sdpa.params;
             query = op_of query;
             key = op_of key;
             value = op_of value;
             mask = Option.map op_of mask;
           })
  (* Rejected by [Domain.check] before the walk starts; reaching them means the
     domain check and this match disagree, which is a bug in one of them.
     [Index_tensor] joins that set until its own counterpart exists (see
     [Domain.check_node]'s comment) rather than gaining a real conversion arm
     here -- it has no Native4D counterpart at all yet, the same "dialect does
     not have it" answer. [Repeat]/[RepeatInterleave]/[Select_scatter]/
     [Softmax]/[Batched_matmul]/[Sdpa] no longer join them: all six now have
     real conversion arms above. *)
  | Adaptive_max_pool2d_with_indices _ | Discard _ | Index_tensor _
  | Max_pool2d_with_indices _ ->
      Err.fail (`Unsupported_op (node, n.Node.op))
