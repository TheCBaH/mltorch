(* See lower.mli.

   THE ID POLICY, which is the decision everything else hangs off. An edge whose
   value the conversion preserves keeps its SOURCE raw id; every created edge
   takes a fresh id above the source watermark. That is forced, not chosen:
   [Graph_map.create] rejects a source id that is neither mentioned nor deleted
   but absent from the destination, so without preservation every edge would
   need an explicit cluster and the implicit-identity bulk that keeps a map
   proportional to what changed would disappear.

   The hazard — one raw id, two different values — is what claim closure and the
   local verifier catch, and stage 6 adds the crossed-operand mutation that
   proves they do. *)

(* Bound BEFORE [open Graph_ir], which shadows [Graph] with the Native one. *)
module G4 = Graph
open Graph_ir
open Err.Syntax

type ('src, 'dst) t = {
  dst : 'dst Framework.Snapshot4.t;
  map : ('src, 'dst) Graph_map.t;
  constants : Tensor.packed Tensor_id.Map.t;
  constant_store : Constant_store.t;
}

type 'src packed = Pack : ('src, 'dst) t -> 'src packed

let graph r = Framework.Snapshot4.graph r.dst

type eval_error =
  [ Const_ssa_materialize.error | `Direct4 of Eval_direct4.error ]

let pp_eval_error ppf : [< eval_error ] -> unit = function
  | #Const_ssa_materialize.error as e -> Const_ssa_materialize.pp_error ppf e
  | `Direct4 e -> Eval_direct4.pp_error ppf e

let evaluate resolver r ~inputs =
  let open Err.Syntax in
  let* store, report =
    Const_ssa_materialize.materialize resolver r.constant_store
    |> Err.map_error (fun e -> (e :> eval_error))
  in
  let constants =
    Tensor_id.Map.union
      (fun _ _ planned -> Some planned)
      r.constants
      (Constant_store.materialized store)
  in
  let+ env =
    Eval_direct4.run (graph r)
      ~constants:(Tensor_id.Map.bindings constants)
      ~inputs
    |> Err.map_error (fun e -> `Direct4 e)
  in
  (env, report)

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
   (Mean keepdim=false, Bmm) takes a fresh id above the source watermark.
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

(* Grouping becomes a constructor. §7.2: one group is [Conv2D], one input
   channel per group is [DepthwiseConv2D], anything else was rejected by
   [Domain.check] — so an unexpected count here is a domain-check bug, not a
   user error, and it is reported as the same typed error rather than asserted
   away. *)
let forward_conv ~node ~params ~x ~weight ~bias ~weight_shape =
  let groups = (params.Conv.Conv2d.groups :> int) in
  let payload =
    { Ops4.Conv_payload.params = conv_params params; x; weight; bias }
  in
  if groups = 1 then Err.return (Op.Conv2d payload)
  else if Dim.to_int (Vec6.get weight_shape Axis.C) = 1 then
    Err.return (Op.Depthwise_conv2d payload)
  else Err.fail (`Unsupported_grouped_conv (node, groups))

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

let unit_window : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent 1;
    stride = Op_config.Pos.of_int 1;
    pad_before = Op_config.Nonneg.of_int 0;
    pad_after = Op_config.Nonneg.of_int 0;
    dilation = Op_config.Pos.of_int 1;
  }

let unit_conv_params ~in_channels : Ops4.Conv_params.t =
  { h = unit_window; w = unit_window; in_channels = Dim.extent in_channels }

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
  | Relu { Pointwise.Relu.x } -> simple (Op.Relu { Pointwise.Relu.x = op_of x })
  | Sigmoid { Pointwise.Sigmoid.x } ->
      simple (Op.Sigmoid { Pointwise.Sigmoid.x = op_of x })
  | Silu { Pointwise.Silu.x } -> simple (Op.Silu { Pointwise.Silu.x = op_of x })
  | Sqrt { Pointwise.Sqrt.x } -> simple (Op.Sqrt { Pointwise.Sqrt.x = op_of x })
  | Avg_pool2d { Pool.AvgPool2d.params; x } ->
      simple (Op.Avg_pool2d { Pool.AvgPool2d.params; x = op_of x })
  | Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x } ->
      simple
        (Op.Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x = op_of x })
  | Max_pool2d { Pool.MaxPool2d.params; x } ->
      simple (Op.Max_pool2d { Pool.MaxPool2d.params; x = op_of x })
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
  (* §7.5 as corrected by C1. keepdim=false becomes MeanKeepDims + Reshape4, and
     the reshape target is the PACKED Native output shape — which for a
     reduction over H,W puts the batch extent on D, so it re-enters the dialect
     only when that extent is 1. [shape4] is where that is caught. *)
  | Mean { Reduce.Mean.params; x } ->
      let* dims = dims4 ~node params.Reduce.Mean.dims in
      if params.Reduce.Mean.keepdim then
        simple
          (Op.Mean_keepdims
             { Ops4.Mean_keepdims.params = { dims }; x = op_of x })
      else
        let* x_shape = sig_of x in
        let* kept =
          match
            Reduce.Mean.output_shape ~x_shape
              { Reduce.Mean.dims = params.Reduce.Mean.dims; keepdim = true }
          with
          | Ok s -> Err.return s
          | Error _ -> Err.fail (`Unsupported_op (node, n.Node.op))
        in
        let* packed = sig_of (single ()) in
        let* kept4 = shape4 ~id:(single ()) kept in
        let* packed4 = shape4 ~id:(single ()) packed in
        let mid, acc = fresh_tensor acc kept4 in
        let acc =
          emit acc ~from:node
            (Op.Mean_keepdims
               { Ops4.Mean_keepdims.params = { dims }; x = op_of x })
            [ mid ]
        in
        let acc =
          { acc with provenance = ([ op_of x ], mid) :: acc.provenance }
        in
        Err.return
          (emit acc ~from:node
             (Op.Reshape4
                { Ops4.Reshape4.params = { shape = packed4 }; x = mid })
             [ single () ])
  (* Same shape as [Mean] above, corrected by C1 the same way: keepdim=false
     becomes MaxKeepDims + Reshape4, and the reshape target re-enters the
     dialect only when the packed Native output shape stays four-axis. *)
  | Amax { Reduce.Amax.params; x } ->
      let* dims = dims4 ~node params.Reduce.Amax.dims in
      if params.Reduce.Amax.keepdim then
        simple
          (Op.Max_keepdims { Ops4.Max_keepdims.params = { dims }; x = op_of x })
      else
        let* x_shape = sig_of x in
        let* kept =
          match
            Reduce.Amax.output_shape ~x_shape
              { Reduce.Amax.dims = params.Reduce.Amax.dims; keepdim = true }
          with
          | Ok s -> Err.return s
          | Error _ -> Err.fail (`Unsupported_op (node, n.Node.op))
        in
        let* packed = sig_of (single ()) in
        let* kept4 = shape4 ~id:(single ()) kept in
        let* packed4 = shape4 ~id:(single ()) packed in
        let mid, acc = fresh_tensor acc kept4 in
        let acc =
          emit acc ~from:node
            (Op.Max_keepdims
               { Ops4.Max_keepdims.params = { dims }; x = op_of x })
            [ mid ]
        in
        let acc =
          { acc with provenance = ([ op_of x ], mid) :: acc.provenance }
        in
        Err.return
          (emit acc ~from:node
             (Op.Reshape4
                { Ops4.Reshape4.params = { shape = packed4 }; x = mid })
             [ single () ])
  (* Same shape as [Mean]/[Amax] above, corrected by C1 the same way. *)
  | Vector_norm { Reduce.Vector_norm.params; x } ->
      let* dims = dims4 ~node params.Reduce.Vector_norm.dims in
      if params.Reduce.Vector_norm.keepdim then
        simple
          (Op.Vector_norm_keepdims
             { Ops4.Vector_norm_keepdims.params = { dims }; x = op_of x })
      else
        let* x_shape = sig_of x in
        let* kept =
          match
            Reduce.Vector_norm.output_shape ~x_shape
              {
                Reduce.Vector_norm.dims = params.Reduce.Vector_norm.dims;
                keepdim = true;
              }
          with
          | Ok s -> Err.return s
          | Error _ -> Err.fail (`Unsupported_op (node, n.Node.op))
        in
        let* packed = sig_of (single ()) in
        let* kept4 = shape4 ~id:(single ()) kept in
        let* packed4 = shape4 ~id:(single ()) packed in
        let mid, acc = fresh_tensor acc kept4 in
        let acc =
          emit acc ~from:node
            (Op.Vector_norm_keepdims
               { Ops4.Vector_norm_keepdims.params = { dims }; x = op_of x })
            [ mid ]
        in
        let acc =
          { acc with provenance = ([ op_of x ], mid) :: acc.provenance }
        in
        Err.return
          (emit acc ~from:node
             (Op.Reshape4
                { Ops4.Reshape4.params = { shape = packed4 }; x = mid })
             [ single () ])
  (* §7.4. Only a single batch legalizes: for batch > 1 mat2 varies with the
     output's H coordinate while convolution weights are shared across spatial
     positions. mat2[H=1,W=contract,C=cols] permutes to the weight layout
     [N=cols,H=1,W=1,C=contract], and the 1x1 convolution's output is already
     Bmm's own shape, so NO trailing reshape is needed. *)
  | Bmm { Matmul.Bmm.input; mat2 } ->
      let* input_shape = sig_of input in
      let batch = Vec6.get input_shape Axis.H in
      if Dim.to_int batch <> 1 then
        Err.fail (`Unsupported_bmm_batch (node, batch))
      else
        let* mat2_shape = sig_of mat2 in
        let contract = Vec6.get mat2_shape Axis.W in
        let cols = Vec6.get mat2_shape Axis.C in
        let w_shape =
          Shape4.make ~n:cols ~h:(Dim.extent 1) ~w:(Dim.extent 1) ~c:contract
        in
        let wid, acc = fresh_tensor acc w_shape in
        let acc =
          emit acc ~from:node
            (Op.Permute4
               {
                 (* mat2 [H=1,W=contract,C=cols] -> weight
                    [N=cols,H=1,W=1,C=contract]. Every output axis takes a
                    DISTINCT input axis, which is what makes it a bijection:
                    N<-C, H<-H, W<-N, C<-W. Reusing an input axis (the obvious
                    N<-C, C<-W) leaves N unread and the perm is rejected. *)
                 Ops4.Permute4.perm =
                   Ops4.Permute4.of_fn (function
                     | Axis4.N -> Axis4.C
                     | Axis4.W -> Axis4.N
                     | Axis4.C -> Axis4.W
                     | Axis4.H -> Axis4.H);
                 x = op_of mat2;
               })
            [ wid ]
        in
        let acc =
          { acc with provenance = ([ op_of mat2 ], wid) :: acc.provenance }
        in
        Err.return
          (emit acc ~from:node
             (Op.Conv2d
                {
                  Ops4.Conv_payload.params =
                    unit_conv_params ~in_channels:(Dim.to_int contract);
                  x = op_of input;
                  weight = wid;
                  bias = None;
                })
             [ single () ])
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
  (* Rejected by [Domain.check] before the walk starts; reaching them means the
     domain check and this match disagree, which is a bug in one of them.
     [Group_norm]/[Select]/[Split_with_sizes]/[Stack] join that set until
     [Group_norm4]/[Select4]/[Split4]/[Stack4] exist (see
     [Domain.check_node]'s comment) rather than gaining a real conversion arm
     here. *)
  | Max_pool2d_with_indices _ | Discard _ | Sdpa _ | Group_norm _ | Select _
  | Split_with_sizes _ | Stack _ ->
      Err.fail (`Unsupported_op (node, n.Node.op))

(* ---- constants ------------------------------------------------------------ *)

(* Validated over the WHOLE supplied map before any legalization runs, against
   the same contract [Rewrite.origin] holds its own payloads to: shape and
   format must match the recorded signature, and the id must be an effective
   constant. Checking only what batch norm consumes would not do — every other
   payload is copied through into [result.constants] and evaluated much later,
   so a malformed conv weight would cross the boundary intact and fail far from
   the mistake. *)
let fmt_name (Payload.Fmt f) = Payload.fmt_name f

let check_constants ~view constants =
  Err.List.iter
    (fun (id, Tensor.Tensor payload) ->
      let* () =
        if Graph_view.is_constant view id then Err.return ()
        else Err.fail (`Bad_constant_payload id)
      in
      match Graph_view.sig_of view id with
      | None -> Err.fail (`Bad_constant_payload id)
      | Some sg ->
          let shape_ok =
            List.for_all
              (fun axis ->
                Dim.equal
                  (Vec6.get sg.Tensor_sig.shape axis)
                  (Vec6.get payload.Tensor.shape axis))
              Axis.all
          in
          if
            shape_ok
            && String.equal
                 (fmt_name sg.Tensor_sig.fmt)
                 (Payload.fmt_name payload.Tensor.payload.Payload.fmt)
          then Err.return ()
          else Err.fail (`Bad_constant_payload id))
    (Tensor_id.Map.bindings constants)

(* ---- the conversion ------------------------------------------------------- *)

let convert ?(constants = Tensor_id.Map.empty)
    ?(constant_store = Constant_store.empty) (src : 'src Snapshot.t) =
  let view = Snapshot.view src in
  let g = Snapshot.graph src in
  let* () = (Domain.check view :> (unit, Error.t) Err.t) in
  let* () = check_constants ~view constants in
  (* Fresh ids start above the source watermark, so a created edge can never
     collide with a preserved one. *)
  let watermark =
    Tensor_id.Map.fold
      (fun id _ acc -> max acc (Tensor_id.to_int id + 1))
      g.Graph.tensors 0
  in
  (* An unread constant is model-bound state, not interface, so it is OMITTED
     and recorded as a deletion — the same seam [Rewrite.apply] cuts along ("a
     constant nobody reads is gone"). An unused user INPUT is not omitted; it is
     part of the signature, and [Domain.check] already rejected a non-four-axis
     one. *)
  let read id =
    List.exists
      (fun (n : node) -> List.mem id (Graph_ir.operands n.Node.op))
      g.Graph.nodes
    || List.mem id g.Graph.outputs
  in
  let kept_inputs, dropped_inputs =
    List.partition
      (fun id -> Graph_ir.input_kind g id = Input.Input || read id)
      g.Graph.inputs
  in
  let acc0 =
    {
      nodes = [];
      tensors =
        Tensor_id.Map.filter
          (fun id _ -> not (List.mem id dropped_inputs))
          g.Graph.tensors;
      subst = Tensor_id.Map.empty;
      next_tid = watermark;
      next_nid =
        List.fold_left
          (fun acc (n : node) -> max acc (Node_id.to_int n.Node.id + 1))
          0 g.Graph.nodes;
      created = [];
      deleted = dropped_inputs;
      claims = [];
      node_pairs = [];
      provenance = [];
      constants;
      fresh_constants = [];
    }
  in
  let* acc = Err.List.fold_left (lower_node ~view) acc0 g.Graph.nodes in
  let dst_graph =
    {
      G4.Graph.nodes = List.rev acc.nodes;
      root =
        {
          Graph_common.Group.id = Graph_common.Group_id.of_int 0;
          label = None;
          items =
            List.map
              (fun (n : G4.node) -> Graph_common.Group.Node n.G4.Node.id)
              (List.rev acc.nodes);
        };
      tensors = acc.tensors;
      inputs = kept_inputs @ List.rev acc.fresh_constants;
      input_kinds =
        List.fold_left
          (fun m id -> Tensor_id.Map.add id Input.Constant m)
          (Tensor_id.Map.filter
             (fun id _ -> List.mem id kept_inputs)
             g.Graph.input_kinds)
          acc.fresh_constants;
      outputs = List.map (resolve acc) g.Graph.outputs;
    }
  in
  let* (Framework.Snapshot4.Pack dst) =
    Err.map_error (fun e -> `View e) (Framework.Snapshot4.create dst_graph)
  in
  let* constant_store =
    Constant_store.restrict_and_rename_exports constant_store (fun id ->
        if
          Graph_common.input_kind dst_graph id = Input.Constant
          && Tensor_id.Map.mem id dst_graph.G4.Graph.tensors
        then Some id
        else None)
    |> Err.map_error (fun e -> `Constant_store e)
  in
  (* Claims, PROPAGATED FORWARD. One claim per legalized node is not enough: the
     moment any legalization is weaker than Identical every edge downstream is
     too, and an edge left implicit reads as Identical — which
     [Graph_map.create]'s closure check rejects. [Rewrite.apply]'s steps 9-10 are
     the template, and the table is the DESTINATION dialect's. *)
  let explicit =
    List.fold_left
      (fun m (id, rel) -> Tensor_id.Map.add id rel m)
      Tensor_id.Map.empty acc.claims
  in
  let propagated =
    Output_transfer4.propagate ~explicit
      ~preserved:(fun id -> Snapshot.edge src id <> None)
      dst_graph
  in
  let all_claims =
    Tensor_id.Map.union (fun _ a _ -> Some a) explicit propagated
  in
  let edge_src id = Snapshot.edge src id in
  let edge_dst id = Framework.Snapshot4.edge dst id in
  let value_clusters =
    (* Weaker-than-Identical edges, stated outright. *)
    Tensor_id.Map.fold
      (fun id rel acc ->
        match (edge_src id, edge_dst id) with
        | Some s, Some d -> Correspondence.pair s d rel :: acc
        | _ -> acc)
      all_claims []
    (* Clone removal is {removed, kept} <-> {kept}, not {removed} <-> {kept}.
       The surviving edge is present in BOTH graphs, so mentioning it only as a
       destination makes it [Unpaired_dst]: an id named on one side while
       present in both must be named on the other. This is the same shape
       .ai/native_transform_design.md §3 gives for trimming an identity
       permute. *)
    @ List.filter_map
        (fun (removed, kept) ->
          match (edge_src removed, edge_src kept, edge_dst kept) with
          | Some removed, Some kept_src, Some kept_dst ->
              Some
                {
                  Correspondence.Cluster.src =
                    Correspondence.Set.of_list [ removed; kept_src ];
                  dst = Correspondence.Set.singleton kept_dst;
                  label = Correspondence.Identical;
                }
          | _ -> None)
        (Tensor_id.Map.bindings acc.subst)
    @ List.filter_map
        (fun id -> Option.map Correspondence.delete (edge_src id))
        acc.deleted
    @ List.filter_map
        (fun id -> Option.map Correspondence.create (edge_dst id))
        acc.created
  in
  (* One source node may become SEVERAL destination nodes — Mean keepdim=false
     becomes MeanKeepDims plus Reshape4, Bmm becomes Permute4 plus Conv2d — and
     [Node_map.fused] is the other direction (many sources, one destination).
     So the cluster is built directly: taking only the first destination would
     leave the second [Uncovered_dst]. *)
  let node_clusters =
    List.filter_map
      (fun (s, ds) ->
        let dst_ids =
          List.filter_map (fun d -> Framework.Snapshot4.node dst d) ds
        in
        match (Snapshot.node src s, dst_ids) with
        | Some s, (_ :: _ as ds) ->
            Some
              {
                Node_map.Cluster.src = Node_map.Set.singleton s;
                dst = Node_map.Set.of_list ds;
                label = ();
              }
        | _ -> None)
      acc.node_pairs
    (* A source node with no destination — [Clone], which contributes none — is
       a deletion, and saying nothing about it would leave it [Uncovered_src]. *)
    @ List.filter_map
        (fun (n : node) ->
          if
            List.exists (fun (s, _) -> Node_id.equal s n.Node.id) acc.node_pairs
          then None
          else Option.map Node_map.delete (Snapshot.node src n.Node.id))
        g.Graph.nodes
  in
  let provenance =
    List.fold_left
      (fun p (sources, target) ->
        match Framework.Snapshot4.edge dst target with
        | None -> p
        | Some target ->
            (* Sources that still exist in the source snapshot; one that does
               not simply contributes nothing. *)
            Provenance.add
              ~sources:
                (Correspondence.Set.of_list (List.filter_map edge_src sources))
              target p)
      Provenance.empty acc.provenance
  in
  let+ map =
    Err.map_error
      (fun e -> `Map e)
      (Framework.Map_from_native.create ~src ~dst ~values:value_clusters
         ~nodes:node_clusters ~provenance)
  in
  Pack { dst; map; constants = acc.constants; constant_store }
