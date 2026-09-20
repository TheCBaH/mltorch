(* A functional (state-monad) builder for the Native4D IR, the twin of
   [Graph_builder]. Output shapes are COMPUTED via [Graph_shape4], never
   supplied.

   Its public constructors take [Shape4.t] and [Axis4.t], which is where the
   dialect's contract is enforced for anything built by hand: a caller cannot
   name T or D, and cannot state a shape that has extent on them. That is the
   acceptance criterion for stage 2 — "no non-4D [Shape4.t] constructible
   through the public API" — carried into graph construction. *)

type error =
  [ Graph_shape4.error | `Expected_single_output_shape of output_count ]

and output_count = { count : int }

type state = {
  next_tid : int;
  next_nid : int;
  dtype : Payload.packed_fmt;
  rev_nodes : Graph.node list;
  rev_items : Graph_ir.Group.item list;
  tensors : Tensor_sig.t Tensor_id.Map.t;
  rev_inputs : Tensor_id.t list;
  input_kinds : Graph_ir.Input.kind Tensor_id.Map.t;
}

type 'a t = state -> ('a, error) Err.t * state

let pp_error ppf : [< error ] -> unit = function
  | #Graph_shape4.error as e -> Graph_shape4.pp_error ppf e
  | `Expected_single_output_shape { count } ->
      Fmt.pf ppf "expected a single output shape, got %d" count

let return x s = (Ok x, s)
let lift_result (r : ('a, [< error ]) Err.t) s = ((r :> ('a, error) Err.t), s)

let ( let* ) m f s =
  match m s with Ok x, s' -> f x s' | Error e, s' -> (Error e, s')

let ( let+ ) m f s =
  match m s with Ok x, s' -> (Ok (f x), s') | Error e, s' -> (Error e, s')

let get s = (Ok s, s)
let f32 = Payload.Fmt Payload.F32

(* [shape] is a [Shape4.t]; the stored signature is the [Vec6.shape] it unwraps
   to, per correction C3 — the guard is on the way in, not in storage. *)
let source ~kind ~(shape : Shape4.t) ?fmt ?quant () s =
  let tid = Tensor_id.of_int s.next_tid in
  let sg =
    Tensor_sig.create ~id:tid ~name:"" ~shape:(Shape4.to_vec6 shape)
      ~fmt:(Option.value fmt ~default:s.dtype)
      ?quant ()
  in
  ( Ok tid,
    {
      s with
      next_tid = s.next_tid + 1;
      tensors = Tensor_id.Map.add tid sg s.tensors;
      rev_inputs = tid :: s.rev_inputs;
      input_kinds = Tensor_id.Map.add tid kind s.input_kinds;
    } )

let input ~shape ?fmt ?quant () =
  source ~kind:Graph_ir.Input.Input ~shape ?fmt ?quant ()

let constant ~shape ?fmt ?quant () =
  source ~kind:Graph_ir.Input.Constant ~shape ?fmt ?quant ()

let new_edge ?fmt ?quant (shape : Shape4.t) s =
  let tid = Tensor_id.of_int s.next_tid in
  let sg =
    Tensor_sig.create ~id:tid ~name:"" ~shape:(Shape4.to_vec6 shape)
      ~fmt:(Option.value fmt ~default:f32)
      ?quant ()
  in
  ( Ok tid,
    {
      s with
      next_tid = s.next_tid + 1;
      tensors = Tensor_id.Map.add tid sg s.tensors;
    } )

let push_node op outputs s =
  let nid = Graph_ir.Node_id.of_int s.next_nid in
  ( Ok (),
    {
      s with
      next_nid = s.next_nid + 1;
      rev_nodes = { Graph.Node.id = nid; op; outputs } :: s.rev_nodes;
      rev_items = Graph_ir.Group.Node nid :: s.rev_items;
    } )

(* Every op but [Unbind] is single-output, and this is the form they use: the
   arity check is not vestigial, it is what stops an op that grew an output from
   silently dropping it. [Unbind] uses [opN] below. *)
let op1 ?fmt op : Tensor_id.t t =
  let* s = get in
  let* shapes =
    lift_result
      (Graph_shape4.output_shape op ~sig_of:(fun r ->
           Tensor_id.Map.find_opt r s.tensors
           |> Err.of_option (`Missing_tensor_sig r)))
  in
  let* shape =
    match shapes with
    | [ sh ] -> return sh
    | _ ->
        fun s ->
          ( Err.fail
              (`Expected_single_output_shape { count = List.length shapes }),
            s )
  in
  let* tid = new_edge ?fmt shape in
  let* () = push_node op [ tid ] in
  return tid

(* The variable-arity form: one edge per inferred shape, one node holding all of
   them in order. No expected count to check against — for an op whose arity is
   part of its input signature there is none. The same shape as Native's
   [Graph_builder.opN], including the shared id-space guard, so the two dialects'
   overflow behaviour cannot drift. *)
let opN ?fmt ?quant op : Tensor_id.t list t =
  let* s = get in
  let* shapes =
    lift_result
      (Graph_shape4.output_shape op ~sig_of:(fun r ->
           Tensor_id.Map.find_opt r s.tensors
           |> Err.of_option (`Missing_tensor_sig r)))
  in
  Tensor_id.check_room ~next:s.next_tid ~count:(List.length shapes);
  (* Tail-recursive for the reason [Graph_builder.opN] documents: a monadic
     frame per output overflows node's stack at a few thousand outputs. *)
  let rec alloc acc = function
    | [] -> return (List.rev acc)
    | shape :: rest ->
        let* tid = new_edge ?fmt ?quant shape in
        alloc (tid :: acc) rest
  in
  let* ids = alloc [] shapes in
  let* () = push_node op ids in
  return ids

let batch_norm_no_stats ?fmt params ~x ?weight ?bias () =
  opN ?fmt
    (Op.Batch_norm_no_stats { Ops4.Batch_norm_no_stats.params; x; weight; bias })

let batch_norm ?fmt params ~x ?weight ?bias ~running_mean ~running_var () =
  op1 ?fmt
    (Op.Batch_norm
       { Ops4.Batch_norm.params; x; weight; bias; running_mean; running_var })

(* Op constructors in global alphabetical order, as in [Graph_builder]. *)

(* Thread the operand's own I64 format/quant into the output edge, matching
   Native's own [Graph_builder.add]/[reshape4]/[permute4]'s precedent above
   -- ONLY when both operands are I64, since [Eval_direct4]'s [Compute_i64]
   dispatch can only deliver an exact result when they agree; a mismatched
   pair falls through to [op1]'s F32 default (mixed promotion is out of this
   slice's scope, matching Native's own [add]). *)
let add a b =
  let* s = get in
  let a_sig = Tensor_id.Map.find a s.tensors in
  let b_sig = Tensor_id.Map.find b s.tensors in
  match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
  | Payload.Fmt Payload.I64, Payload.Fmt Payload.I64 ->
      op1 ~fmt:a_sig.Tensor_sig.fmt (Op.Add { Pointwise.Bin.a; b })
  | _ -> op1 (Op.Add { Pointwise.Bin.a; b })

let addcmul value self tensor1 tensor2 =
  op1
    (Op.Addcmul
       {
         Pointwise.Addcmul.self;
         tensor1;
         tensor2;
         value = Json_util.f32_to_f32 value;
       })

let add_scalar scalar x = op1 (Op.Add_scalar { Pointwise.Scalar_bin.x; scalar })

let adaptive_avg_pool2d params x =
  op1 (Op.Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x })

let adaptive_max_pool2d params x =
  op1 (Op.Adaptive_max_pool2d { Pool.AdaptiveMaxPool2d.params; x })

(* Two outputs, value then indices -- [opN], not [op1], with no [~fmt]
   override: both edges default to the same f32 [new_edge] falls back to,
   matching Native's own [Graph_builder.adaptive_max_pool2d_with_indices],
   which allocates both outputs the same way rather than through its own
   [opN]-equivalent. The indices output is a real f32-stored flat position,
   not a genuine integer format -- Native has none to give it. *)
let adaptive_max_pool2d_with_indices params x =
  opN
    (Op.Adaptive_max_pool2d_with_indices
       { Pool.AdaptiveMaxPool2dWithIndices.params; x })

let avg_pool2d params x = op1 (Op.Avg_pool2d { Pool.AvgPool2d.params; x })

let batched_matmul input mat2 =
  op1 (Op.Batched_matmul { Matmul.Batched_matmul.input; mat2 })

(* Unconditionally [Bool], matching [to_copy]'s own [Bool] case above and
   Native's own [Graph_builder.bitwise_not] -- real ATen's bitwise-complement
   on Native's only routed operand (a bool mask) produces a bool result. *)
let bitwise_not x =
  op1 ~fmt:Payload.(Fmt Bool) (Op.Bitwise_not { Pointwise.Bitwise_not.x })

let clamp params x = op1 (Op.Clamp { Pointwise.Clamp.params; x })
let col2im params x = op1 (Op.Col2im { Im2col.Col2im.params; x })

(* Takes the dialect's own [Ops4.Concat4.params], whose axis is [Axis4.t]: a
   concat naming T or D is not constructible through this API, the same rule
   [unbind] below follows. *)
let concat4 params xs = op1 (Op.Concat4 { Ops4.Concat4.params; xs })

let conv2d params ~x ~weight ?bias () =
  op1 (Op.Conv2d { Ops4.Conv_payload.params; x; weight; bias })

let depthwise_conv2d params ~x ~weight ?bias () =
  op1 (Op.Depthwise_conv2d { Ops4.Conv_payload.params; x; weight; bias })

let cos x = op1 (Op.Cos { Pointwise.Cos.x })

(* Takes the dialect's own [Ops4_cumsum.Cumsum4.params], whose axis is [Axis4.t]: a
   cumsum naming T or D is not constructible through this API. *)
let cumsum4 params x = op1 (Op.Cumsum4 { Ops4_cumsum.Cumsum4.params; x })
let div a b = op1 (Op.Div { Pointwise.Bin.a; b })
let div_scalar scalar x = op1 (Op.Div_scalar { Pointwise.Scalar_bin.x; scalar })

let floor_div_scalar scalar x =
  op1 (Op.Floor_div_scalar { Pointwise.Scalar_bin.x; scalar })

(* Takes a [Shape4.t] target, so an expansion naming T or D is not
   constructible through this API -- [reshape4]'s rule. *)
let expand4 size x = op1 (Op.Expand4 { Ops4.Expand4.params = { size }; x })

let gelu (approximate : Pointwise.Gelu.approximate) x =
  op1 (Op.Gelu { Pointwise.Gelu.x; approximate })

let group_norm4 params ~x ?weight ?bias () =
  op1 (Op.Group_norm4 { Ops4.Group_norm4.params; x; weight; bias })

let grouped_conv2d params ~x ~weight ?bias () =
  op1 (Op.Grouped_conv2d { Ops4.Grouped_conv_payload.params; x; weight; bias })

(* Unconditionally [Bool], matching [to_copy]/[bitwise_not]'s own convention
   above and Native's own [Graph_builder.gt_scalar] -- real ATen's [gt.
   Scalar] always produces a bool result. *)
let gt_scalar scalar x =
  op1 ~fmt:Payload.(Fmt Bool) (Op.Gt_scalar { Pointwise.Scalar_bin.x; scalar })

let hardsigmoid x = op1 (Op.Hardsigmoid { Pointwise.Hardsigmoid.x })
let hardswish x = op1 (Op.Hardswish { Pointwise.Hardswish.x })
let hardtanh params x = op1 (Op.Hardtanh { Pointwise.Hardtanh.params; x })
let im2col params x = op1 (Op.Im2col { Im2col.Im2col.params; x })

(* Takes the dialect's own [Ops4.IndexTensor4.params], whose axis is
   [Axis4.t]: a gather naming T or D is not constructible through this API,
   the same rule [select4]'s does. *)
let index_tensor4 params ~self ~index =
  op1 (Op.IndexTensor4 { Ops4.IndexTensor4.params; self; index })

let leaky_relu params x = op1 (Op.Leaky_relu { Pointwise.Leaky_relu.params; x })

(* Returns a real triple, the same shape [Graph_builder.lstm] uses, rather
   than [opN]'s list -- three distinctly-typed outputs (output, h_n, c_n) are
   checked once here instead of leaving every caller to pattern-match a
   3-element list. *)
let lstm params ~input ~layers ~h0 ~c0 () =
  let op = Op.Lstm { Lstm.Lstm.params; layers; input; h0; c0 } in
  let* s = get in
  let* shapes =
    lift_result
      (Graph_shape4.output_shape op ~sig_of:(fun r ->
           Tensor_id.Map.find_opt r s.tensors
           |> Err.of_option (`Missing_tensor_sig r)))
  in
  match shapes with
  | [ out_shape; hn_shape; cn_shape ] ->
      let* out_id = new_edge out_shape in
      let* hn_id = new_edge hn_shape in
      let* cn_id = new_edge cn_shape in
      let* () = push_node op [ out_id; hn_id; cn_id ] in
      return (out_id, hn_id, cn_id)
  | _ ->
      fun s ->
        ( Err.fail (`Expected_single_output_shape { count = List.length shapes }),
          s )

let max_keepdims ?(keepdim = true) dims x =
  op1 (Op.Max_keepdims { Ops4.Max_keepdims.params = { dims; keepdim }; x })

let max_pool2d params x = op1 (Op.Max_pool2d { Pool.MaxPool2d.params; x })

(* Same "two outputs, no [~fmt] override" shape as
   [adaptive_max_pool2d_with_indices] above. *)
let max_pool2d_with_indices params x =
  opN (Op.Max_pool2d_with_indices { Pool.MaxPool2dWithIndices.params; x })

let mean_keepdims ?(keepdim = true) dims x =
  op1 (Op.Mean_keepdims { Ops4.Mean_keepdims.params = { dims; keepdim }; x })

(* Variadic in both directions, like [concat4]'s operands and [unbind]'s
   outputs at once -- [opN], not [op1]. *)
let meshgrid tensors = opN (Op.Meshgrid { Meshgrid.Meshgrid.tensors })

(* Same I64-only threading as [add]; see its comment. *)
let mul a b =
  let* s = get in
  let a_sig = Tensor_id.Map.find a s.tensors in
  let b_sig = Tensor_id.Map.find b s.tensors in
  match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
  | Payload.Fmt Payload.I64, Payload.Fmt Payload.I64 ->
      op1 ~fmt:a_sig.Tensor_sig.fmt (Op.Mul { Pointwise.Bin.a; b })
  | _ -> op1 (Op.Mul { Pointwise.Bin.a; b })

let mul_scalar scalar x = op1 (Op.Mul_scalar { Pointwise.Scalar_bin.x; scalar })
let pow scalar x = op1 (Op.Pow { Pointwise.Scalar_bin.x; scalar })

(* Takes the dialect's own [Ops4.Pad4.params], whose entries are keyed by
   [Axis4.t]: a pad naming T or D is not constructible through this API, the
   same rule [unbind] below follows. *)
let pad4 params x = op1 (Op.Pad4 { Ops4.Pad4.params; x })

(* I64-only fmt threading, the Native4D twin of [Graph_builder.reshape]/
   [permute]'s own fix: [Eval_direct4]'s generic pixel fallback
   ([Schedule.evaluate]/[Tensor.materialize]) allocates its result as F32
   unconditionally, so declaring a non-I64, non-F32 format here would make
   the declared [Tensor_sig] lie about what the fallback actually writes --
   see Native's own `Trim_permute`-adjacent regression this file's Native
   twin already learned from. Every other format keeps [op1]'s F32 default,
   matching what the fallback delivers. *)
let permute4 perm x =
  let* s = get in
  let sg = Tensor_id.Map.find x s.tensors in
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.I64 ->
      op1 ~fmt:sg.Tensor_sig.fmt (Op.Permute4 { Ops4.Permute4.perm; x })
  | _ -> op1 (Op.Permute4 { Ops4.Permute4.perm; x })

let relu x = op1 (Op.Relu { Pointwise.Relu.x })

let repeat4 repeats x =
  op1 (Op.Repeat4 { Ops4.Repeat4.params = { repeats }; x })

let repeat_interleave4 axis repeats x =
  op1
    (Op.RepeatInterleave4
       { Ops4.RepeatInterleave4.params = { axis; repeats }; x })

let reshape4 shape x =
  let* s = get in
  let sg = Tensor_id.Map.find x s.tensors in
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.I64 ->
      op1 ~fmt:sg.Tensor_sig.fmt
        (Op.Reshape4 { Ops4.Reshape4.params = { shape }; x })
  | _ -> op1 (Op.Reshape4 { Ops4.Reshape4.params = { shape }; x })

(* Takes the dialect's own [Ops4.Layer_norm.params], whose [dims] are
   [Axis4.t]: a normalization naming T or D is not constructible through this
   API. Both affine operands stay optional. *)
let layer_norm4 params ~x ?weight ?bias () =
  op1 (Op.Layer_norm { Ops4.Layer_norm.params; x; weight; bias })

let rms_norm params ~x ?weight () =
  op1 (Op.Rms_norm { Ops4.Rms_norm.params; x; weight })

let rpow_scalar scalar x =
  op1 (Op.Rpow_scalar { Pointwise.Scalar_bin.x; scalar })

let rsub_scalar params x =
  op1 (Op.Rsub_scalar { Pointwise.Rsub_scalar.params; x })

(* Reuses [Attention.Sdpa.t] unchanged, exactly as [rms_norm]/[layer_norm4]
   reuse Native's params -- the payload names no axis and carries no shape. *)
let sdpa params ~query ~key ~value ?mask () =
  op1 (Op.Sdpa { Attention.Sdpa.params; query; key; value; mask })

(* Takes the dialect's own [Ops4.Select4.params], whose axis is [Axis4.t]: a
   select naming T or D is not constructible through this API. [index] stays
   validated rather than typed, the same choice [slice4]'s bounds make. *)
let select4 params x = op1 (Op.Select4 { Ops4.Select4.params; x })

let select_scatter4 params ~self ~src =
  op1 (Op.Select_scatter4 { Ops4.Select_scatter4.params; self; src })

let sigmoid x = op1 (Op.Sigmoid { Pointwise.Sigmoid.x })
let silu x = op1 (Op.Silu { Pointwise.Silu.x })
let sin x = op1 (Op.Sin { Pointwise.Sin.x })

(* Takes the dialect's own [Ops4.Slice4.params], whose axis is [Axis4.t]: a
   slice naming T or D is not constructible through this API. The BOUNDS are
   still validated rather than typed -- canonical is a relation between three
   ints and an extent, which no type here carries. *)
let slice4 params x = op1 (Op.Slice4 { Ops4.Slice4.params; x })

(* Takes the dialect's own [Ops4.Softmax4.params], whose axis is [Axis4.t]: a
   softmax naming T or D is not constructible through this API. *)
let softmax4 params x = op1 (Op.Softmax4 { Ops4.Softmax4.params; x })

(* Takes [Axis4.t] and [sizes] together, so a split naming T or D is not
   constructible through this API -- [unbind]'s rule, extended to a caller-
   chosen arity. [sizes] itself stays validated rather than typed, the same
   choice [slice4]'s bounds make: "sums to the axis extent" is a relation
   between a list and an extent, which no type here carries. *)
let split_with_sizes4 axis sizes x =
  let* s = get in
  let sg = Tensor_id.Map.find x s.tensors in
  opN ~fmt:sg.Tensor_sig.fmt ?quant:sg.Tensor_sig.quant
    (Op.Split_with_sizes4 { Ops4.Split_with_sizes4.params = { axis; sizes }; x })

let sqrt x = op1 (Op.Sqrt { Pointwise.Sqrt.x })

(* Takes the dialect's own [Ops4.Stack4.params], whose axis is [Axis4.t]: a
   stack naming T or D is not constructible through this API, the same rule
   [concat4] above follows. *)
let stack4 params xs = op1 (Op.Stack4 { Ops4.Stack4.params; xs })

(* Same I64-only threading as [add]; see its comment. *)
let sub a b =
  let* s = get in
  let a_sig = Tensor_id.Map.find a s.tensors in
  let b_sig = Tensor_id.Map.find b s.tensors in
  match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
  | Payload.Fmt Payload.I64, Payload.Fmt Payload.I64 ->
      op1 ~fmt:a_sig.Tensor_sig.fmt (Op.Sub { Pointwise.Bin.a; b })
  | _ -> op1 (Op.Sub { Pointwise.Bin.a; b })

let sum_keepdims ?(keepdim = true) dims x =
  op1 (Op.Sum_keepdims { Ops4.Sum_keepdims.params = { dims; keepdim }; x })

(* [Long]'s output dtype is I64 regardless of the operand's own format (ATen's
   `.long()` always produces int64), so this threads unconditionally -- unlike
   [add]/[sub]/[mul]/[reshape4]/[permute4]'s operand-conditional threading,
   matching Native's own [Graph_builder.to_copy]. [Bool]'s output is genuine
   [Payload.Bool] storage too, unconditionally, now that
   [Eval_direct4]'s own [To_copy(Bool)] arm writes it -- matching
   Native's own [Graph_builder.to_copy] convention exactly. [Float] keeps
   [op1]'s F32 default: its output genuinely is F32. *)
let to_copy target x =
  match target with
  | Pointwise.To_copy.Long ->
      op1 ~fmt:Payload.(Fmt I64) (Op.To_copy { Pointwise.To_copy.target; x })
  | Pointwise.To_copy.Bool ->
      op1 ~fmt:Payload.(Fmt Bool) (Op.To_copy { Pointwise.To_copy.target; x })
  | Pointwise.To_copy.Float -> op1 (Op.To_copy { Pointwise.To_copy.target; x })

let transposed_conv2d params ~x ~weight ?bias () =
  op1 (Op.Transposed_conv2d { Ops4.Transposed_conv2d.params; x; weight; bias })

(* Takes [Axis4.t], so a graph naming T or D is not constructible through this
   API — the dialect's rule that invalid states are unrepresentable rather than
   validated. Returns every slice, in ordinal order. *)
let unbind axis x =
  let* s = get in
  let sg = Tensor_id.Map.find x s.tensors in
  opN ~fmt:sg.Tensor_sig.fmt ?quant:sg.Tensor_sig.quant
    (Op.Unbind { Ops4.Unbind.params = { axis }; x })

let upsample_bicubic2d params x =
  op1 (Op.Upsample_bicubic2d { Resize.Bicubic2d.params; x })

let upsample_bilinear2d params x =
  op1 (Op.Upsample_bilinear2d { Resize.Bilinear2d.params; x })

let upsample_nearest2d params x =
  op1 (Op.Upsample_nearest2d { Resize.Nearest2d.params; x })

let vector_norm_keepdims ?(keepdim = true) dims x =
  op1
    (Op.Vector_norm_keepdims
       { Ops4.Vector_norm_keepdims.params = { dims; keepdim }; x })

let arange4 params =
  op1 ~fmt:params.Ops4.Arange4.fmt (Op.Arange4 { Ops4.Arange4.params })

let zeros4 params =
  op1 ~fmt:params.Ops4.Zeros4.fmt (Op.Zeros4 { Ops4.Zeros4.params })

let eye4 params = op1 ~fmt:params.Ops4.Eye4.fmt (Op.Eye4 { Ops4.Eye4.params })

let build ?(dtype = f32) ~outputs (m : 'a t) =
  let s0 =
    {
      next_tid = 0;
      next_nid = 0;
      dtype;
      rev_nodes = [];
      rev_items = [];
      tensors = Tensor_id.Map.empty;
      rev_inputs = [];
      input_kinds = Tensor_id.Map.empty;
    }
  in
  match m s0 with
  | Error e, _ -> Error e
  | Ok a, s ->
      Ok
        {
          Graph.Graph.nodes = List.rev s.rev_nodes;
          root =
            {
              Graph_ir.Group.id = Graph_ir.Group_id.of_int 0;
              label = None;
              items = List.rev s.rev_items;
            };
          tensors = s.tensors;
          inputs = List.rev s.rev_inputs;
          input_kinds = s.input_kinds;
          outputs = outputs a;
        }
