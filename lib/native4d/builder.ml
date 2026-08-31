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

(* Op constructors in global alphabetical order, as in [Graph_builder]. *)

let add a b = op1 (Op.Add { Pointwise.Bin.a; b })
let add_scalar scalar x = op1 (Op.Add_scalar { Pointwise.Scalar_bin.x; scalar })

let adaptive_avg_pool2d params x =
  op1 (Op.Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x })

let avg_pool2d params x = op1 (Op.Avg_pool2d { Pool.AvgPool2d.params; x })
let clamp params x = op1 (Op.Clamp { Pointwise.Clamp.params; x })

(* Takes the dialect's own [Ops4.Concat4.params], whose axis is [Axis4.t]: a
   concat naming T or D is not constructible through this API, the same rule
   [unbind] below follows. *)
let concat4 params xs = op1 (Op.Concat4 { Ops4.Concat4.params; xs })

let conv2d params ~x ~weight ?bias () =
  op1 (Op.Conv2d { Ops4.Conv_payload.params; x; weight; bias })

let depthwise_conv2d params ~x ~weight ?bias () =
  op1 (Op.Depthwise_conv2d { Ops4.Conv_payload.params; x; weight; bias })

let div a b = op1 (Op.Div { Pointwise.Bin.a; b })
let div_scalar scalar x = op1 (Op.Div_scalar { Pointwise.Scalar_bin.x; scalar })

let gelu (approximate : Pointwise.Gelu.approximate) x =
  op1 (Op.Gelu { Pointwise.Gelu.x; approximate })

let group_norm4 params ~x ?weight ?bias () =
  op1 (Op.Group_norm4 { Ops4.Group_norm4.params; x; weight; bias })

let grouped_conv2d params ~x ~weight ?bias () =
  op1 (Op.Grouped_conv2d { Ops4.Grouped_conv_payload.params; x; weight; bias })

let hardsigmoid x = op1 (Op.Hardsigmoid { Pointwise.Hardsigmoid.x })
let hardswish x = op1 (Op.Hardswish { Pointwise.Hardswish.x })
let hardtanh params x = op1 (Op.Hardtanh { Pointwise.Hardtanh.params; x })
let leaky_relu params x = op1 (Op.Leaky_relu { Pointwise.Leaky_relu.params; x })

let max_keepdims dims x =
  op1 (Op.Max_keepdims { Ops4.Max_keepdims.params = { dims }; x })

let max_pool2d params x = op1 (Op.Max_pool2d { Pool.MaxPool2d.params; x })

let mean_keepdims dims x =
  op1 (Op.Mean_keepdims { Ops4.Mean_keepdims.params = { dims }; x })

let mul a b = op1 (Op.Mul { Pointwise.Bin.a; b })
let mul_scalar scalar x = op1 (Op.Mul_scalar { Pointwise.Scalar_bin.x; scalar })
let pow scalar x = op1 (Op.Pow { Pointwise.Scalar_bin.x; scalar })

(* Takes the dialect's own [Ops4.Pad4.params], whose entries are keyed by
   [Axis4.t]: a pad naming T or D is not constructible through this API, the
   same rule [unbind] below follows. *)
let pad4 params x = op1 (Op.Pad4 { Ops4.Pad4.params; x })
let permute4 perm x = op1 (Op.Permute4 { Ops4.Permute4.perm; x })
let relu x = op1 (Op.Relu { Pointwise.Relu.x })
let reshape4 shape x = op1 (Op.Reshape4 { Ops4.Reshape4.params = { shape }; x })

(* Takes the dialect's own [Ops4.Layer_norm.params], whose [dims] are
   [Axis4.t]: a normalization naming T or D is not constructible through this
   API. Both affine operands stay optional. *)
let layer_norm4 params ~x ?weight ?bias () =
  op1 (Op.Layer_norm { Ops4.Layer_norm.params; x; weight; bias })

let rms_norm params ~x ?weight () =
  op1 (Op.Rms_norm { Ops4.Rms_norm.params; x; weight })

let sigmoid x = op1 (Op.Sigmoid { Pointwise.Sigmoid.x })
let silu x = op1 (Op.Silu { Pointwise.Silu.x })

(* Takes the dialect's own [Ops4.Slice4.params], whose axis is [Axis4.t]: a
   slice naming T or D is not constructible through this API. The BOUNDS are
   still validated rather than typed -- canonical is a relation between three
   ints and an extent, which no type here carries. *)
let slice4 params x = op1 (Op.Slice4 { Ops4.Slice4.params; x })

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
let sub a b = op1 (Op.Sub { Pointwise.Bin.a; b })

let sum_keepdims dims x =
  op1 (Op.Sum_keepdims { Ops4.Sum_keepdims.params = { dims }; x })

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

let upsample_bilinear2d params x =
  op1 (Op.Upsample_bilinear2d { Resize.Bilinear2d.params; x })

let upsample_nearest2d params x =
  op1 (Op.Upsample_nearest2d { Resize.Nearest2d.params; x })

let vector_norm_keepdims dims x =
  op1
    (Op.Vector_norm_keepdims { Ops4.Vector_norm_keepdims.params = { dims }; x })

let arange4 params =
  op1 ~fmt:params.Ops4.Arange4.fmt (Op.Arange4 { Ops4.Arange4.params })

let zeros4 params =
  op1 ~fmt:params.Ops4.Zeros4.fmt (Op.Zeros4 { Ops4.Zeros4.params })

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
