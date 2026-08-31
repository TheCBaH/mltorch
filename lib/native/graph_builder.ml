(* See graph_builder.mli. [state] carries the tree-wide id counters, the default
   element type, and the accumulators (built up reversed). The monad is this
   file's own: a result-and-state computation defined below, not a generic
   state monad instantiated here — an earlier version of this comment claimed
   the latter, and the [Core.Monad.State] it named is gone. Op-output edges are
   F32 (the compute domain); only [input] honours the chosen element type. *)

open Graph_ir

type error =
  [ Graph_shape.error | `Expected_single_output_shape of output_count ]

and output_count = { count : int }

type state = {
  next_tid : int;
  next_nid : int;
  next_gid : int;
  dtype : Payload.packed_fmt;
  rev_nodes : node list;
  rev_items : Group.item list;
  tensors : Tensor_sig.t Tensor_id.Map.t;
  rev_inputs : tensor_ref list;
  input_kinds : Input.kind Tensor_id.Map.t;
}

type 'a t = state -> ('a, error) Err.t * state

let pp_error ppf : [< error ] -> unit = function
  | #Graph_shape.error as e -> Graph_shape.pp_error ppf e
  | `Expected_single_output_shape { count } ->
      Format.fprintf ppf "expected a single output shape, got %d" count

let return x s = (Ok x, s)
let lift_result (r : ('a, [< error ]) Err.t) s = ((r :> ('a, error) Err.t), s)

let ( let* ) m f s =
  match m s with Ok x, s' -> f x s' | Error e, s' -> (Error e, s')

let ( let+ ) m f s =
  match m s with Ok x, s' -> (Ok (f x), s') | Error e, s' -> (Error e, s')

let get s = (Ok s, s)
let f32 = Payload.Fmt Payload.F32

let source ~kind ~shape ?name ?fmt ?quant () s =
  let tid_int = s.next_tid in
  let tid = Tensor_id.of_int tid_int in
  let fmt = Option.value fmt ~default:s.dtype in
  let sg =
    Tensor_sig.create ~id:tid
      ~name:(Option.value name ~default:"")
      ~shape ~fmt ?quant ()
  in
  ( Ok tid,
    {
      s with
      next_tid = tid_int + 1;
      tensors = Tensor_id.Map.add tid sg s.tensors;
      rev_inputs = tid :: s.rev_inputs;
      input_kinds = Tensor_id.Map.add tid kind s.input_kinds;
    } )

let input ~shape ?name ?fmt ?quant () =
  source ~kind:Input.Input ~shape ?name ?fmt ?quant ()

let constant ~shape ?name ?fmt ?quant () =
  source ~kind:Input.Constant ~shape ?name ?fmt ?quant ()

(* Allocate a fresh output edge. Arithmetic outputs use the default F32; view
   ops such as [unbind] explicitly retain their source storage format. *)
let new_edge ?name ?fmt ?quant ~kind:_ shape s =
  let tid_int = s.next_tid in
  let tid = Tensor_id.of_int tid_int in
  let sg =
    Tensor_sig.create ~id:tid
      ~name:(Option.value name ~default:"")
      ~shape
      ~fmt:(Option.value fmt ~default:f32)
      ?quant ()
  in
  ( Ok tid,
    {
      s with
      next_tid = tid_int + 1;
      tensors = Tensor_id.Map.add tid sg s.tensors;
    } )

let push_node op outputs s =
  let nid = Node_id.of_int s.next_nid in
  ( Ok (),
    {
      s with
      next_nid = s.next_nid + 1;
      rev_nodes = { Node.id = nid; op; outputs } :: s.rev_nodes;
      rev_items = Group.Node nid :: s.rev_items;
    } )

(* A single-output op: compute its output shape from the current edge metadata,
  allocate the output edge, append the node. *)
let op1 ?name ?fmt ~kind op : Tensor_id.t t =
  let* s = get in
  let* shapes =
    lift_result
      (Graph_shape.output_shape op ~sig_of:(fun r ->
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
  let* tid = new_edge ?name ?fmt ~kind shape in
  let* () = push_node op [ tid ] in
  return tid

(* The variable-arity form: allocate one edge per inferred shape and push one
   node holding all of them, in order. No arity constraint at all — for an op
   whose output count is part of its input signature ([Unbind]) there is no
   expected number to check against, and the ceiling that DOES bound it lives in
   the op's [output_shapes], which is the only place it can run before the shape
   list exists.

   [op1] keeps its loud singleton check rather than being defined through this:
   for a single-output op a shape list of any other length is a bug worth
   naming. [max_pool2d_with_indices] keeps its own body too — it names its two
   edges with different kinds, which this cannot express. *)
let opN ?name ?fmt ?quant ~kind op : Tensor_id.t list t =
  let* s = get in
  let* shapes =
    lift_result
      (Graph_shape.output_shape op ~sig_of:(fun r ->
           Tensor_id.Map.find_opt r s.tensors
           |> Err.of_option (`Missing_tensor_sig r)))
  in
  Tensor_id.check_room ~next:s.next_tid ~count:(List.length shapes);
  (* TAIL-RECURSIVE, with an accumulator, and that is not a style choice: the
     obvious [let* tid = … in let* ids = alloc rest in return (tid :: ids)]
     holds a monadic frame per output, and at a few thousand outputs it
     overflows node's stack — caught by [make jsoo.inline-runtest], which is the
     only gate that runs these suites under the tighter stack. *)
  let rec alloc acc = function
    | [] -> return (List.rev acc)
    | shape :: rest ->
        let* tid = new_edge ?name ?fmt ?quant ~kind shape in
        alloc (tid :: acc) rest
  in
  let* ids = alloc [] shapes in
  let* () = push_node op ids in
  return ids

(* Op constructors in global alphabetical order (see graph_ir.mli). The record
   payloads are built with their first label qualified, which disambiguates the
   op module each belongs to (the [node.Node.outputs] convention). *)
let add ?name a b = op1 ?name ~kind:"add" (Add { Pointwise.Bin.a; b })

(* Narrow every scalar op parameter to its f32-canonical value here, at the one
   entry point both the PT2 importer and hand-built graphs go through, rather
   than trusting each caller to remember. The engine's tensors are F32, so an
   unnarrowed float64 literal would compute in a precision the payload cannot
   hold and would disagree with its own JSON round-trip (the codec is
   [Json_util.f32_jsont], whose [f32_to_f32] this mirrors). *)
let f32_scalar = Json_util.f32_to_f32

let add_scalar ?name scalar x =
  op1 ?name ~kind:"add_scalar"
    (Add_scalar { Pointwise.Scalar_bin.x; scalar = f32_scalar scalar })

let adaptive_avg_pool2d ?name params x =
  op1 ?name ~kind:"adaptive_avg_pool2d"
    (Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params; x })

let amax ?name params x =
  op1 ?name ~kind:"amax" (Amax { Reduce.Amax.params; x })

let avg_pool2d ?name params x =
  op1 ?name ~kind:"avg_pool2d" (Avg_pool2d { Pool.AvgPool2d.params; x })

let batch_norm ?name params ~x ?weight ?bias ~running_mean ~running_var () =
  op1 ?name ~kind:"batch_norm"
    (Batch_norm
       { Norm.BatchNorm.params; x; weight; bias; running_mean; running_var })

let batch_norm_no_stats ?name params ~x ?weight ?bias () =
  opN ?name ~kind:"batch_norm_no_stats"
    (Batch_norm_no_stats { Norm.BatchNormNoStats.params; x; weight; bias })

let batched_matmul ?name input mat2 =
  op1 ?name ~kind:"batched_matmul"
    (Batched_matmul { Matmul.Batched_matmul.input; mat2 })

let bmm ?name input mat2 =
  op1 ?name ~kind:"bmm" (Bmm { Matmul.Bmm.input; mat2 })

let clamp ?name (params : Pointwise.Clamp.params) x =
  op1 ?name ~kind:"clamp"
    (Clamp
       {
         Pointwise.Clamp.params =
           {
             min = Option.map f32_scalar params.min;
             max = Option.map f32_scalar params.max;
           };
         x;
       })

let clone ?name x = op1 ?name ~kind:"clone" (Clone { Pointwise.Clone.x })

let concat ?name params xs =
  op1 ?name ~kind:"concat" (Concat { Concat.Concat.params; xs })

let conv2d ?name params ~x ~weight ?bias () =
  op1 ?name ~kind:"conv2d" (Conv2d { Conv.Conv2d.params; x; weight; bias })

let conv2d_padding ?name params ~x ~weight ?bias () =
  op1 ?name ~kind:"conv2d_padding"
    (Conv2d_padding { Conv.Conv2d_padding.params; x; weight; bias })

let convolution ?name params ~x ~weight ?bias () =
  op1 ?name ~kind:"convolution"
    (Convolution { Conv.Convolution.params; x; weight; bias })

(* A sink for a dead edge: appends a [Discard] node with no output. *)
let discard x = push_node (Discard { x }) []

let expand ?name params x =
  op1 ?name ~kind:"expand" (Expand { Pointwise.Expand.params; x })

let gelu ?name (approximate : Pointwise.Gelu.approximate) x =
  op1 ?name ~kind:"gelu" (Gelu { Pointwise.Gelu.x; approximate })

let group_norm ?name params ~x ?weight ?bias () =
  op1 ?name ~kind:"group_norm"
    (Group_norm { Norm.GroupNorm.params; x; weight; bias })

let hardsigmoid ?name x =
  op1 ?name ~kind:"hardsigmoid" (Hardsigmoid { Pointwise.Hardsigmoid.x })

let hardswish ?name x =
  op1 ?name ~kind:"hardswish" (Hardswish { Pointwise.Hardswish.x })

let hardtanh ?name (params : Pointwise.Hardtanh.params) x =
  op1 ?name ~kind:"hardtanh"
    (Hardtanh
       {
         Pointwise.Hardtanh.params =
           {
             min_val = f32_scalar params.min_val;
             max_val = f32_scalar params.max_val;
           };
         x;
       })

(* Plain [op1], the same choice [slice]/[select] make: the output dtype
   defaults to F32 rather than preserving [self]'s, a pre-existing
   [Graph_builder.op1] characteristic (round 6 of the design record) that is
   correct for the one evidenced occurrence ([self] is ordinary F32 data;
   [index] is the separate Long operand supplying positions, never itself
   becoming the output) and not in scope to fix generally here. *)
let index_tensor ?name params ~self ~index =
  op1 ?name ~kind:"index_tensor"
    (Index_tensor { Index_tensor.Index_tensor.params; self; index })

let layer_norm ?name params ~x ?weight ?bias () =
  op1 ?name ~kind:"layer_norm"
    (Layer_norm { Norm.LayerNorm.params; x; weight; bias })

let leaky_relu ?name params x =
  op1 ?name ~kind:"leaky_relu" (Leaky_relu { Pointwise.Leaky_relu.params; x })

let linear ?name params ~x ~weight ?bias () =
  op1 ?name ~kind:"linear" (Linear { Linear.Linear.params; x; weight; bias })

let max_pool2d ?name params x =
  op1 ?name ~kind:"max_pool2d" (Max_pool2d { Pool.MaxPool2d.params; x })

(* Two outputs (values, indices): allocate an edge per output shape, append the
   node with both, and return them as a pair. *)
let max_pool2d_with_indices ?name params x =
  let op = Max_pool2d_with_indices { Pool.MaxPool2dWithIndices.params; x } in
  let* s = get in
  let* shapes =
    lift_result
      (Graph_shape.output_shape op ~sig_of:(fun r ->
           Tensor_id.Map.find_opt r s.tensors
           |> Err.of_option (`Missing_tensor_sig r)))
  in
  match shapes with
  | [ vshape; ishape ] ->
      let* vid = new_edge ?name ~kind:"max_pool2d_with_indices" vshape in
      let* iid = new_edge ~kind:"max_pool2d_with_indices_idx" ishape in
      let* () = push_node op [ vid; iid ] in
      return (vid, iid)
  | _ ->
      fun s ->
        ( Err.fail (`Expected_single_output_shape { count = List.length shapes }),
          s )

let div ?name a b = op1 ?name ~kind:"div" (Div { Pointwise.Bin.a; b })

let div_scalar ?name scalar x =
  op1 ?name ~kind:"div_scalar"
    (Div_scalar { Pointwise.Scalar_bin.x; scalar = f32_scalar scalar })

let mean ?name params x =
  op1 ?name ~kind:"mean" (Mean { Reduce.Mean.params; x })

let mul ?name a b = op1 ?name ~kind:"mul" (Mul { Pointwise.Bin.a; b })

let mul_scalar ?name scalar x =
  op1 ?name ~kind:"mul_scalar"
    (Mul_scalar { Pointwise.Scalar_bin.x; scalar = f32_scalar scalar })

(* The fill is narrowed to f32 HERE, at the one point every construction path
   goes through, exactly as [add_scalar]'s scalar is: an unnarrowed float64
   literal would compute in a precision the payload cannot store. *)
let pad ?name (params : Pad.Pad.params) x =
  let params =
    match params.Pad.Pad.mode with
    | Pad.Pad.Reflect -> params
    | Pad.Pad.Constant v ->
        { params with Pad.Pad.mode = Pad.Pad.Constant (f32_scalar v) }
  in
  op1 ?name ~kind:"pad" (Pad { Pad.Pad.params; x })

let permute ?name perm x =
  op1 ?name ~kind:"permute" (Permute { Permute.Permute.perm; x })

let pow ?name scalar x =
  op1 ?name ~kind:"pow"
    (Pow { Pointwise.Scalar_bin.x; scalar = f32_scalar scalar })

let relu ?name x = op1 ?name ~kind:"relu" (Relu { Pointwise.Relu.x })

let repeat ?name params x =
  op1 ?name ~kind:"repeat" (Repeat { Repeat.Repeat.params; x })

let repeat_interleave ?name params x =
  op1 ?name ~kind:"repeat_interleave"
    (RepeatInterleave { Repeat.RepeatInterleave.params; x })

let reshape ?name params x =
  op1 ?name ~kind:"reshape" (Reshape { Reshape.Reshape.params; x })

let rms_norm ?name params ~x ?weight () =
  op1 ?name ~kind:"rms_norm" (Rms_norm { Norm.RmsNorm.params; x; weight })

let sdpa ?name params ~query ~key ~value ?mask () =
  op1 ?name ~kind:"sdpa"
    (Sdpa { Attention.Sdpa.params; query; key; value; mask })

let select ?name params x =
  op1 ?name ~kind:"select" (Select { Split.Select.params; x })

let select_scatter ?name params ~self ~src =
  op1 ?name ~kind:"select_scatter"
    (Select_scatter { Split.Select_scatter.params; self; src })

let sigmoid ?name x =
  op1 ?name ~kind:"sigmoid" (Sigmoid { Pointwise.Sigmoid.x })

let silu ?name x = op1 ?name ~kind:"silu" (Silu { Pointwise.Silu.x })

let softmax ?name params x =
  op1 ?name ~kind:"softmax" (Softmax { Reduce.Softmax.params; x })

let sqrt ?name x = op1 ?name ~kind:"sqrt" (Sqrt { Pointwise.Sqrt.x })

let slice ?name params x =
  op1 ?name ~kind:"slice" (Slice { Split.Slice.params; x })

(* Same shape as [unbind]: the output count comes from [params.sizes] via
   [Graph_shape], and every piece carries the input's own dtype/quant. *)
let split_with_sizes ?name params x =
  let* s = get in
  let sg = Tensor_id.Map.find x s.tensors in
  opN ?name ~fmt:sg.Tensor_sig.fmt ?quant:sg.Tensor_sig.quant
    ~kind:"split_with_sizes"
    (Split_with_sizes { Split.Split_with_sizes.params; x })

let stack ?name params xs =
  op1 ?name ~kind:"stack" (Stack { Concat.Stack.params; xs })

let sub ?name a b = op1 ?name ~kind:"sub" (Sub { Pointwise.Bin.a; b })
let sum ?name params x = op1 ?name ~kind:"sum" (Sum { Reduce.Sum.params; x })

(* Returns every slice, in ordinal order. The count comes from the input
   signature via [Graph_shape], never from the caller — which is what lets a
   serialized graph's SSA name list be CHECKED against the node's arity instead
   of silently zipped against it. *)
let unbind ?name params x =
  let* s = get in
  let sg = Tensor_id.Map.find x s.tensors in
  opN ?name ~fmt:sg.Tensor_sig.fmt ?quant:sg.Tensor_sig.quant ~kind:"unbind"
    (Unbind { Split.Unbind.params; x })

let upsample_bilinear2d ?name params x =
  op1 ?name ~kind:"upsample_bilinear2d"
    (Upsample_bilinear2d { Resize.Bilinear2d.params; x })

let upsample_nearest2d ?name params x =
  op1 ?name ~kind:"upsample_nearest2d"
    (Upsample_nearest2d { Resize.Nearest2d.params; x })

let vector_norm ?name params x =
  op1 ?name ~kind:"vector_norm" (Vector_norm { Reduce.Vector_norm.params; x })

let arange ?name params =
  op1 ?name ~fmt:params.Factory.Arange.fmt ~kind:"arange"
    (Arange { Factory.Arange.params })

let zeros ?name params =
  op1 ?name ~fmt:params.Factory.Zeros.fmt ~kind:"zeros"
    (Zeros { Factory.Zeros.params })

let group ?label (body : 'a t) : 'a t =
 fun s ->
  (* A group changes only structural ownership.  Its child shares every global
     SSA accumulator and counter with the parent. *)
  let child_start = { s with next_gid = s.next_gid + 1; rev_items = [] } in
  match body child_start with
  | Error e, _ -> (Error e, s)
  | Ok value, child_end ->
      let group =
        Group.
          {
            id = Group_id.of_int s.next_gid;
            label;
            items = List.rev child_end.rev_items;
          }
      in
      (Ok value, { child_end with rev_items = Group.Group group :: s.rev_items })

let build ?(dtype = f32) ~name:_ ~outputs (m : 'a t) =
  let s0 =
    {
      next_tid = 0;
      next_nid = 0;
      next_gid = 1;
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
        Graph.
          {
            nodes = List.rev s.rev_nodes;
            root =
              Group.
                {
                  id = Group_id.of_int 0;
                  label = None;
                  items = List.rev s.rev_items;
                };
            tensors = s.tensors;
            inputs = List.rev s.rev_inputs;
            input_kinds = s.input_kinds;
            outputs = outputs a;
          }
