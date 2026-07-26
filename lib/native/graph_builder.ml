(* See graph_builder.mli. [state] carries the tree-wide id counters, the default
   element type, and the accumulators (built up reversed). The monad itself is the
   generic Core.Monad.State threaded over this concrete state. Op-output edges are
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

type 'a t = state -> ('a, error) Core.result * state

let pp_error ppf : [< error ] -> unit = function
  | #Graph_shape.error as e -> Graph_shape.pp_error ppf e
  | `Expected_single_output_shape { count } ->
      Format.fprintf ppf "expected a single output shape, got %d" count

let return x s = (Ok x, s)

let lift_result (r : ('a, [< error ]) Core.result) s =
  ((r :> ('a, error) Core.result), s)

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

(* Allocate one fresh F32 output edge with [shape] (default-named from [kind]). *)
let new_edge ?name ~kind:_ shape s =
  let tid_int = s.next_tid in
  let tid = Tensor_id.of_int tid_int in
  let sg =
    Tensor_sig.create ~id:tid
      ~name:(Option.value name ~default:"")
      ~shape ~fmt:f32 ()
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
let op1 ?name ~kind op : Tensor_id.t t =
  let* s = get in
  let* shapes =
    lift_result
      (Graph_shape.output_shape op ~sig_of:(fun r ->
           match Tensor_id.Map.find_opt r s.tensors with
           | Some sg -> Core.return sg
           | None -> Core.fail (`Missing_tensor_sig r)))
  in
  let* shape =
    match shapes with
    | [ sh ] -> return sh
    | _ ->
        fun s ->
          ( Core.fail
              (`Expected_single_output_shape { count = List.length shapes }),
            s )
  in
  let* tid = new_edge ?name ~kind shape in
  let* () = push_node op [ tid ] in
  return tid

(* Op constructors in global alphabetical order (see graph_ir.mli). The record
   payloads are built with their first label qualified, which disambiguates the
   op module each belongs to (the [node.Node.outputs] convention). *)
let add ?name a b = op1 ?name ~kind:"add" (Add { Pointwise.Bin.a; b })

let avg_pool2d ?name params x =
  op1 ?name ~kind:"avg_pool2d" (Avg_pool2d { Pool.AvgPool2d.params; x })

let batch_norm ?name params ~x ?weight ?bias ~running_mean ~running_var () =
  op1 ?name ~kind:"batch_norm"
    (Batch_norm
       { Norm.BatchNorm.params; x; weight; bias; running_mean; running_var })

let bmm ?name input mat2 =
  op1 ?name ~kind:"bmm" (Bmm { Matmul.Bmm.input; mat2 })

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
           match Tensor_id.Map.find_opt r s.tensors with
           | Some sg -> Core.return sg
           | None -> Core.fail (`Missing_tensor_sig r)))
  in
  match shapes with
  | [ vshape; ishape ] ->
      let* vid = new_edge ?name ~kind:"max_pool2d_with_indices" vshape in
      let* iid = new_edge ~kind:"max_pool2d_with_indices_idx" ishape in
      let* () = push_node op [ vid; iid ] in
      return (vid, iid)
  | _ ->
      fun s ->
        ( Core.fail
            (`Expected_single_output_shape { count = List.length shapes }),
          s )

let div ?name a b = op1 ?name ~kind:"div" (Div { Pointwise.Bin.a; b })

let mean ?name params x =
  op1 ?name ~kind:"mean" (Mean { Reduce.Mean.params; x })

let mul ?name a b = op1 ?name ~kind:"mul" (Mul { Pointwise.Bin.a; b })

let permute ?name perm x =
  op1 ?name ~kind:"permute" (Permute { Permute.Permute.perm; x })

let relu ?name x = op1 ?name ~kind:"relu" (Relu { Pointwise.Relu.x })

let reshape ?name params x =
  op1 ?name ~kind:"reshape" (Reshape { Reshape.Reshape.params; x })

let rms_norm ?name params ~x ?weight () =
  op1 ?name ~kind:"rms_norm" (Rms_norm { Norm.RmsNorm.params; x; weight })

let sqrt ?name x = op1 ?name ~kind:"sqrt" (Sqrt { Pointwise.Sqrt.x })
let sub ?name a b = op1 ?name ~kind:"sub" (Sub { Pointwise.Bin.a; b })

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
