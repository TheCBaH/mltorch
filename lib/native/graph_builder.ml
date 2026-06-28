(* See graph_builder.mli. [state] carries the tree-wide id counters, the default
   element type, and the accumulators (built up reversed). The monad itself is the
   generic Core.Monad.State threaded over this concrete state. Op-output edges are
   F32 (the compute domain); only [input] honours the chosen element type. *)

open Graph_ir

type state = {
  next_tid : int;
  next_nid : int;
  dtype : Payload.packed_fmt;
  rev_nodes : node list;
  tensors : Tensor_sig.t Tensor_id.Map.t;
  rev_inputs : tensor_ref list;
}

type 'a t = (state, 'a) Core.Monad.State.t

let return = Core.Monad.State.return
let ( let* ) = Core.Monad.State.( let* )
let ( let+ ) = Core.Monad.State.( let+ )
let get = Core.Monad.State.get
let f32 = Payload.Fmt Payload.F32

let input ~shape ?name ?fmt ?quant () s =
  let tid_int = s.next_tid in
  let tid = Tensor_id.of_int tid_int in
  let fmt = Option.value fmt ~default:s.dtype in
  let name =
    match name with Some n -> n | None -> Printf.sprintf "input_%d" tid_int
  in
  let sg = Tensor_sig.create ~name ~shape ~fmt ?quant () in
  ( tid,
    {
      s with
      next_tid = tid_int + 1;
      tensors = Tensor_id.Map.add tid sg s.tensors;
      rev_inputs = tid :: s.rev_inputs;
    } )

(* Allocate one fresh F32 output edge with [shape] (default-named from [kind]). *)
let new_edge ?name ~kind shape s =
  let tid_int = s.next_tid in
  let tid = Tensor_id.of_int tid_int in
  let name =
    match name with Some n -> n | None -> Printf.sprintf "%s_%d" kind tid_int
  in
  let sg = Tensor_sig.create ~name ~shape ~fmt:f32 () in
  ( tid,
    {
      s with
      next_tid = tid_int + 1;
      tensors = Tensor_id.Map.add tid sg s.tensors;
    } )

let push_node op outputs s =
  let nid = Node_id.of_int s.next_nid in
  ( (),
    {
      s with
      next_nid = s.next_nid + 1;
      rev_nodes = { Node.id = nid; op; outputs } :: s.rev_nodes;
    } )

(* A single-output op: compute its output shape from the current edge metadata,
   allocate the output edge, append the node. *)
let op1 ?name ~kind op : Tensor_id.t t =
  let* s = get in
  let shape =
    match
      Graph_shape.output_shape op ~sig_of:(fun r ->
          Tensor_id.Map.find r s.tensors)
    with
    | [ sh ] -> sh
    | _ -> invalid_arg "Graph_builder.op1: expected a single-output op"
  in
  let* tid = new_edge ?name ~kind shape in
  let* () = push_node op [ tid ] in
  return tid

(* Op constructors in global alphabetical order (see graph_ir.mli). *)
let add ?name a b = op1 ?name ~kind:"add" (Add { a; b })

let avg_pool2d ?name params x =
  op1 ?name ~kind:"avg_pool2d" (Avg_pool2d { params; x })

let bmm ?name input mat2 = op1 ?name ~kind:"bmm" (Bmm { input; mat2 })

let conv2d ?name params ~x ~weight ?bias () =
  op1 ?name ~kind:"conv2d" (Conv2d { params; x; weight; bias })

let linear ?name params ~x ~weight ?bias () =
  op1 ?name ~kind:"linear" (Linear { params; x; weight; bias })

let max_pool2d ?name params x =
  op1 ?name ~kind:"max_pool2d" (Max_pool2d { params; x })

let mean ?name params x = op1 ?name ~kind:"mean" (Mean { params; x })
let mul ?name a b = op1 ?name ~kind:"mul" (Mul { a; b })
let permute ?name perm x = op1 ?name ~kind:"permute" (Permute { perm; x })
let relu ?name x = op1 ?name ~kind:"relu" (Relu { x })

let rms_norm ?name params ~x ?weight () =
  op1 ?name ~kind:"rms_norm" (Rms_norm { params; x; weight })

let subgraph ~name (body : Tensor_id.t list t) : graph t =
 fun s ->
  (* run [body] in a child accumulation that shares the id counters/dtype *)
  let child_start =
    { s with rev_nodes = []; rev_inputs = []; tensors = Tensor_id.Map.empty }
  in
  let outputs, child_end = body child_start in
  let g =
    {
      Graph.name;
      nodes = List.rev child_end.rev_nodes;
      tensors = child_end.tensors;
      inputs = List.rev child_end.rev_inputs;
      outputs;
    }
  in
  (* keep the advanced counters; restore the parent's own accumulators *)
  (g, { s with next_tid = child_end.next_tid; next_nid = child_end.next_nid })

let invoke ?names (g : graph) (args : tensor_ref list) : Tensor_id.t list t =
 fun s ->
  let names_arr =
    match names with Some ns -> Array.of_list ns | None -> [||]
  in
  let rec alloc i outs next_tid tensors acc =
    match outs with
    | [] -> (List.rev acc, next_tid, tensors)
    | oid :: rest ->
        let sub_sig : Tensor_sig.t = Tensor_id.Map.find oid g.Graph.tensors in
        let tid_int = next_tid in
        let tid = Tensor_id.of_int tid_int in
        let name =
          if i < Array.length names_arr then names_arr.(i)
          else Printf.sprintf "%s_%d" g.Graph.name tid_int
        in
        let sg = Tensor_sig.create ~name ~shape:sub_sig.shape ~fmt:f32 () in
        alloc (i + 1) rest (tid_int + 1)
          (Tensor_id.Map.add tid sg tensors)
          (tid :: acc)
  in
  let out_ids, next_tid, tensors =
    alloc 0 g.Graph.outputs s.next_tid s.tensors []
  in
  let nid = Node_id.of_int s.next_nid in
  let node =
    { Node.id = nid; op = Subgraph { graph = g; args }; outputs = out_ids }
  in
  ( out_ids,
    {
      s with
      next_tid;
      next_nid = s.next_nid + 1;
      tensors;
      rev_nodes = node :: s.rev_nodes;
    } )

let build ?(dtype = f32) ~name ~outputs (m : 'a t) : graph =
  let s0 =
    {
      next_tid = 0;
      next_nid = 0;
      dtype;
      rev_nodes = [];
      tensors = Tensor_id.Map.empty;
      rev_inputs = [];
    }
  in
  let a, s = m s0 in
  {
    Graph.name;
    nodes = List.rev s.rev_nodes;
    tensors = s.tensors;
    inputs = List.rev s.rev_inputs;
    outputs = outputs a;
  }
