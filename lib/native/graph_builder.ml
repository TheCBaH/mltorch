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
  dtype : Payload.packed_fmt;
  rev_nodes : node list;
  tensors : Tensor_sig.t Tensor_id.Map.t;
  rev_inputs : tensor_ref list;
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

let input ~shape ?name ?fmt ?quant () s =
  let tid_int = s.next_tid in
  let tid = Tensor_id.of_int tid_int in
  let fmt = Option.value fmt ~default:s.dtype in
  let name =
    match name with Some n -> n | None -> Printf.sprintf "input_%d" tid_int
  in
  let sg = Tensor_sig.create ~id:tid ~name ~shape ~fmt ?quant () in
  ( Ok tid,
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
  let sg = Tensor_sig.create ~id:tid ~name ~shape ~fmt:f32 () in
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

let mean ?name params x =
  op1 ?name ~kind:"mean" (Mean { Reduce.Mean.params; x })

let mul ?name a b = op1 ?name ~kind:"mul" (Mul { Pointwise.Bin.a; b })

let permute ?name perm x =
  op1 ?name ~kind:"permute" (Permute { Permute.Permute.perm; x })

let relu ?name x = op1 ?name ~kind:"relu" (Relu { Pointwise.Relu.x })

let rms_norm ?name params ~x ?weight () =
  op1 ?name ~kind:"rms_norm" (Rms_norm { Norm.RmsNorm.params; x; weight })

let subgraph ~name (body : Tensor_id.t list t) : graph t =
 fun s ->
  (* run [body] in a child accumulation that shares the id counters/dtype *)
  let child_start =
    { s with rev_nodes = []; rev_inputs = []; tensors = Tensor_id.Map.empty }
  in
  match body child_start with
  | Error e, _ -> (Error e, s)
  | Ok outputs, child_end ->
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
      ( Ok g,
        { s with next_tid = child_end.next_tid; next_nid = child_end.next_nid }
      )

let invoke ?names (g : graph) (args : tensor_ref list) : Tensor_id.t list t =
 fun s ->
  let names_arr =
    match names with Some ns -> Array.of_list ns | None -> [||]
  in
  let rec alloc i outs next_tid tensors acc =
    match outs with
    | [] -> Ok (List.rev acc, next_tid, tensors)
    | oid :: rest -> (
        match Tensor_id.Map.find_opt oid g.Graph.tensors with
        | None -> Core.fail (`Missing_subgraph_output_sig oid)
        | Some (sub_sig : Tensor_sig.t) ->
            let tid_int = next_tid in
            let tid = Tensor_id.of_int tid_int in
            let name =
              if i < Array.length names_arr then names_arr.(i)
              else Printf.sprintf "%s_%d" g.Graph.name tid_int
            in
            let sg =
              Tensor_sig.create ~id:tid ~name ~shape:sub_sig.shape ~fmt:f32 ()
            in
            alloc (i + 1) rest (tid_int + 1)
              (Tensor_id.Map.add tid sg tensors)
              (tid :: acc))
  in
  match alloc 0 g.Graph.outputs s.next_tid s.tensors [] with
  | Error e -> (Error e, s)
  | Ok (out_ids, next_tid, tensors) ->
      let nid = Node_id.of_int s.next_nid in
      let node =
        { Node.id = nid; op = Subgraph { graph = g; args }; outputs = out_ids }
      in
      ( Ok out_ids,
        {
          s with
          next_tid;
          next_nid = s.next_nid + 1;
          tensors;
          rev_nodes = node :: s.rev_nodes;
        } )

let build ?(dtype = f32) ~name ~outputs (m : 'a t) =
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
  match m s0 with
  | Error e, _ -> Error e
  | Ok a, s ->
      Ok
        {
          Graph.name;
          nodes = List.rev s.rev_nodes;
          tensors = s.tensors;
          inputs = List.rev s.rev_inputs;
          outputs = outputs a;
        }
