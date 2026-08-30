(* Serialized ExportedProgram fixtures, built as JSON and decoded through the
   generated schema. Shared by the lowering tests in this directory: they need
   node and graph shapes no exporter emits, and hand-building the OCaml records
   would be far longer than the JSON. *)

let jstr fmt = Printf.sprintf fmt

let tensor_meta sizes =
  jstr
    {|{"dtype":7,"sizes":[%s],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    (String.concat "," (List.map (fun i -> jstr {|{"as_int":%d}|} i) sizes))

let as_tensor n = jstr {|{"as_tensor":{"name":"%s"}}|} n

let as_tensors ns =
  jstr {|{"as_tensors":[%s]}|}
    (String.concat "," (List.map (fun n -> jstr {|{"name":"%s"}|} n) ns))

let unbind_node ~dim ~outs =
  jstr
    {|{"target":"torch.ops.aten.unbind.int","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"dim","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") dim (as_tensors outs)

(* [tensor_values] carries the input, plus whatever [extra_tensor_values] adds.
   Node OUTPUT metadata is usually omitted: nothing reads it on the paths under
   test, and listing thousands of entries would make a fixture larger than the
   property it demonstrates. It is supplied where a test needs a node's DECLARED
   shape to differ from the one Native infers. *)
(* [params] are CAPTURED tensors -- weights and biases -- which reach the env as
   [InputSpec.Parameter] rather than [User_input]. An op that reads one needs it
   in both places or the arm fails at [env_find] with [`Undefined_ssa] before it
   reaches whatever the fixture was written to check. *)
let program ?(extra_tensor_values = []) ?(params = []) ~x_sizes ~nodes
    ~graph_outputs () =
  let tensor_values =
    String.concat ","
      (jstr {|"x":%s|} (tensor_meta x_sizes)
      :: List.map (fun (n, m) -> jstr {|"%s":%s|} n m) extra_tensor_values)
  in
  let inputs = String.concat "," (as_tensor "x" :: List.map as_tensor params) in
  let input_specs =
    String.concat ","
      (jstr {|{"user_input":{"arg":%s}}|} (as_tensor "x")
      :: List.map
           (fun n ->
             jstr {|{"parameter":{"arg":{"name":"%s"},"parameter_name":"%s"}}|}
               n n)
           params)
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[%s],"output_specs":[%s]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    inputs
    (String.concat "," graph_outputs)
    (String.concat "," nodes) tensor_values input_specs
    (String.concat ","
       (List.map (fun o -> jstr {|{"user_output":{"arg":%s}}|} o) graph_outputs))

let slice_names n = List.init n (fun i -> jstr "u%d" i)

(* A fixture that will not decode is a broken test, not a test result. *)
let lower json =
  match
    Jsont_bytesrw.decode_string Pytorch_types.ExportedProgram.jsont json
  with
  | Error e -> failwith ("fixture did not decode: " ^ e)
  | Ok prog -> Native_interp.lower prog

let show label json =
  Format.printf "%-26s %a@." label
    (Core.Pretty.err_result
       ~ok:(fun ppf (l : Pt2_native_graph.t) ->
         Format.fprintf ppf "lowered, nodes=%d"
           (List.length l.Pt2_native_graph.graph.Graph_ir.Graph.nodes))
       ~error:Native_interp.pp_error)
    (lower json)

let unbind_program count =
  program ~x_sizes:[ count; 2 ]
    ~nodes:[ unbind_node ~dim:0 ~outs:(slice_names count) ]
    ~graph_outputs:[ as_tensor "u0" ]
    ()

let split_tensor_node ~split_size ~outs =
  jstr
    {|{"target":"torch.ops.aten.split.Tensor","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"split_size","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") split_size (as_tensors outs)
