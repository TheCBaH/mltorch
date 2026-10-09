(* The empty-cache normalization: a lifted [0]-shaped Tensor_constant, cloned
   and concatenated with a step's new keys, is dropped (ATen's cat skips a 1-D
   size-0 operand: test/aten_ops_test.ml). Everything else about a zero-length
   tensor stays refused. *)

open Programs

let meta ?(dtype = 7) sizes =
  jstr
    {|{"dtype":%d,"sizes":[%s],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    dtype
    (String.concat "," (List.map (fun i -> jstr {|{"as_int":%d}|} i) sizes))

let node target inputs outputs =
  jstr {|{"target":"%s","inputs":[%s],"outputs":[%s],"metadata":{}}|} target
    (String.concat "," inputs)
    (String.concat "," (List.map as_tensor outputs))

let in_tensor name n =
  jstr {|{"name":"%s","arg":%s,"kind":1}|} name (as_tensor n)

let in_tensors name ns =
  jstr {|{"name":"%s","arg":%s,"kind":1}|} name (as_tensors ns)

let in_int name i = jstr {|{"name":"%s","arg":{"as_int":%d},"kind":1}|} name i

let clone out src =
  node "torch.ops.aten.clone.default" [ in_tensor "self" src ] [ out ]

let cat out srcs dim =
  node "torch.ops.aten.cat.default"
    [ in_tensors "tensors" srcs; in_int "dim" dim ]
    [ out ]

(* [x] is the [2,3] user input; each [(name, meta)] in [constants] is a lifted
   Tensor_constant. [mutating] adds a buffer-mutation output spec. *)
let prog ?(mutating = false) ?(constants = [ ("c", meta [ 0 ]) ]) ~values ~nodes
    ~outputs () =
  let tensor_values =
    String.concat ","
      (List.map
         (fun (n, m) -> jstr {|"%s":%s|} n m)
         ((("x", meta [ 2; 3 ]) :: constants) @ values))
  in
  let inputs =
    String.concat "," (List.map as_tensor ("x" :: List.map fst constants))
  in
  let input_specs =
    String.concat ","
      (jstr {|{"user_input":{"arg":%s}}|} (as_tensor "x")
      :: List.map
           (fun (n, _) ->
             jstr
               {|{"tensor_constant":{"arg":{"name":"%s"},"tensor_constant_name":"%s"}}|}
               n n)
           constants)
  in
  let output_specs =
    String.concat ","
      (List.map
         (fun o -> jstr {|{"user_output":{"arg":%s}}|} (as_tensor o))
         outputs
      @
      if mutating then
        [ {|{"buffer_mutation":{"arg":{"name":"x"},"buffer_name":"b"}}|} ]
      else [])
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[%s],"output_specs":[%s]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    inputs
    (String.concat "," (List.map as_tensor outputs))
    (String.concat "," nodes) tensor_values input_specs output_specs

let decode json =
  match
    Jsont_bytesrw.decode_string Pytorch_types.ExportedProgram.jsont json
  with
  | Ok p -> p
  | Error e -> failwith ("fixture did not decode: " ^ e)

let pp_report ppf (r : Native_interp.Empty_cache_report.t) =
  List.iter
    (fun (s : Native_interp.Empty_cache_report.source) ->
      Format.fprintf ppf "  source %s: clones=[%s] cats=[%s]@." s.ssa
        (String.concat "; " s.clones)
        (String.concat "; "
           (List.map
              (fun (c : Native_interp.Empty_cache_report.cat) ->
                Printf.sprintf "%s drops %s" c.cat
                  (String.concat ","
                     (List.map
                        (fun (p : Native_interp.Empty_cache_report.Operand.t) ->
                          string_of_int (p :> int))
                        c.removed)))
              s.cats)))
    r.sources

let lowered_nodes ppf program =
  match Native_interp.lower program with
  | Error e ->
      Format.fprintf ppf "  lower: %a@." Native_interp.pp_error
        (Err.Error.kind e)
  | Ok l ->
      Format.fprintf ppf "  lower: %a@." Graph_ir.pp l.Pt2_native_graph.graph

let show label json =
  let program = decode json in
  Format.printf "%s@." label;
  Format.printf "  strict lower, before:@.";
  lowered_nodes Format.std_formatter program;
  match Native_interp.normalize_empty_caches program with
  | Error e ->
      Format.printf "  refused: %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok (program, report) ->
      pp_report Format.std_formatter report;
      Format.printf "  after:@.";
      lowered_nodes Format.std_formatter program

let%expect_test "clone + cat of the empty cache becomes a clone of the keys" =
  show "one operand left"
    (prog
       ~values:[ ("k", meta []); ("y", meta [ 2; 3 ]) ]
       ~nodes:[ clone "k" "c"; cat "y" [ "k"; "x" ] 0 ]
       ~outputs:[ "y" ] ());
  [%expect
    {|
    one operand left
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      source c: clones=[k] cats=[y drops 0]
      after:
      lower: graph
             inputs: [t0 f32 [W=2 C=3] ->[n0]]
             nodes:
               n0: [t1 f32 [W=2 C=3]] = clone x=t0
             outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let%expect_test "several operands keep their order" =
  show "cat [k, x, k2?, x]"
    (prog
       ~values:[ ("k", meta []); ("y", meta [ 4; 3 ]) ]
       ~nodes:[ clone "k" "c"; cat "y" [ "x"; "k"; "x" ] (-2) ]
       ~outputs:[ "y" ] ());
  show "two sources, one cat"
    (prog
       ~constants:[ ("c", meta [ 0 ]); ("d", meta [ 0 ]) ]
       ~values:[ ("k", meta []); ("l", meta []); ("y", meta [ 2; 3 ]) ]
       ~nodes:[ clone "k" "c"; clone "l" "d"; cat "y" [ "k"; "x"; "l" ] 0 ]
       ~outputs:[ "y" ] ());
  show "a clone chain"
    (prog
       ~values:[ ("k", meta []); ("k2", meta []); ("y", meta [ 2; 3 ]) ]
       ~nodes:[ clone "k" "c"; clone "k2" "k"; cat "y" [ "k2"; "x" ] 0 ]
       ~outputs:[ "y" ] ());
  show "an unread source is dropped"
    (prog
       ~values:[ ("y", meta [ 2; 3 ]) ]
       ~nodes:[ clone "y" "x" ]
       ~outputs:[ "y" ] ());
  [%expect
    {|
    cat [k, x, k2?, x]
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      source c: clones=[k] cats=[y drops 1]
      after:
      lower: graph
             inputs: [t0 f32 [W=2 C=3] ->[n0]]
             nodes:
               n0: [t1 f32 [W=4 C=3]] = concat xs=[t0, t0] params={axis=W}
             outputs: [t1 f32 [W=4 C=3] <-n0]
    two sources, one cat
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      source c: clones=[k] cats=[y drops 0]
      source d: clones=[l] cats=[y drops 2]
      after:
      lower: graph
             inputs: [t0 f32 [W=2 C=3] ->[n0]]
             nodes:
               n0: [t1 f32 [W=2 C=3]] = clone x=t0
             outputs: [t1 f32 [W=2 C=3] <-n0]
    a clone chain
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      source c: clones=[k; k2] cats=[y drops 0]
      after:
      lower: graph
             inputs: [t0 f32 [W=2 C=3] ->[n0]]
             nodes:
               n0: [t1 f32 [W=2 C=3]] = clone x=t0
             outputs: [t1 f32 [W=2 C=3] <-n0]
    an unread source is dropped
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      source c: clones=[] cats=[]
      after:
      lower: graph
             inputs: [t0 f32 [W=2 C=3] ->[n0]]
             nodes:
               n0: [t1 f32 [W=2 C=3]] = clone x=t0
             outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let%expect_test "every other use stays refused" =
  show "relu of the empty"
    (prog
       ~values:[ ("y", meta [ 0 ]) ]
       ~nodes:
         [ node "torch.ops.aten.relu.default" [ in_tensor "self" "c" ] [ "y" ] ]
       ~outputs:[ "y" ] ());
  show "returned directly"
    (prog
       ~values:[ ("k", meta []) ]
       ~nodes:[ clone "k" "c" ]
       ~outputs:[ "k" ] ());
  show "all operands empty"
    (prog
       ~values:[ ("k", meta []); ("y", meta [ 0 ]) ]
       ~nodes:[ clone "k" "c"; cat "y" [ "k"; "k" ] 0 ]
       ~outputs:[ "y" ] ());
  show "dtype differs from the keys"
    (prog
       ~constants:[ ("c", meta ~dtype:5 [ 0 ]) ]
       ~values:[ ("k", meta ~dtype:5 []); ("y", meta [ 2; 3 ]) ]
       ~nodes:[ clone "k" "c"; cat "y" [ "k"; "x" ] 0 ]
       ~outputs:[ "y" ] ());
  show "mutating signature"
    (prog ~mutating:true
       ~values:[ ("k", meta []); ("y", meta [ 2; 3 ]) ]
       ~nodes:[ clone "k" "c"; cat "y" [ "k"; "x" ] 0 ]
       ~outputs:[ "y" ] ());
  show "clone read by a non-cat, non-clone"
    (prog
       ~values:[ ("k", meta []); ("y", meta [ 0 ]) ]
       ~nodes:
         [
           clone "k" "c";
           node "torch.ops.aten.relu.default" [ in_tensor "self" "k" ] [ "y" ];
         ]
       ~outputs:[ "y" ] ());
  [%expect
    {|
    relu of the empty
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      refused: empty cache: c is read by torch.ops.aten.relu.default, not a clone or a cat
    returned directly
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      refused: empty cache: k is returned by the graph
    all operands empty
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      refused: empty cache: every operand of the cat producing y is empty
    dtype differs from the keys
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      refused: empty cache: k and x differ in dtype, so dropping k is not value-preserving
    mutating signature
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      refused: empty cache: the signature mutates state
    clone read by a non-cat, non-clone
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      refused: empty cache: k is read by torch.ops.aten.relu.default, not a clone or a cat |}]

let%expect_test "a zero-length that is not a [0] constant is left alone" =
  show "[0, 3] constant"
    (prog
       ~constants:[ ("c", meta [ 0; 3 ]) ]
       ~values:[ ("y", meta [ 2; 3 ]) ]
       ~nodes:[ clone "y" "x" ]
       ~outputs:[ "y" ] ());
  show "no empties at all"
    (prog ~constants:[]
       ~values:[ ("y", meta [ 2; 3 ]) ]
       ~nodes:[ clone "y" "x" ]
       ~outputs:[ "y" ] ());
  [%expect
    {|
    [0, 3] constant
      strict lower, before:
      lower: malformed PT2 graph: c has a zero-length dimension
      after:
      lower: malformed PT2 graph: c has a zero-length dimension
    no empties at all
      strict lower, before:
      lower: graph
             inputs: [t0 f32 [W=2 C=3] ->[n0]]
             nodes:
               n0: [t1 f32 [W=2 C=3]] = clone x=t0
             outputs: [t1 f32 [W=2 C=3] <-n0]
      after:
      lower: graph
             inputs: [t0 f32 [W=2 C=3] ->[n0]]
             nodes:
               n0: [t1 f32 [W=2 C=3]] = clone x=t0
             outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* --- end to end: the same program through [run_named] --- *)

let archive program =
  let none =
    {
      Pytorch_weights_config.ModelWeightsConfig.config =
        Schema_runtime.String_map.empty;
    }
  in
  Pt2_archive.of_parts ~program ~weights:none ~constants:none ~load:(fun _ ->
      Error "no payload")

let x_tensor =
  let values = [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let data = Bytes.create (4 * List.length values) in
  List.iteri
    (fun i v -> Bytes.set_int32_le data (4 * i) (Int32.bits_of_float v))
    values;
  {
    Pt2_tensor.dtype = Pt2_dtype.Float32;
    sizes = [ 2; 3 ];
    strides = [ 3; 1 ];
    storage_offset = 0;
    data = Pt2_storage.of_string (Bytes.to_string data);
  }

let floats packed =
  let (Tensor.Tensor t) = packed in
  let extent a = (Vec6.get t.Tensor.shape a :> int) in
  List.concat
    (List.init (extent Axis.W) (fun w ->
         List.init (extent Axis.C) (fun c ->
             Tensor.read packed (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w ~c))))

let run ?empty_caches json =
  match
    Native_interp.run_named ?empty_caches
      (archive (decode json))
      ~inputs:[ ("x", x_tensor) ]
  with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok outputs ->
      List.iter
        (fun o ->
          Format.printf "  %s@."
            (String.concat " " (List.map (Printf.sprintf "%g") (floats o))))
        outputs

let%expect_test "run_named executes the normalized graph, and only on request" =
  let json =
    prog
      ~values:[ ("k", meta []); ("y", meta [ 4; 3 ]) ]
      ~nodes:[ clone "k" "c"; cat "y" [ "x"; "k"; "x" ] 0 ]
      ~outputs:[ "y" ] ()
  in
  Format.printf "default (strict):@.";
  run json;
  Format.printf "with empty_caches:@.";
  run ~empty_caches:(fun r -> pp_report Format.std_formatter r) json;
  [%expect
    {|
    default (strict):
      malformed PT2 graph: c has a zero-length dimension
    with empty_caches:
      source c: clones=[k] cats=[y drops 1]
      1 2 3 4 5 6 1 2 3 4 5 6 |}]
