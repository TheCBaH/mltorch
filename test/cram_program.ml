(* Hand-authored serialization witnesses, formerly inline Python setup. *)
open Cram_json

let scalar key value = obj [ (key, value) ]
let i n = scalar "as_int" (J.int n)
let f n = scalar "as_float" (J.number n)
let b n = scalar "as_bool" (J.bool n)
let ints ns = scalar "as_ints" (J.list (List.map J.int ns))
let s x = scalar "as_string" (J.string x)
let t name = scalar "as_tensor" (obj [ ("name", J.string name) ])

let arg name value =
  obj [ ("name", J.string name); ("arg", value); ("kind", J.int 1) ]

let args xs = List.map (fun (name, value) -> arg name value) xs

let metadata stack =
  if stack = "" then obj [] else obj [ ("nn_module_stack", J.string stack) ]

let node ?(stack = "") ?outputs target ins out =
  obj
    [
      ("target", J.string ("torch.ops.aten." ^ target));
      ("inputs", J.list (args ins));
      ("outputs", J.list (Option.value ~default:[ t out ] outputs));
      ("metadata", metadata stack);
    ]

let tm_sizes sizes =
  obj
    [
      ("dtype", J.int 7);
      ("sizes", J.list sizes);
      ("requires_grad", J.bool false);
      ("device", obj [ ("type", J.string "cpu") ]);
      ("strides", J.list [ i 1 ]);
      ("storage_offset", i 0);
      ("layout", J.int 7);
    ]

let tm sizes = tm_sizes (List.map i sizes)

let program ?(inputs = [ "x" ]) ?(params = []) ?(outputs = [ t "y" ]) ?specs
    nodes tv =
  let specs =
    Option.value
      ~default:
        (List.map (fun n -> obj [ ("user_input", obj [ ("arg", t n) ]) ]) inputs
        @ List.map
            (fun (n, name) ->
              obj
                [
                  ( "parameter",
                    obj
                      [
                        ("arg", obj [ ("name", J.string n) ]);
                        ("parameter_name", J.string name);
                      ] );
                ])
            params)
      specs
  in
  obj
    [
      ( "graph_module",
        obj
          [
            ( "graph",
              obj
                [
                  ("inputs", J.list (List.map t (inputs @ List.map fst params)));
                  ("outputs", J.list outputs);
                  ("nodes", J.list nodes);
                  ("tensor_values", obj tv);
                  ("sym_int_values", obj []);
                  ("sym_bool_values", obj []);
                  ("is_single_tensor_return", J.bool true);
                ] );
            ( "signature",
              obj
                [
                  ("input_specs", J.list specs);
                  ( "output_specs",
                    J.list
                      (List.map
                         (fun o -> obj [ ("user_output", obj [ ("arg", o) ]) ])
                         outputs) );
                ] );
            ("module_call_graph", J.list []);
          ] );
      ("opset_version", obj [ ("aten", J.int 15) ]);
      ("range_constraints", obj []);
      ("schema_version", obj [ ("major", J.int 8); ("minor", J.int 5) ]);
    ]

let shapes xs = List.map (fun (name, sizes) -> (name, tm sizes)) xs
let save name p = get (write (name ^ ".json") p)
let simple ?params name nodes tv = save name (program ?params nodes (shapes tv))
let named ns = List.map (fun n -> (n, n)) ns
