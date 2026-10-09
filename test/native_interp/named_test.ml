(* Named user inputs: the binder behind [Native_interp.run_named], and the
   one-input [run] it must not change. Graphs are built as JSON; reference
   values are worked out by hand in the comments. *)

let jstr = Printf.sprintf
let float_meta sizes = (7, sizes)
let long_meta sizes = (5, sizes)
let bool_meta sizes = (12, sizes)

let tensor_meta (dtype, sizes) =
  jstr
    {|{"dtype":%d,"sizes":[%s],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    dtype
    (String.concat "," (List.map (fun i -> jstr {|{"as_int":%d}|} i) sizes))

let as_tensor n = jstr {|{"as_tensor":{"name":"%s"}}|} n

let to_copy ~src ~dst ~dtype =
  jstr
    {|{"target":"torch.ops.aten._to_copy.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"dtype","arg":{"as_scalar_type":%d},"kind":2},{"name":"non_blocking","arg":{"as_bool":false},"kind":2}],"outputs":[%s],"metadata":{}}|}
    (as_tensor src) dtype (as_tensor dst)

let binary target ~a ~b ~dst =
  jstr
    {|{"target":"torch.ops.aten.%s","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"other","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    target (as_tensor a) (as_tensor b) (as_tensor dst)

(* [inputs] are (name, meta) user inputs in graph order; [nodes] and
   [outputs] as usual. [extra_output_specs] lets a test append a non-user
   output to the signature. *)
let program ?(extra_output_specs = []) ~inputs ~nodes ~outputs ~values () =
  let tensor_values =
    String.concat ","
      (List.map
         (fun (n, m) -> jstr {|"%s":%s|} n (tensor_meta m))
         (inputs @ values))
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":false},"signature":{"input_specs":[%s],"output_specs":[%s]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (String.concat "," (List.map (fun (n, _) -> as_tensor n) inputs))
    (String.concat "," (List.map as_tensor outputs))
    (String.concat "," nodes) tensor_values
    (String.concat ","
       (List.map
          (fun (n, _) -> jstr {|{"user_input":{"arg":%s}}|} (as_tensor n))
          inputs))
    (String.concat ","
       (List.map
          (fun o -> jstr {|{"user_output":{"arg":%s}}|} (as_tensor o))
          outputs
       @ extra_output_specs))

(* x: f32, ids: i64, keep: bool -- all [2,3]. y = x + float(ids);
   z = float(keep) * x. *)
let mixed ?extra_output_specs () =
  program ?extra_output_specs
    ~inputs:
      [
        ("x", float_meta [ 2; 3 ]);
        ("ids", long_meta [ 2; 3 ]);
        ("keep", bool_meta [ 2; 3 ]);
      ]
    ~values:
      [
        ("idsf", float_meta [ 2; 3 ]);
        ("keepf", float_meta [ 2; 3 ]);
        ("y", float_meta [ 2; 3 ]);
        ("z", float_meta [ 2; 3 ]);
      ]
    ~nodes:
      [
        to_copy ~src:"ids" ~dst:"idsf" ~dtype:7;
        to_copy ~src:"keep" ~dst:"keepf" ~dtype:7;
        binary "add.Tensor" ~a:"x" ~b:"idsf" ~dst:"y";
        binary "mul.Tensor" ~a:"keepf" ~b:"x" ~dst:"z";
      ]
    ~outputs:[ "y"; "z" ] ()

(* a - b, both f32 [3]: swapping the two changes the answer. *)
let difference () =
  program
    ~inputs:[ ("a", float_meta [ 3 ]); ("b", float_meta [ 3 ]) ]
    ~values:[ ("d", float_meta [ 3 ]) ]
    ~nodes:[ binary "sub.Tensor" ~a:"a" ~b:"b" ~dst:"d" ]
    ~outputs:[ "d" ] ()

let archive json =
  match
    Jsont_bytesrw.decode_string Pytorch_types.ExportedProgram.jsont json
  with
  | Error e -> failwith ("fixture did not decode: " ^ e)
  | Ok program ->
      let none =
        {
          Pytorch_weights_config.ModelWeightsConfig.config =
            Schema_runtime.String_map.empty;
        }
      in
      Pt2_archive.of_parts ~program ~weights:none ~constants:none
        ~load:(fun _ -> Error "no payload")

(* --- tensors --- *)

let contiguous sizes =
  let rec go = function
    | [] -> []
    | _ :: rest as l -> List.fold_left ( * ) 1 (List.tl l) :: go rest
  in
  go sizes

let f32 sizes values =
  let data = Bytes.create (4 * List.length values) in
  List.iteri
    (fun i v -> Bytes.set_int32_le data (4 * i) (Int32.bits_of_float v))
    values;
  {
    Pt2_tensor.dtype = Pt2_dtype.Float32;
    sizes;
    strides = contiguous sizes;
    storage_offset = 0;
    data = Pt2_storage.of_string (Bytes.to_string data);
  }

let i64 sizes values =
  let data = Bytes.create (8 * List.length values) in
  List.iteri
    (fun i v -> Bytes.set_int64_le data (8 * i) (Int64.of_int v))
    values;
  {
    Pt2_tensor.dtype = Pt2_dtype.Int64;
    sizes;
    strides = contiguous sizes;
    storage_offset = 0;
    data = Pt2_storage.of_string (Bytes.to_string data);
  }

let bool sizes values =
  let data =
    String.init (List.length values) (fun i ->
        if List.nth values i then '\001' else '\000')
  in
  {
    Pt2_tensor.dtype = Pt2_dtype.Bool;
    sizes;
    strides = contiguous sizes;
    storage_offset = 0;
    data = Pt2_storage.of_string data;
  }

(* The elements of a rank-2 result in logical row-major order: sizes right-align
   into the frame, so a [rows; cols] tensor is [W=rows], [C=cols]. *)
let read_floats packed =
  let (Tensor.Tensor t) = packed in
  let extent a = (Vec6.get t.Tensor.shape a :> int) in
  List.concat
    (List.init (extent Axis.W) (fun w ->
         List.init (extent Axis.C) (fun c ->
             Tensor.read packed (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w ~c))))

let show_outputs = function
  | Ok outputs ->
      List.iteri
        (fun i o ->
          print_endline
            (Printf.sprintf "output %d: %s" i
               (String.concat " "
                  (List.map (Printf.sprintf "%g") (read_floats o)))))
        outputs
  | Error e -> Fmt.pr "%a@." Native_interp.pp_error (Err.Error.kind e)

let x = f32 [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]
let ids = i64 [ 2; 3 ] [ 10; 20; 30; 40; 50; 60 ]
let keep = bool [ 2; 3 ] [ true; false; true; false; true; false ]

let%expect_test "mixed dtypes bind by name, in any order" =
  (* y = x + ids = 11 22 33 44 55 66; z = keep * x = 1 0 3 0 5 0 *)
  show_outputs
    (Native_interp.run_named
       (archive (mixed ()))
       ~inputs:[ ("x", x); ("ids", ids); ("keep", keep) ]);
  show_outputs
    (Native_interp.run_named
       (archive (mixed ()))
       ~inputs:[ ("keep", keep); ("x", x); ("ids", ids) ]);
  [%expect
    {|
    output 0: 11 22 33 44 55 66
    output 1: 1 0 3 0 5 0
    output 0: 11 22 33 44 55 66
    output 1: 1 0 3 0 5 0 |}]

let fails = function
  | Ok _ -> print_endline "ran"
  | Error e -> Fmt.pr "%a@." Native_interp.pp_error (Err.Error.kind e)

let named ?(json = mixed ()) inputs =
  fails (Native_interp.run_named (archive json) ~inputs)

let%expect_test "a named call is refused before anything is evaluated" =
  let all = [ ("x", x); ("ids", ids); ("keep", keep) ] in
  print_endline "-- names";
  named (List.remove_assoc "ids" all);
  named (("extra", x) :: all);
  named (("x", x) :: all);
  named [];
  print_endline "-- dtype: values under the wrong name";
  named [ ("x", ids); ("ids", x); ("keep", keep) ];
  named [ ("x", x); ("ids", ids); ("keep", ids) ];
  named
    [
      ("x", { x with Pt2_tensor.dtype = Pt2_dtype.Float64 });
      ("ids", ids);
      ("keep", keep);
    ];
  print_endline "-- shape and rank";
  named
    [
      ("x", f32 [ 3; 2 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]);
      ("ids", ids);
      ("keep", keep);
    ];
  named
    [
      ("x", f32 [ 6 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]); ("ids", ids); ("keep", keep);
    ];
  named
    [
      ("x", f32 [ 1; 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ]);
      ("ids", ids);
      ("keep", keep);
    ];
  print_endline "-- a signature that mutates state";
  named
    ~json:
      (mixed
         ~extra_output_specs:
           [ {|{"buffer_mutation":{"arg":{"name":"y"},"buffer_name":"b"}}|} ]
         ())
    all;
  [%expect
    {|
    -- names
    named input "ids" is not supplied
    named input "extra" is not a user input of the graph
    named input "x" is supplied more than once
    named input "x" is not supplied
    -- dtype: values under the wrong name
    named input "x": graph declares float32, got int64
    named input "keep": graph declares bool, got int64
    named input "x": graph declares float32, got float64
    -- shape and rank
    named input "x": graph declares shape [2; 3], got [3; 2]
    named input "x": graph declares shape [2; 3], got [6]
    named input "x": graph declares shape [2; 3], got [1; 2; 3]
    -- a signature that mutates state
    the signature mutates state (buffer mutation); a named call cannot bind it |}]

(* The same-dtype case a dtype check cannot catch: only the name decides. *)
let%expect_test "names, not positions, decide which tensor is which" =
  let a = f32 [ 3 ] [ 10.; 20.; 30. ] and b = f32 [ 3 ] [ 1.; 2.; 3. ] in
  let run inputs = Native_interp.run_named (archive (difference ())) ~inputs in
  (* a - b = 9 18 27, whichever order the list arrives in *)
  let read = function
    | Ok [ o ] ->
        let (Tensor.Tensor t) = o in
        let c = (Vec6.get t.Tensor.shape Axis.C :> int) in
        print_endline
          (String.concat " "
             (List.init c (fun c ->
                  Printf.sprintf "%g"
                    (Tensor.read o (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c)))))
    | Ok _ -> print_endline "wrong output count"
    | Error e -> Fmt.pr "%a@." Native_interp.pp_error (Err.Error.kind e)
  in
  read (run [ ("a", a); ("b", b) ]);
  read (run [ ("b", b); ("a", a) ]);
  (* swapped values under the same names is a different, correct, answer *)
  read (run [ ("a", b); ("b", a) ]);
  [%expect {|
    9 18 27
    9 18 27
    -9 -18 -27 |}]

let%expect_test "run keeps its one-input contract" =
  (* Single user input: [run] and [run_named] agree. *)
  let single =
    program
      ~inputs:[ ("a", float_meta [ 3 ]) ]
      ~values:[ ("d", float_meta [ 3 ]) ]
      ~nodes:[ binary "mul.Tensor" ~a:"a" ~b:"a" ~dst:"d" ]
      ~outputs:[ "d" ] ()
  in
  let a = f32 [ 3 ] [ 1.; 2.; 3. ] in
  let floats = function
    | Ok [ o ] ->
        String.concat " "
          (List.init 3 (fun c ->
               Printf.sprintf "%g"
                 (Tensor.read o (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c))))
    | _ -> "failed"
  in
  Fmt.pr "run:       %s@."
    (floats (Native_interp.run (archive single) ~input:a));
  Fmt.pr "run_named: %s@."
    (floats (Native_interp.run_named (archive single) ~inputs:[ ("a", a) ]));
  (* Several user inputs: [run] still refuses, with its own row, and does not
     validate dtypes or names. *)
  fails (Native_interp.run (archive (mixed ())) ~input:x);
  fails (Native_interp.run (archive (difference ())) ~input:a);
  [%expect
    {|
    run:       1 4 9
    run_named: 1 4 9
    unsupported PT2 input: expected one user input, got 3
    unsupported PT2 input: expected one user input, got 2 |}]
