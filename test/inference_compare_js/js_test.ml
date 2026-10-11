open Graph_ir
module J = Inference_compare_js.Js_engines
module S = Inference_compare_js.Node_session
module W = Inference_compare_js_wire.Wire
module P = Inference_compare_pt2
module F = Native_test.Graph_fixtures
module Tid = Tensor_id

let engines :
    (string * (module P.Graph_runner.ENGINE with type op = Graph_ir.op)) list =
  [
    ("loop-js", (module J.Loop_js));
    ("ssa-js", (module J.Ssa_js));
    ("native-jsoo", (module J.Native_jsoo));
  ]

(* Every source an input, as an extracted body has them. *)
let body (g : graph) = { g with Graph.input_kinds = Tid.Map.empty }
let sig_of (g : graph) id = Tid.Map.find id g.Graph.tensors

let inputs ~salt (g : graph) =
  List.map
    (fun id ->
      ( id,
        Tensor.materialize (sig_of g id).Tensor_sig.shape (fun c ->
            float_of_int
              (((Vec6.offset (sig_of g id).Tensor_sig.shape c :> int) + salt)
              mod 7)
            /. 4.
            -. 0.75) ))
    g.Graph.inputs

let direct g inputs =
  match Eval_direct.run g ~inputs with
  | Ok env -> List.map (fun id -> Tid.Map.find id env) g.Graph.outputs
  | Error _ -> failwith "direct"

let two_outputs () =
  F.build2 "two_outputs"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:2 ~w:3 ~c:4) () in
      let* y = input ~shape:(F.nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = relu x in
      let* b = mul x y in
      return (a, b))

let graphs =
  [
    ("two outputs", two_outputs ());
    ("permute", F.sink_permute_unary ());
    ("residual", F.residual ());
  ]

let verdicts (module E : P.Graph_runner.ENGINE with type op = Graph_ir.op) g =
  match E.compile g with
  | Error m -> "refused: " ^ m
  | Ok c ->
      let v =
        List.map
          (fun salt ->
            let ins = inputs ~salt g in
            match E.execute c g ~inputs:ins with
            | Error m -> m
            | Ok outs ->
                string_of_bool
                  (List.length outs = List.length g.Graph.outputs
                  && List.for_all2 Tensor.equal_bits outs (direct g ins)))
          [ 0; 3 ]
      in
      E.release c;
      String.concat ", " v

let%expect_test "every JavaScript engine gives the direct evaluator's bits" =
  List.iter
    (fun (name, e) ->
      List.iter
        (fun (gname, g) ->
          Printf.printf "%s, %s: %s\n" name gname (verdicts e (body g)))
        graphs)
    engines;
  [%expect
    {|
    loop-js, two outputs: true, true
    loop-js, permute: true, true
    loop-js, residual: true, true
    ssa-js, two outputs: true, true
    ssa-js, permute: true, true
    ssa-js, residual: true, true
    native-jsoo, two outputs: true, true
    native-jsoo, permute: true, true
    native-jsoo, residual: true, true |}]

(* The check can fail: the same engine against the evaluator run on other
   inputs. *)
let%expect_test "a deviation is seen" =
  let g = body (F.residual ()) in
  let module E = J.Loop_js in
  (match E.compile g with
  | Error m -> print_endline m
  | Ok c ->
      (match E.execute c g ~inputs:(inputs ~salt:0 g) with
      | Error m -> print_endline m
      | Ok outs ->
          Printf.printf "other inputs: %b\n"
            (List.for_all2 Tensor.equal_bits outs (direct g (inputs ~salt:3 g))));
      E.release c);
  [%expect {| other inputs: false |}]

(* ---- exact integers and booleans ---- *)

let as_fmt fmt (g : graph) =
  let g = F.with_fmt (List.hd g.Graph.inputs) fmt g in
  F.with_fmt (List.hd g.Graph.outputs) fmt g

let permute_only () =
  F.build "permute_only"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:2 ~w:3 ~c:4) () in
      permute F.rotate_hwc x)

let big = Int64.add (Int64.shift_left 1L 53) 1L

let packed_i64 shape cells =
  let data =
    Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout (List.length cells)
  in
  List.iteri (fun i v -> Bigarray.Array1.set data i v) cells;
  Tensor.Tensor
    {
      Tensor.shape;
      payload = { Payload.fmt = Payload.I64; quant = Payload.No_quant; data };
    }

let%expect_test "an integer above 2^53 survives the round trip" =
  let g = body (as_fmt (Payload.Fmt Payload.I64) (permute_only ())) in
  let id = List.hd g.Graph.inputs in
  let shape = (sig_of g id).Tensor_sig.shape in
  let n = (Vec6.numel shape :> int) in
  let cells = List.init n (fun i -> Int64.add big (Int64.of_int i)) in
  let module E = J.Native_jsoo in
  (match E.compile g with
  | Error m -> print_endline ("refused: " ^ m)
  | Ok c ->
      (match E.execute c g ~inputs:[ (id, packed_i64 shape cells) ] with
      | Error m -> print_endline ("failed: " ^ m)
      | Ok [ Tensor.Tensor t ] -> (
          match t.Tensor.payload.Payload.fmt with
          | Payload.I64 ->
              Printf.printf "first %Ld last %Ld\n"
                (Bigarray.Array1.get t.Tensor.payload.Payload.data 0)
                (Bigarray.Array1.get t.Tensor.payload.Payload.data (n - 1))
          | _ -> print_endline "not i64")
      | Ok _ -> print_endline "wrong output count");
      E.release c);
  Printf.printf "expected first %Ld last %Ld\n" big
    (Int64.add big (Int64.of_int (n - 1)));
  [%expect
    {|
    first 9007199254740993 last 9007199254741016
    expected first 9007199254740993 last 9007199254741016 |}]

(* A boolean and an integer-valued output, compared bitwise with the direct
   evaluator by every engine. *)
let mixed () =
  Graph_builder.build ~name:"mixed"
    ~outputs:(fun (b, c) -> [ b; c ])
    (let open Graph_builder in
     let* x = input ~shape:(F.nhwc ~h:2 ~w:3 ~c:4) () in
     let* b = gt_scalar 0.1 x in
     let* c = argmax { Reduce.Argmax.axis = Axis.H; keepdim = false } x in
     return (b, c))
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let bool_only () =
  F.build "bool_only"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:2 ~w:3 ~c:4) () in
      gt_scalar 0.1 x)

let%expect_test "boolean and integer outputs are the direct evaluator's" =
  List.iter
    (fun (name, e) ->
      Printf.printf "%s: bool %s | bool and int64 %s\n" name
        (verdicts e (body (bool_only ())))
        (verdicts e (body (mixed ()))))
    engines;
  [%expect
    {|
    loop-js: bool true, true | bool and int64 refused: t2: a stored value must be f32 or bool and unquantized, got i64
    ssa-js: bool true, true | bool and int64 refused: t2: a stored value must be f32 or bool and unquantized, got i64
    native-jsoo: bool true, true | bool and int64 true, true |}]

(* ---- the helper's answers ---- *)

let session () =
  match J.helper () with
  | Error m -> failwith m
  | Ok helper -> (
      match S.start ~node:(J.node ()) ~helper with
      | Ok s -> s
      | Error m -> failwith m)

let show = function
  | Ok (W.Refused m) -> "refused: " ^ m
  | Ok (W.Failed m) -> "failed: " ^ m
  | Ok W.Prepared -> "prepared"
  | Ok W.Released -> "released"
  | Ok (W.Outputs ts) -> Printf.sprintf "%d outputs" (List.length ts)
  | Ok (W.Welcome { engines; _ }) -> String.concat "," engines
  | Error m -> "error: " ^ m

let%expect_test "the helper says what it holds, and refuses what it cannot take"
    =
  let s = session () in
  print_endline (show (S.request s W.Hello));
  let g = body (F.residual ()) in
  let graph =
    match Jsont_bytesrw.encode_string Graph_ir.graph_jsont g with
    | Ok t -> t
    | Error m -> failwith m
  in
  print_endline
    (show (S.request s (W.Prepare { handle = 1; engine = "loop-js"; graph })));
  print_endline
    (show (S.request s (W.Prepare { handle = 1; engine = "loop-js"; graph })));
  print_endline
    (show (S.request s (W.Prepare { handle = 2; engine = "wasm"; graph })));
  print_endline
    (show
       (S.request s
          (W.Prepare { handle = 3; engine = "loop-js"; graph = "{}" })));
  print_endline (show (S.request s (W.Run { handle = 9; inputs = [] })));
  print_endline (show (S.request s (W.Release 9)));
  print_endline (show (S.request s (W.Release 1)));
  S.close s;
  [%expect
    {|
    loop-js,native-jsoo,ssa-js
    prepared
    refused: handle already in use
    refused: no engine "wasm"
    refused: the graph does not decode: graph: missing "tensors"
    failed: no such prepared body
    released
    released |}]

(* A body that is not a message gets an answer, not a hang. *)
let%expect_test "a malformed request is answered" =
  let s = session () in
  output_string s.S.to_child (W.frame "not a message");
  flush s.S.to_child;
  let head = really_input_string s.S.from_child 4 in
  let n = Result.get_ok (W.frame_length head) in
  print_endline
    (show (W.decode_response (really_input_string s.S.from_child n)));
  print_endline (show (S.request s W.Hello));
  S.close s;
  [%expect
    {|
    failed: malformed request: not a comparison message
    loop-js,native-jsoo,ssa-js |}]

let%expect_test "a helper that dies is reported, with how" =
  let s = session () in
  ignore (S.request s W.Hello);
  Unix.kill s.S.pid Sys.sigkill;
  ignore (Unix.waitpid [ Unix.WNOHANG ] s.S.pid);
  Unix.sleepf 0.1;
  (match S.request s W.Hello with
  | Error m -> print_endline (String.sub m 0 (min 40 (String.length m)))
  | Ok _ -> print_endline "answered");
  S.close s;
  [%expect {| node helper: Broken pipe (killed by sign |}]

let alive pid =
  match Unix.kill pid 0 with () -> true | exception Unix.Unix_error _ -> false

let%expect_test "the helper process ends with the last body" =
  let g = body (F.residual ()) in
  let module E = J.Loop_js in
  let a = Result.get_ok (E.compile g) and b = Result.get_ok (E.compile g) in
  let pid = a.E.session.S.pid in
  Printf.printf "shared: %b, alive: %b\n"
    (a.E.session == b.E.session)
    (alive pid);
  E.release a;
  Printf.printf "after one release: %b\n" (alive pid);
  E.release b;
  Printf.printf "after both: %b\n" (alive pid);
  [%expect
    {|
    shared: true, alive: true
    after one release: true
    after both: false |}]

let%expect_test "a missing helper is a refusal naming it" =
  Inference_compare_js.Js_engines.register ();
  Unix.putenv "MLTORCH_COMPARE_JS" "/no/such/helper.js";
  (match P.Provider.instantiate "loop-js" with
  | Ok _ -> print_endline "instantiated"
  | Error m -> print_endline m);
  Unix.putenv "MLTORCH_COMPARE_JS" "";
  [%expect
    {| loop-js is not available: the JavaScript helper js/jsoo/compare_js_engine/compare_js_engine.bc.js is not built (dune build js/jsoo/compare_js_engine), or name it in MLTORCH_COMPARE_JS |}]

let%expect_test
    "each word advertises its own identity, before any body is prepared" =
  Inference_compare_js.Js_engines.register ();
  let identity name =
    match P.Provider.instantiate name with
    | Error m -> failwith m
    | Ok i ->
        let (module R) = i.runner in
        Printf.printf "%s: runner = advertised: %b\n" name
          (R.identity () = i.identity);
        i.identity
  in
  let l = identity "loop-js"
  and s = identity "ssa-js"
  and j = identity "jsoo" in
  Printf.printf "distinct: %b\n" (l <> s && s <> j && l <> j);
  Printf.printf "repeatable: %b\n" (l = identity "loop-js");
  [%expect
    {|
    loop-js: runner = advertised: true
    ssa-js: runner = advertised: true
    jsoo: runner = advertised: true
    distinct: true
    loop-js: runner = advertised: true
    repeatable: true |}]
