open Inference_compare_schema
module Plan = Inference_compare_plan
module Ex = Inference_compare_extract
module E = Inference_compare_engines
module R = E.Synthetic_runner
module L = Inference_compare_codecs.Logical_value
module Run = Inference_compare_core.Runner
module P = Plan_fixtures
module C = Chain_fixture
module Codecs = Inference_compare_codecs

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let ctx = Codec_fixtures.context ()
let local (r : Tensor_ref.t) = Id.Tensor_local.to_string r.local_id

let extractions s =
  match Ex.Extraction.extract_all ~hashers ~algorithm s with
  | Ok l -> l
  | Error e ->
      Format.kasprintf failwith "%a" Ex.Extract_error.pp (Err.Error.kind e)

let body l u side =
  (List.find
     (fun (e : Ex.Extraction.t) ->
       Id.Unit.to_string e.unit = u && e.side = side)
     l)
    .body

let vec xs = Codec_fixtures.integers ctx C.i64_spec xs

(* Prepare, import, run, export and clean up one graph. Inputs are keyed by
   the local id of the source tensor. Returns outputs by local id, in order. *)
let run_graph (runner : R.t) (g : Graph_desc.t) inputs =
  let module M = (val runner) in
  let pp e = Format.asprintf "%a" M.Error.pp e in
  let ( let* ) = Result.bind in
  let unwrap r = Result.map_error (fun e -> pp (Err.Error.kind e)) r in
  let* p = unwrap (M.prepare g) in
  let cleanup = ref [] in
  Fun.protect
    ~finally:(fun () ->
      List.iter M.release_value !cleanup;
      M.close p)
    (fun () ->
      let* imported =
        List.fold_left
          (fun acc (s : Graph_source.t) ->
            let* acc = acc in
            let xs = List.assoc (local s.tensor) inputs in
            let v = vec xs in
            let r = M.import ctx p s.spec v in
            L.drop v;
            let* h = unwrap r in
            cleanup := h :: !cleanup;
            Ok ((s.tensor, h) :: acc))
          (Ok []) g.sources
      in
      let* outs = unwrap (M.run p ~inputs:(List.rev imported)) in
      List.iter (fun (_, v) -> cleanup := v :: !cleanup) outs;
      List.fold_left
        (fun acc (t, v) ->
          let* acc = acc in
          let* x = unwrap (M.export ctx v) in
          let ints = Codec_fixtures.to_integers ctx x in
          L.drop x;
          Ok (acc @ [ (local t, ints) ]))
        (Ok []) outs)

let pp_outputs = function
  | Ok outs ->
      String.concat " "
        (List.map
           (fun (n, xs) ->
             Printf.sprintf "%s=%s" n
               (Format.asprintf "%a" Codec_fixtures.pp_ints xs))
           outs)
  | Error m -> "ERROR " ^ m

let u1u2 = extractions (C.structural C.u1u2)
let u1 = body u1u2 "U1" Left
let u2 = body u1u2 "U2" Left

let%expect_test "the worked example in exact integers, both donors" =
  let left, _ = R.create (R.default_config "left") in
  let right, _ =
    R.create { (R.default_config "right") with offsets = [ ("n1", 1L) ] }
  in
  let show name runner g inputs =
    Format.printf "%s: %s@." name (pp_outputs (run_graph runner g inputs))
  in
  show "U1 left" left u1 [ ("x", [ 1L; 2L; 3L ]) ];
  show "U1 right (deviates by +1)" right u1 [ ("x", [ 1L; 2L; 3L ]) ];
  (* Either committed h feeds both U2 graphs. *)
  show "U2 left on left's h" left u2 [ ("h", [ 2L; 3L; 4L ]) ];
  show "U2 right on left's h" right u2 [ ("h", [ 2L; 3L; 4L ]) ];
  show "U2 left on right's h" left u2 [ ("h", [ 3L; 4L; 5L ]) ];
  show "U2 right on right's h" right u2 [ ("h", [ 3L; 4L; 5L ]) ];
  [%expect
    {|
    U1 left: h=[2; 3; 4]
    U1 right (deviates by +1): h=[3; 4; 5]
    U2 left on left's h: y=[200; 300; 400]
    U2 right on left's h: y=[200; 300; 400]
    U2 left on right's h: y=[300; 400; 500]
    U2 right on right's h: y=[300; 400; 500] |}]

let%expect_test "the runner sees a graph and inputs, nothing else" =
  let runner, counters = R.create (R.default_config "r") in
  let module M = (val runner) in
  ignore (run_graph runner u1 [ ("x", [ 1L; 2L; 3L ]) ]);
  Format.printf "input namespace seen by the engine: %s@."
    (String.concat ","
       (List.map (fun (n, _) -> n) (List.hd counters.observed_inputs)));
  Format.printf "graph key version: %a, scope of its tensors: %s@."
    Id.Graph_version.pp u1.key.version
    (String.concat "/"
       (List.map Name.Scope_component.to_string
          (List.hd
             (List.map (fun (s : Graph_source.t) -> s.tensor.scope) u1.sources))));
  [%expect
    {|
    input namespace seen by the engine: x
    graph key version: v1#U1:left, scope of its tensors: U1 |}]

let%expect_test "complete execution equals its own segmented execution" =
  let runner, _ = R.create (R.default_config "r") in
  let s = C.structural C.u1u2 in
  let whole = Plan.Graph_index.graph (Plan.Structural.left s) in
  let direct = run_graph runner whole [ ("x", [ 1L; 2L; 3L ]) ] in
  let h = run_graph runner u1 [ ("x", [ 1L; 2L; 3L ]) ] in
  let segmented =
    match h with
    | Ok [ (_, hv) ] -> run_graph runner u2 [ ("h", hv) ]
    | _ -> Error "no h"
  in
  Format.printf "whole:     %s@.segmented: %s@." (pp_outputs direct)
    (pp_outputs segmented);
  [%expect
    {|
    whole:     y=[200; 300; 400]
    segmented: y=[200; 300; 400] |}]

let%expect_test "input namespaces and inventories are enforced" =
  let runner, _ = R.create (R.default_config "r") in
  let module M = (val runner) in
  let g = u1 in
  let p = Result.get_ok (M.prepare g) in
  let v = vec [ 1L; 2L; 3L ] in
  let h = Result.get_ok (M.import ctx p C.i64_spec v) in
  let show name inputs =
    match M.run p ~inputs with
    | Ok outs ->
        Format.printf "%s: ok (%d outputs)@." name (List.length outs);
        List.iter (fun (_, o) -> M.release_value o) outs
    | Error e -> Format.printf "%s: %a@." name M.Error.pp (Err.Error.kind e)
  in
  let real = (List.hd g.sources).tensor in
  show "correct input" [ (real, h) ];
  show "no inputs" [];
  show "original-graph id instead of the extracted one"
    [ (P.tensor C.left_key "x", h) ];
  show "extra input" [ (real, h); (P.tensor C.left_key "extra", h) ];
  let wrong =
    Codec_fixtures.floats ctx (Codec_fixtures.f64 [ 3L ]) [ 1.0; 2.0; 3.0 ]
  in
  (match M.import ctx p (Codec_fixtures.f64 [ 3L ]) wrong with
  | Ok _ -> Format.printf "float import: accepted@."
  | Error e -> Format.printf "float import: %a@." M.Error.pp (Err.Error.kind e));
  (match M.import ctx p C.i64_spec wrong with
  | Ok _ -> Format.printf "spec/value mismatch: accepted@."
  | Error e ->
      Format.printf "spec/value mismatch: %a@." M.Error.pp (Err.Error.kind e));
  L.drop wrong;
  L.drop v;
  M.release_value h;
  M.close p;
  [%expect
    {|
    correct input: ok (1 outputs)
    no inputs: input inventory: wrong number of inputs
    original-graph id instead of the extracted one: input inventory: missing or foreign input
    extra input: input inventory: wrong number of inputs
    float import: import mismatch: only i64 tensors
    spec/value mismatch: import mismatch: spec |}]

let%expect_test "multi-output order and repeated operands survive execution" =
  let t =
    C.synthetic
      [
        C.node "n_dup" "dup" [ "x" ] [ "a"; "b" ];
        C.node "n_add" "addc:10" [ "a" ] [ "c" ];
        C.node "n_mul" "mul" [ "b"; "b" ] [ "d" ];
      ]
      ~sources:[ "x" ] ~finals:[ "c"; "d"; "b" ]
      ~units:[ ("U", [ "n_dup"; "n_add"; "n_mul" ]) ]
  in
  let all = extractions (C.structural t) in
  let runner, _ = R.create (R.default_config "r") in
  Format.printf "%s@."
    (pp_outputs
       (run_graph runner (body all "U" Left) [ ("x", [ 1L; 2L; 3L ]) ]));
  [%expect {| b=[1; 2; 3] c=[11; 12; 13] d=[1; 4; 9] |}]

let%expect_test "assertions run: zero arithmetic outputs, real failure" =
  let t =
    C.synthetic
      [
        C.node "chk" "assert_nonneg" [ "x" ] [] ~assertion:true;
        C.node "n1" "id" [ "x" ] [ "y" ];
      ]
      ~sources:[ "x" ] ~finals:[ "y" ]
      ~units:[ ("CHK", [ "chk" ]); ("U", [ "n1" ]) ]
  in
  let all = extractions (C.structural t) in
  let runner, counters = R.create (R.default_config "r") in
  let g = body all "CHK" Left in
  Format.printf "satisfied: %s@."
    (pp_outputs (run_graph runner g [ ("x", [ 1L; 2L; 3L ]) ]));
  Format.printf "violated: %s@."
    (pp_outputs (run_graph runner g [ ("x", [ 1L; -2L; 3L ]) ]));
  Format.printf "assertion executed twice: %b@."
    (List.length (List.filter (( = ) "chk") counters.executed_nodes) = 2);
  [%expect
    {|
    satisfied:
    violated: ERROR node chk failed
    assertion executed twice: true |}]

let%expect_test "faults: refusal, failing node, corrupted import" =
  let x = [ ("x", [ 1L; 2L; 3L ]) ] in
  let r1, c1 =
    R.create { (R.default_config "refuser") with refuse_ops = [ "addc:1" ] }
  in
  let refused = pp_outputs (run_graph r1 u1 x) in
  Format.printf "refused operator: %s (prepared open %d)@." refused
    c1.prepared_open;
  let r2, c2 =
    R.create { (R.default_config "failing") with fail_nodes = [ "n1" ] }
  in
  let failing = pp_outputs (run_graph r2 u1 x) in
  Format.printf "failing node: %s (prepared open %d, values live %d)@." failing
    c2.prepared_open c2.values_live;
  let r3, c3 =
    R.create
      { (R.default_config "corrupt") with corrupt_imports = [ ("x", 5L) ] }
  in
  let corrupted = pp_outputs (run_graph r3 u1 x) in
  Format.printf "corrupted import: %s; engine observed %s@." corrupted
    (String.concat ";"
       (List.map
          (fun (n, xs) ->
            n ^ "=" ^ Format.asprintf "%a" Codec_fixtures.pp_ints xs)
          (List.hd c3.observed_inputs)));
  let other_dialect =
    let t = { C.u1u2 with left_dialect = "aten" } in
    body (extractions (C.structural t)) "U1" Left
  in
  let r4, _ = R.create (R.default_config "strict") in
  Format.printf "foreign dialect operators: %s@."
    (pp_outputs (run_graph r4 other_dialect x));
  let bad_arity =
    let t =
      C.synthetic
        [ C.node "n1" "add" [ "x" ] [ "y" ] ]
        ~sources:[ "x" ] ~finals:[ "y" ]
        ~units:[ ("U", [ "n1" ]) ]
    in
    body (extractions (C.structural t)) "U" Left
  in
  Format.printf "wrong arity: %s@." (pp_outputs (run_graph r4 bad_arity x));
  [%expect
    {|
    refused operator: ERROR operator addc:1 refused (prepared open 0)
    failing node: ERROR node n1 failed (prepared open 0, values live 0)
    corrupted import: h=[7; 8; 9]; engine observed x=[6; 7; 8]
    foreign dialect operators: ERROR operator addc:1 refused
    wrong arity: ERROR unsupported graph: arity of add |}]

let%expect_test "asynchronous completion and cleanup counters" =
  let runner, c =
    R.create { (R.default_config "async") with async_polls = 3 }
  in
  let out = run_graph runner u1 [ ("x", [ 1L; 2L; 3L ]) ] in
  Format.printf "%s after %d completion polls@." (pp_outputs out) c.polls;
  Format.printf
    "prepares=%d imports=%d runs=%d exports=%d prepared open=%d values \
     live=%d@."
    c.prepares c.imports c.runs c.exports c.prepared_open c.values_live;
  let module M = (val runner) in
  let p = Result.get_ok (M.prepare u1) in
  M.close p;
  (try M.close p
   with Invalid_argument m -> Format.printf "double close: %s@." m);
  (match M.run p ~inputs:[] with
  | Error e ->
      Format.printf "run after close: %a@." M.Error.pp (Err.Error.kind e)
  | Ok _ -> Format.printf "run after close: ok@.");
  let v = vec [ 1L; 2L; 3L ] in
  let p2 = Result.get_ok (M.prepare u1) in
  let h = Result.get_ok (M.import ctx p2 C.i64_spec v) in
  M.release_value h;
  (try M.release_value h
   with Invalid_argument m -> Format.printf "double release: %s@." m);
  (match M.export ctx h with
  | Error e ->
      Format.printf "export after release: %a@." M.Error.pp (Err.Error.kind e)
  | Ok _ -> Format.printf "export after release: ok@.");
  L.drop v;
  M.close p2;
  Format.printf "balanced: prepared open=%d values live=%d@." c.prepared_open
    c.values_live;
  [%expect
    {|
    h=[2; 3; 4] after 3 completion polls
    prepares=1 imports=1 runs=1 exports=1 prepared open=0 values live=0
    double close: Synthetic_runner.close: already closed
    run after close: prepared graph is closed
    double release: Synthetic_runner.release_value: already released
    export after release: value released
    balanced: prepared open=0 values live=0 |}]

let%expect_test "reused output storage cannot change an exported snapshot" =
  let runner, _ =
    R.create { (R.default_config "reuse") with reuse_output_storage = true }
  in
  let module M = (val runner) in
  let p = Result.get_ok (M.prepare u1) in
  let src = (List.hd u1.sources).tensor in
  let run xs =
    let v = vec xs in
    let h = Result.get_ok (M.import ctx p C.i64_spec v) in
    L.drop v;
    let outs = Result.get_ok (M.run p ~inputs:[ (src, h) ]) in
    M.release_value h;
    snd (List.hd outs)
  in
  let first = run [ 1L; 2L; 3L ] in
  let snapshot = Result.get_ok (M.export ctx first) in
  let second = run [ 10L; 20L; 30L ] in
  let again = Result.get_ok (M.export ctx first) in
  Format.printf "exported snapshot after the second run: %a@."
    Codec_fixtures.pp_ints
    (Codec_fixtures.to_integers ctx snapshot);
  Format.printf "unexported first handle now reads: %a (storage was reused)@."
    Codec_fixtures.pp_ints
    (Codec_fixtures.to_integers ctx again);
  List.iter L.drop [ snapshot; again ];
  M.release_value first;
  M.release_value second;
  M.close p;
  [%expect
    {|
    exported snapshot after the second run: [2; 3; 4]
    unexported first handle now reads: [11; 21; 31] (storage was reused) |}]

let%expect_test "identity pins the configuration" =
  let a, _ = R.create (R.default_config "a") in
  let b, _ =
    R.create { (R.default_config "a") with offsets = [ ("n1", 1L) ] }
  in
  let module A = (val a) in
  let module B = (val b) in
  let ia = A.identity () and ib = B.identity () in
  Format.printf "same engine name: %b, same build: %b, same configuration: %b@."
    (Id.Engine.equal ia.engine ib.engine)
    (Digest.Build.equal ia.build_digest ib.build_digest)
    (Digest.Configuration.equal ia.configuration_digest ib.configuration_digest);
  [%expect
    {| same engine name: true, same build: true, same configuration: false |}]
