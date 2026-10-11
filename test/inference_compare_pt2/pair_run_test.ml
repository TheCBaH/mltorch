(* The diagnostic runner on a graph whose outputs are float32, bool and int64:
   each comparison takes the policy of its element type, so the pair completes,
   and a flipped boolean or index disagrees however large the tolerance. *)

open Inference_compare_schema
module P = Inference_compare_pt2
module Host = Inference_compare_host
module Core = Inference_compare_core
module Ex = Inference_compare_extract
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module G = P.Graph_model

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let get = function Ok x -> x | Error m -> failwith m

(* relu (float32), a threshold (bool) and the argmax along an axis (int64): one
   structure whatever the two parameters, so two parameterisations are related
   by the identity. Each node is a cluster fed the donor's inputs, so a
   difference shows in the node that holds it. *)
let graph ~threshold ~axis =
  Graph_builder.build ~name:"mixed"
    ~outputs:(fun (a, b, c) -> [ a; b; c ])
    (let open Graph_builder in
     let* x = input ~shape:(Native_test.Pipeline_test.nhwc ~h:2 ~w:2 ~c:3) () in
     let* a = relu x in
     let* b = gt_scalar threshold x in
     let* c = argmax { Reduce.Argmax.axis; keepdim = false } x in
     return (a, b, c))
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let model ~version g =
  get
    (G.make ~ops:G.native_ops ~hashers ~algorithm ~version
       ~identity_text:version g)

let ctx =
  {
    L.Context.codecs = Codecs.Element_codec.Registry.builtin;
    hashers;
    algorithm;
    ledger = Codecs.Owned_blob.Ledger.create ~limit:1_000_000L ();
  }

(* The one source of the graph, an input, with the same value on both sides. *)
let input_bytes (g : Graph_ir.graph) _id =
  let id = List.hd g.inputs in
  let sg = Graph_ir.Tensor_id.Map.find id g.tensors in
  let packed =
    Tensor.materialize sg.Tensor_sig.shape (fun c ->
        float_of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 7) /. 4.)
  in
  let spec = get (G.spec_of g id) in
  let v = get (P.Conv.logical_of_native ctx spec packed) in
  let b = L.contents v in
  L.drop v;
  Ok b

let endpoint (m : Graph_ir.op G.t) : Core.Side_exec.endpoint =
  Endpoint
    {
      runner = P.Native_direct.create ();
      build =
        (fun ex ->
          Ex.Dialect.realize (module P.Native_direct.Dialect.Dialect) m ex);
    }

let run ~tolerance (lg, rg) =
  let g0 = graph ~threshold:0.5 ~axis:Axis.H in
  let l = model ~version:"left" lg and r = model ~version:"right" rg in
  let digest b =
    Codecs.Payload_hasher.digest_string hashers algorithm b
    |> Result.map_error (fun _ -> "digest")
  in
  let p =
    get
      (P.Pass_map.make ~name:"s" ~digest
         ~src:{ P.Pass_map.model = l; bytes = input_bytes l.graph }
         ~dst:{ P.Pass_map.model = r; bytes = input_bytes r.graph }
         ~export:(P.Pass_map.identity_export g0))
  in
  let initial =
    get (P.Pass_map.pinned_values ~ctx ~mapping:p.mapping ~known:[] p.pins)
  in
  Host.Pair_run.run ~on_event:ignore ~donor:Left ~tolerance ~mapping:p.mapping
    ~left_graph:l.desc ~right_graph:r.desc ~initial
    ~node_attributes:(fun side n ->
      match (side : Side.t) with
      | Left -> G.attributes_of l n
      | Right -> G.attributes_of r n)
    ~left_endpoint:(endpoint l) ~right_endpoint:(endpoint r)

let show ~tolerance pair =
  match run ~tolerance pair with
  | Error e -> Format.printf "refused: %a@." Host.Session.pp_error e
  | Ok o ->
      let r = o.result in
      let count v =
        List.length
          (List.filter
             (fun (c : Comparison_result.t) -> c.verdict = v)
             r.comparisons)
      in
      Printf.printf "%d pass, %d mismatch, exit %d\n" (count Pass)
        (count Mismatch)
        (Host.Pair_run.exit_code r)

let same = (graph ~threshold:0.5 ~axis:Axis.H, graph ~threshold:0.5 ~axis:Axis.H)

let%expect_test "the same graph twice completes, exactly" =
  show ~tolerance:0. same;
  [%expect {| 3 pass, 0 mismatch, exit 0 |}]

let%expect_test "a flipped boolean disagrees at any tolerance" =
  show ~tolerance:1e9
    (graph ~threshold:0.5 ~axis:Axis.H, graph ~threshold:1.0 ~axis:Axis.H);
  [%expect {| 2 pass, 1 mismatch, exit 1 |}]

let%expect_test "a different index disagrees at any tolerance" =
  show ~tolerance:1e9
    (graph ~threshold:0.5 ~axis:Axis.H, graph ~threshold:0.5 ~axis:Axis.W);
  [%expect {| 2 pass, 1 mismatch, exit 1 |}]

let failure phase : Failure.t =
  {
    code = Execution_failed;
    phase;
    side = None;
    unit = None;
    node = None;
    tensor = None;
    rule = None;
    comparison = None;
    expected = None;
    actual = None;
  }

let outcome execution numerical failure : Core.Coordinator.result =
  {
    completion =
      {
        execution;
        coverage = Complete;
        numerical;
        evidence = Independent_baseline;
      };
    failure;
    comparisons = [];
    commits = [];
    finals = [];
    executed = [];
    lineage = [];
    events = [];
  }

let%expect_test "each outcome maps to its documented exit code" =
  List.iter
    (fun (name, r) -> Printf.printf "%s: %d\n" name (Host.Pair_run.exit_code r))
    [
      ("agreement", outcome Complete Agreement None);
      ("disagreement", outcome Complete Disagreement None);
      ("invalid plan", outcome Refused Not_established (Some (failure Planning)));
      ( "failed execution",
        outcome Failed Not_established (Some (failure Execution)) );
      ("not established", outcome Complete Not_established None);
      ("interrupted", outcome Interrupted Not_established None);
    ];
  [%expect
    {|
    agreement: 0
    disagreement: 1
    invalid plan: 2
    failed execution: 3
    not established: 4
    interrupted: 4 |}]
