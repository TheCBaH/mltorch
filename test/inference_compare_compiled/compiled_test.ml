(* The embedded CompCert and the direct assembly routes as graph engines and
   as a pair under the shared coordinator, over small graphs: ownership of the
   input and output buffers, repeated and poisoned calls, a pair with several
   units under both donors, and a route that fails while the other does not. *)

open Inference_compare_schema
module C = Inference_compare_compiled
module P = Inference_compare_pt2
module G = P.Graph_model
module Tid = Graph_ir.Tensor_id
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let get = function Ok x -> x | Error m -> failwith m
let chain () = Native_test.Graph_fixtures.chain ()

(* Values that vary in sign and size, so a max, a denominator and a mask all
   matter; [salt] makes a different call's inputs differ. *)
let value ~salt (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      let k = (Vec6.offset sg.Tensor_sig.shape c :> int) + salt in
      0.25 +. (float_of_int (k mod 9) /. 8.))

let sig_of (g : Graph_ir.graph) id = Tid.Map.find id g.tensors

(* Every source an input, as an extracted body has them. *)
let body (g : Graph_ir.graph) =
  { g with Graph_ir.Graph.input_kinds = Tid.Map.empty }

let inputs ~salt (g : Graph_ir.graph) =
  List.map (fun id -> (id, value ~salt (sig_of g id))) g.inputs

let bits (Tensor.Tensor t as packed) =
  let acc = ref [] in
  Vec6.iter t.Tensor.shape (fun c ->
      acc := Int32.bits_of_float (Tensor.read packed c) :: !acc);
  List.rev !acc

let reference g ~salt =
  let all = inputs ~salt g in
  let is_const id = Graph_ir.input_kind g id = Graph_ir.Input.Constant in
  Err.or_raise ~pp_error:Eval_direct.pp_error
    (Eval_direct.run g
       ~constants:(List.filter (fun (id, _) -> is_const id) all)
       ~inputs:(List.filter (fun (id, _) -> not (is_const id)) all))

let same g outs ~salt =
  let expected = reference g ~salt in
  List.for_all2
    (fun id t -> bits t = bits (Tid.Map.find id expected))
    g.Graph_ir.Graph.outputs outs

(* Engines as functions over one prepared graph. *)
type engine = {
  name : string;
  compile : Graph_ir.graph -> (unit -> unit, string) result;
  run :
    poison:bool ->
    Graph_ir.graph ->
    (Tid.t * Tensor.packed) list ->
    (Tensor.packed list, string) result;
}

let compcert =
  let p = ref None in
  {
    name = "embedded CompCert";
    compile =
      (fun g ->
        Result.map
          (fun c ->
            p := Some c;
            fun () -> C.Compcert_engine.release c)
          (C.Compcert_engine.compile g));
    run =
      (fun ~poison g inputs ->
        let c = Option.get !p in
        if poison then C.Compcert_engine.execute_poisoned c g ~inputs
        else C.Compcert_engine.execute c g ~inputs);
  }

let assembly =
  let p = ref None in
  {
    name = "direct assembly";
    compile =
      (fun g ->
        Result.map
          (fun c ->
            p := Some c;
            fun () -> C.Assembly_engine.release c)
          (C.Assembly_engine.compile g));
    run =
      (fun ~poison:_ g inputs ->
        C.Assembly_engine.execute (Option.get !p) g ~inputs);
  }

let%expect_test "repeated calls match the reference, with a poisoned workspace"
    =
  let g = chain () in
  let b = body g in
  List.iter
    (fun e ->
      match e.compile b with
      | Error m -> Printf.printf "%s: refused: %s\n" e.name m
      | Ok release ->
          let verdict ~poison salt =
            match e.run ~poison b (inputs ~salt g) with
            | Error m -> m
            | Ok outs -> if same g outs ~salt then "bitwise" else "DIFFERS"
          in
          Printf.printf "%s: %s\n" e.name
            (String.concat ", "
               [
                 verdict ~poison:false 0;
                 verdict ~poison:false 1;
                 verdict ~poison:true 2;
                 verdict ~poison:true 0;
               ]);
          release ())
    [ compcert; assembly ];
  [%expect
    {|
    embedded CompCert: bitwise, bitwise, bitwise, bitwise
    direct assembly: bitwise, bitwise, bitwise, bitwise |}]

let%expect_test "a missing input is refused, never defaulted" =
  let g = chain () in
  let b = body g in
  List.iter
    (fun e ->
      match e.compile b with
      | Error m -> Printf.printf "%s: refused: %s\n" e.name m
      | Ok release ->
          let all = inputs ~salt:0 g in
          (match e.run ~poison:false b (List.tl all) with
          | Ok _ -> Printf.printf "%s: ran without an input\n" e.name
          | Error m -> Printf.printf "%s: %s\n" e.name m);
          release ())
    [ compcert; assembly ];
  [%expect
    {|
    embedded CompCert: no value for input t0
    direct assembly: no tensor for t0 |}]

(* ---- the pair under the shared coordinator -------------------------------- *)

let ctx : L.Context.t =
  {
    codecs = Codecs.Element_codec.Registry.builtin;
    hashers;
    algorithm;
    ledger = Codecs.Owned_blob.Ledger.create ~limit:1_000_000_000L ();
  }

let bytes_of (g : Graph_ir.graph) id =
  let t = value ~salt:3 (sig_of g id) in
  let b = Buffer.create 64 in
  let (Tensor.Tensor tt) = t in
  Vec6.iter tt.Tensor.shape (fun c ->
      Buffer.add_int32_le b (Int32.bits_of_float (Tensor.read t c)));
  Ok (Buffer.contents b)

let pair_of g =
  get
    (C.Compiled_pair.make ~hashers ~algorithm ~ctx ~graph:g ~bytes:(bytes_of g))

let run_pair ?(assembly = C.Compiled_runners.assembly) ~donor
    (pair : C.Compiled_pair.t) =
  let left_endpoint = C.Compiled_pair.compcert_endpoint pair in
  let right_endpoint : Inference_compare_core.Side_exec.endpoint =
    Endpoint
      {
        runner = assembly ();
        build =
          (fun ex ->
            Inference_compare_extract.Dialect.realize
              (module C.Compiled_pair.Dialect.Dialect)
              pair.assembly ex);
      }
  in
  Pair_harness.Harness.run ~donor ~tol:1e-4 ~mapping:pair.mapping
    ~left_graph:pair.compcert.desc ~right_graph:pair.assembly.desc
    ~initial:pair.initial
    ~node_attributes:(fun side r ->
      match (side : Side.t) with
      | Left -> G.attributes_of pair.compcert r
      | Right -> G.attributes_of pair.assembly r)
    ~left_endpoint ~right_endpoint
  |> ignore

let%expect_test "CompCert against direct assembly, several units, both donors" =
  let pair = pair_of (chain ()) in
  List.iter (fun donor -> run_pair ~donor pair) [ Side.Left; Side.Right ];
  [%expect
    {|
    extracted 6 graphs
    execution=complete numerical=agreement executed=3 units
    comparisons: 3 pass, 0 mismatch, 0 other
    extracted 6 graphs
    execution=complete numerical=agreement executed=3 units
    comparisons: 3 pass, 0 mismatch, 0 other |}]

let%expect_test
    "the shared Rivet ancestry is named, the compilers are not shared" =
  List.iter
    (fun (c : C.Ancestry.component) -> print_endline c.name)
    C.Ancestry.shared;
  [%expect
    {|
    Rivet assembler, image planner and native loader
    Loop IR bundle schedule and storage plan |}]

(* The assembly route refuses one unit's body; nothing runs in its place. *)
module Refusing = P.Graph_runner.Make (struct
  type op = Graph_ir.op
  type compiled = C.Assembly_engine.compiled

  let compile (g : Graph_ir.graph) =
    if
      List.exists
        (fun (n : Graph_ir.node) -> Graph_ir.op_name n.op = "Relu")
        g.nodes
    then Error "injected: this body is refused"
    else C.Assembly_engine.compile g

  let execute = C.Assembly_engine.execute
  let release = C.Assembly_engine.release
end)

let%expect_test "an injected route failure refuses the run, under either donor"
    =
  let pair = pair_of (chain ()) in
  List.iter
    (fun donor ->
      run_pair
        ~assembly:(fun () ->
          Refusing.create ~identity:C.Compiled_runners.assembly_identity)
        ~donor pair)
    [ Side.Left; Side.Right ];
  [%expect
    {|
    extracted 6 graphs
    execution=refused numerical=not_established executed=0 units
    failure: preparation_refused phase=preparation unit=ru00002 text=engine: injected: this body is refused
    comparisons: 0 pass, 0 mismatch, 0 other
    extracted 6 graphs
    execution=refused numerical=not_established executed=0 units
    failure: preparation_refused phase=preparation unit=ru00002 text=engine: injected: this body is refused
    comparisons: 0 pass, 0 mismatch, 0 other |}]
