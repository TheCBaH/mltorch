(* The symbolic evaluator as an engine: grounding a body's stage program gives
   the direct evaluator's bits, on every input tried, and a missing input is a
   refusal rather than a guess. *)

open Graph_ir
module S = Inference_compare_pt2.Symbolic_runner
module Tid = Tensor_id

let chain =
  Native_test.Pipeline_test.build "chain"
    (let open Graph_builder in
     let* x = input ~shape:(Native_test.Pipeline_test.nhwc ~h:4 ~w:4 ~c:3) () in
     let* a = relu x in
     let* b = sqrt a in
     let* c = relu b in
     sqrt c)

let input_of g seed =
  let id = List.hd g.Graph.inputs in
  let sg = Tid.Map.find id g.Graph.tensors in
  ( id,
    Tensor.materialize sg.Tensor_sig.shape (fun c ->
        float_of_int (((Vec6.offset sg.Tensor_sig.shape c :> int) + seed) mod 7)
        /. 4.) )

let get = function Ok x -> x | Error m -> failwith m

let direct g inputs =
  match Eval_direct.run g ~inputs with
  | Ok env -> List.map (fun id -> Tid.Map.find id env) g.Graph.outputs
  | Error _ -> failwith "direct"

let%expect_test "grounding the stage program gives the direct bits" =
  let program = get (S.Native_engine.compile chain) in
  List.iter
    (fun seed ->
      let inputs = [ input_of chain seed ] in
      let symbolic = get (S.Native_engine.execute program chain ~inputs) in
      Printf.printf "seed %d: %b\n" seed
        (List.for_all2 Tensor.equal_bits symbolic (direct chain inputs)))
    [ 0; 1; 5 ];
  [%expect {|
    seed 0: true
    seed 1: true
    seed 5: true |}]

let%expect_test "different inputs give different bits" =
  let program = get (S.Native_engine.compile chain) in
  let run seed =
    get (S.Native_engine.execute program chain ~inputs:[ input_of chain seed ])
  in
  Printf.printf "%b\n" (List.for_all2 Tensor.equal_bits (run 0) (run 1));
  [%expect {| false |}]

let%expect_test "a missing input is refused" =
  let program = get (S.Native_engine.compile chain) in
  (match S.Native_engine.execute program chain ~inputs:[] with
  | Ok _ -> print_endline "ran"
  | Error m -> print_endline m);
  [%expect {| Native symbolic: no value for input t0 |}]
