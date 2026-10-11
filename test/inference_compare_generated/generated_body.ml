(* One small body and the checks every engine passes: its results are the
   direct evaluator's bit for bit, on a first run and on a rerun with other
   inputs. *)

open Graph_ir
module Tid = Tensor_id
module Pt = Native_test.Pipeline_test

let chain =
  Pt.build "chain"
    (let open Graph_builder in
     let* x = input ~shape:(Pt.nhwc ~h:4 ~w:4 ~c:3) () in
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

let direct g inputs =
  match Eval_direct.run g ~inputs with
  | Ok env -> List.map (fun id -> Tid.Map.find id env) g.Graph.outputs
  | Error _ -> failwith "direct"

module Check_graph
    (E : Inference_compare_pt2.Graph_runner.ENGINE with type op = Graph_ir.op) =
struct
  let run g =
    match E.compile g with
    | Error m -> print_endline ("refused: " ^ m)
    | Ok c ->
        List.iter
          (fun seed ->
            let inputs = [ input_of g seed ] in
            match E.execute c g ~inputs with
            | Error m -> print_endline ("failed: " ^ m)
            | Ok outs ->
                Printf.printf "seed %d: %b\n" seed
                  (List.for_all2 Tensor.equal_bits outs (direct g inputs)))
          [ 0; 3 ];
        E.release c
end

module Check
    (E : Inference_compare_pt2.Graph_runner.ENGINE with type op = Graph_ir.op) =
struct
  module G = Check_graph (E)

  let run () = G.run chain
end
