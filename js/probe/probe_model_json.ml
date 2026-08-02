(* Section 5: decoding a real PT2 model.json. Not melange-reachable: it goes
   through Jsont_bytesrw, and pytorch_types is generated from the schema.yaml in
   the modules/pytorch submodule.

   This is the exact call Pt2_archive makes on a real archive, run on the
   committed model.json of the smallest model in the zoo (resnet18, 224K). It
   answers a question the tensor sections cannot: can a JS runtime read a model
   at all -- which is what a browser viewer or playground needs first.

   What is printed is a STRUCTURAL summary, not the tree. A 224K JSON file
   rendered into the diff would drown every other section, and the summary still
   moves on any decode divergence: a dropped node, a mis-parsed target name or a
   changed arity all show up. *)

open Pytorch_types
open Schema_runtime

let decode_program json =
  (* Plain [result] from Jsont, so [failwith] rather than [Core.or_raise]. And
     it must RAISE, not print: both backends run this same source, so a decoder
     that failed identically on each would print the same text, diff clean and
     exit 0. See [[testing_strategy]]. *)
  match Jsont_bytesrw.decode_string ExportedProgram.jsont json with
  | Ok program -> program
  | Error e -> failwith ("probe: model.json decode: " ^ e)

let run path =
  print_endline "=== model-json ===";
  let json =
    try In_channel.with_open_bin path In_channel.input_all
    with Sys_error msg -> failwith ("probe: cannot read model.json: " ^ msg)
  in
  Printf.printf "bytes %d\n" (String.length json);
  let program = decode_program json in
  let graph = program.ExportedProgram.graph_module.GraphModule.graph in
  Printf.printf "nodes %d\n" (List.length graph.Graph.nodes);
  Printf.printf "inputs %d outputs %d\n"
    (List.length graph.Graph.inputs)
    (List.length graph.Graph.outputs);
  Printf.printf "tensor-values %d\n"
    (String_map.cardinal graph.Graph.tensor_values);
  (* Distinct targets, sorted: order-independent, so the line cannot pick up an
     incidental difference in how either backend traverses the JSON, only a real
     difference in what was decoded. *)
  let targets =
    List.map (fun (n : Node.t) -> n.Node.target) graph.Graph.nodes
    |> List.sort_uniq String.compare
  in
  Printf.printf "distinct-targets %d\n" (List.length targets);
  List.iter (fun t -> Printf.printf "  %s\n" t) targets
