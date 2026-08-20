(* Payload-free compatibility census for a committed functional graph. *)

open Pytorch_types
module Targets = Map.Make (String)

let read path = In_channel.with_open_bin path In_channel.input_all

let report path =
  let contents = read path in
  match Jsont_bytesrw.decode_string ExportedProgram.jsont contents with
  | Error e -> Format.eprintf "parser: error %s@." e
  | Ok program ->
      let graph = program.ExportedProgram.graph_module.graph in
      let targets =
        List.fold_left
          (fun counts (node : Node.t) ->
            Targets.update node.target
              (function None -> Some 1 | Some n -> Some (n + 1))
              counts)
          Targets.empty graph.nodes
      in
      Printf.printf "graph=%s\nfingerprint=%s\nnodes=%d targets=%d\n" path
        (Digest.to_hex (Digest.string contents))
        (List.length graph.nodes) (Targets.cardinal targets);
      Targets.iter
        (fun target count -> Printf.printf "%s %d\n" target count)
        targets

let () =
  match Sys.argv with
  | [| _; path |] -> report path
  | _ ->
      prerr_endline "usage: pt2_census MODEL.json";
      exit 2
