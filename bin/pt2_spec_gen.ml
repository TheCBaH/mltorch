(* CLI: derive one ATen op-spec JSON per node of a real exported .pt2 model
   graph, written to an output directory (one file per node). See
   lib/pt2_spec_gen/pt2_spec_gen.ml and .ai/pt2_node_spec_design.md. *)

open Core.Syntax

let run model_pt2 out_dir =
  let* archive = Pt2_archive.open_pt2 model_pt2 in
  Core.return (Pt2_spec_gen.write_dir ~out_dir archive)

let () =
  match Sys.argv with
  | [| _; model_pt2; out_dir |] -> (
      match run model_pt2 out_dir with
      | Ok (written, skipped, total) ->
          Printf.printf "%d nodes, %d specs written, %d skipped\n" total written
            skipped
      | Error e ->
          Printf.eprintf "pt2_spec_gen: %s\n"
            (Format.asprintf "%a" (Core.Error.pp Pt2_archive.pp_error) e);
          exit 1)
  | _ ->
      prerr_endline "usage: pt2_spec_gen <model.pt2> <out_dir>";
      exit 1
