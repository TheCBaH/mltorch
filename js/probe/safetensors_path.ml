(* Print the path of the checkpoint a model directory's safetensors.json pins,
   fetching it into the huggingface_hub cache first and checking it against the
   pin. Native only. *)

let () =
  match Sys.argv with
  | [| _; model_dir |] ->
      Pt2_safetensors_unix.checkpoint_path model_dir
      |> Err.or_raise ~pp_error:Pt2_safetensors_unix.pp_error
      |> print_endline
  | _ ->
      prerr_endline "usage: safetensors_path <model_dir>";
      exit 2
