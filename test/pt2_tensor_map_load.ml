(* Gated smoke for the producer's external tensor-map contract. *)

let () =
  match Pt2_archive.load_pt_tensor_map Sys.argv.(1) with
  | Error e ->
      Format.eprintf "%a@." (Err.Error.pp Pt2_archive.pp_error) e;
      exit 1
  | Ok tensors ->
      List.iter
        (fun (key, tensor) ->
          Format.printf "%s: %a contiguous=%b@." key Pt2_tensor.pp tensor
            (Pt2_tensor.is_contiguous tensor))
        tensors
