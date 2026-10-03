(* Writes each feature's probe module to <dir>/<feature>.wasm. *)
let () =
  let dir = Sys.argv.(1) in
  List.iter
    (fun f ->
      let oc =
        open_out_bin (Filename.concat dir (Wasm_features.name f ^ ".wasm"))
      in
      output_string oc (Wasm_features.probe f);
      close_out oc)
    Wasm_features.all
