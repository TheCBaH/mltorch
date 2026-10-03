(* Every captured tensor of a model, loaded from its .pt2 zip and from the Hub
   checkpoint its safetensors.json pins, must agree to the byte. Gated on real
   data (see pt2_safetensors_cram.t).
   argv: <model.pt2> <model directory> *)

let fail fmt =
  Format.kasprintf
    (fun s ->
      prerr_endline s;
      exit 1)
    fmt

let () =
  let pt2 = Sys.argv.(1) and dir = Sys.argv.(2) in
  let zip =
    match Pt2_archive.open_pt2 pt2 with
    | Ok a -> a
    | Error e -> fail "%a" Pt2_archive.pp_error (Err.Error.kind e)
  in
  let st =
    match Pt2_safetensors_unix.open_dir dir with
    | Ok a -> a
    | Error e -> fail "%a" Pt2_safetensors_unix.pp_error (Err.Error.kind e)
  in
  let names = Pt2_archive.weight_names zip @ Pt2_archive.constant_names zip in
  let load label archive name =
    match Pt2_archive.load_captured_tensor archive name with
    | Ok t -> t
    | Error e ->
        fail "%s %s: %a" label name Pt2_archive.pp_error (Err.Error.kind e)
  in
  let equal a b =
    let n = Pt2_storage.length a in
    n = Pt2_storage.length b
    &&
    let rec go i =
      i >= n
      || (Pt2_storage.get_uint8 a i = Pt2_storage.get_uint8 b i && go (i + 1))
    in
    go 0
  in
  List.iter
    (fun name ->
      let a = load "pt2" zip name and b = load "safetensors" st name in
      if
        not
          (a.Pt2_tensor.dtype = b.dtype
          && a.sizes = b.sizes && a.strides = b.strides
          && a.storage_offset = b.storage_offset
          && equal a.data b.data)
      then fail "%s differs" name)
    names;
  Printf.printf "%d captured tensors byte-equal\n" (List.length names)
