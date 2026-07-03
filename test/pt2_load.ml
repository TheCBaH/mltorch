(* End-to-end loader smoke test: open a real .pt2, load a weight and a sample
   input image, and bridge both into ATen tensors. Driven by the pt2_load_cram
   test (see `make pt2.runtest`); takes the .pt2 path and a sample image path. *)

let pp_ints a = String.concat "; " (Array.to_list (Array.map string_of_int a))

let main () =
  let open Core.Syntax in
  let pt2_path = Sys.argv.(1) in
  let image_path = Sys.argv.(2) in
  let* m = Pt2_archive.open_pt2 pt2_path in
  Printf.printf "weights: %d\n" (List.length (Pt2_archive.weight_names m));

  let* w = Pt2_archive.load_weight m "conv1.weight" in
  Printf.printf "conv1.weight: %s\n" (Format.asprintf "%a" Pt2_tensor.pp w);
  let wa = Pt2_aten.to_tensor w in
  Printf.printf "conv1.weight (aten): [%s] contiguous=%b\n"
    (pp_ints (Aten_tensor.shape wa))
    (Aten_tensor.is_contiguous wa);

  let* img = Pt2_archive.load_pt image_path in
  Printf.printf "image: %s contiguous=%b\n"
    (Format.asprintf "%a" Pt2_tensor.pp img)
    (Pt2_tensor.is_contiguous img);
  let ia = Pt2_aten.to_tensor img in
  Printf.printf "image (aten): [%s] contiguous=%b\n"
    (pp_ints (Aten_tensor.shape ia))
    (Aten_tensor.is_contiguous ia);
  Core.return ()

let () =
  match main () with
  | Ok () -> ()
  | Error e ->
      Format.eprintf "%a@." (Core.Error.pp Pt2_archive.pp_error) e;
      exit 1
