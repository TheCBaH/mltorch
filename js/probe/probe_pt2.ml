(* Section 6, its own entry point: reading a real .pt2 archive and running
   inference on it. Not melange-reachable, and gated on downloaded weights,
   which is why it is not a section of native_probe -- a run that needs a 12MB
   model cannot share an entry point with one that needs nothing.

   Everything here goes through the in-memory seam ([Pt2_archive.of_string],
   [pt_of_string]) rather than the path-based entry points. Under node either
   would work, but the in-memory one is what a browser has, so it is the one
   worth proving.

   The engine underneath is the pure-OCaml one: native_interp depends on neither
   aten nor interp, so nothing here touches ctypes or libtorch. *)

let read_file path =
  try In_channel.with_open_bin path In_channel.input_all
  with Sys_error msg -> failwith ("probe: cannot read " ^ path ^ ": " ^ msg)

(* [Core.or_raise] for a [Core.result], per [[printer_conventions]]. Raising
   rather than printing is the whole contract of this harness: both backends run
   this same source, so an error that reproduced on each would print identically,
   diff clean and exit 0. See [[testing_strategy]]. *)
let ok what pp r =
  Core.or_raise (fun ppf e -> Fmt.pf ppf "probe: %s: %a" what pp e) r

let run ~pt2 ~input =
  print_endline "=== pt2-inference ===";
  let archive_bytes = read_file pt2 in
  Printf.printf "archive-bytes %d\n" (String.length archive_bytes);
  let archive =
    ok "open archive" Pt2_archive.pp_error
      (Pt2_archive.of_string ~name:pt2 archive_bytes)
  in
  let lowered =
    ok "lower" Native_interp.pp_error (Native_interp.lower_archive archive)
  in
  Printf.printf "lowered-nodes %d\n"
    (List.length lowered.Pt2_native_graph.graph.Graph_ir.Graph.nodes);
  let tensor =
    ok "load input" Pt2_archive.pp_error
      (Pt2_archive.pt_of_string ~name:input (read_file input))
  in
  Printf.printf "input-shape [%s]\n"
    (String.concat "; " (List.map string_of_int tensor.Pt2_tensor.sizes));
  let outputs =
    ok "run" Native_interp.pp_error (Native_interp.run archive ~input:tensor)
  in
  Printf.printf "outputs %d\n" (List.length outputs);
  List.iteri
    (fun i (Tensor.Tensor t as packed) ->
      let shape = t.Tensor.shape in
      let numel = (Vec6.numel shape :> int) in
      Printf.printf "output %d shape %s numel %d\n" i
        (Format.asprintf "%a" Vec6.pp_shape shape)
        numel;
      (* The FULL payload, as f32 hex bit patterns.
         This is the actual check, and it has to reach stdout: the only
         cross-backend comparison that exists is the Makefile's `diff -u` of the
         two runs, so a "bit-exact" boolean computed inside one run would be
         comparing the run against itself and proving nothing. (That differs
         from probe_tensor_json, whose flag is meaningful because it is a
         round-trip within one run.)
         Hex rather than decimal so the comparison is over bits, not over either
         backend's float formatting; eight per line so a divergence localizes to
         a handful of values instead of one enormous line. *)
      let values = ref [] in
      Vec6.iter shape (fun coord ->
          values := Tensor.read packed coord :: !values);
      let values = Array.of_list (List.rev !values) in
      Array.iteri
        (fun j v ->
          if j mod 8 = 0 then Printf.printf "\n  %5d" j;
          Printf.printf " %s" (Walk_core.Float32.to_hex v))
        values;
      print_newline ();
      (* A readable summary AFTER the bits, never instead of them: argmax is
         invariant to any low-order difference the hex dump would catch. *)
      let top = Array.to_list (Array.mapi (fun j v -> (v, j)) values) in
      let top =
        List.sort (fun (a, _) (b, _) -> Float.compare b a) top
        |> List.filteri (fun k _ -> k < 5)
      in
      Printf.printf "  top5 %s\n"
        (String.concat " "
           (List.map
              (fun (v, j) ->
                Printf.sprintf "%d:%s" j (Walk_core.Float32.to_hex v))
              top)))
    outputs
