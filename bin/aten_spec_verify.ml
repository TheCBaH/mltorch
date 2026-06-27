(* CLI: verify or evaluate JSON op-specs against the ATen-vs-native harness.

   Each non-flag argument is a path to a spec file; "-", or no paths at all,
   reads one spec from stdin. A spec ({ target, seed, args }) is decoded, its
   tensor inputs are synthesized from the seed, and the op is run on both the
   ATen and native paths.

   Default mode compares the outputs and exits non-zero on any mismatch, error,
   or parse failure. With --print, the resulting output tensor(s) are printed
   from both paths (ATen and native) instead of compared. *)

let read_source path =
  if path = "-" then In_channel.input_all stdin
  else In_channel.with_open_bin path In_channel.input_all

let handle ~print path =
  match Jsont_bytesrw.decode_string Aten_op_spec.jsont (read_source path) with
  | Error e ->
      Format.printf "[spec] %s: parse error: %s@." path e;
      false
  | Ok spec ->
      if print then (
        Aten_spec_run.eval_print spec;
        true)
      else Aten_spec_run.run spec

let () =
  let print = ref false and paths = ref [] in
  Array.iteri
    (fun i a ->
      if i = 0 then ()
      else if a = "--print" then print := true
      else paths := a :: !paths)
    Sys.argv;
  (* No file arguments means read a single spec from stdin. *)
  let sources = match List.rev !paths with [] -> [ "-" ] | ps -> ps in
  let ok =
    List.fold_left (fun acc p -> handle ~print:!print p && acc) true sources
  in
  if not ok then exit 1
