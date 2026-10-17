(* Decodes the release manifest under [dir] (every configuration ID
   recomputed), then loads each baseline profile (its simulator's SHA-256
   verified) and prints it with the command one run would use. Exits 1 on the
   first refusal. *)

let () =
  let dir = Sys.argv.(1) in
  let or_exit = function
    | Ok v -> v
    | Error e ->
        Fmt.epr "%a@." Coral_profile.pp_error e;
        exit 1
  in
  let text =
    In_channel.with_open_bin
      (Filename.concat dir Coral_profile.manifest_file)
      In_channel.input_all
  in
  let m = or_exit (Err.payload (Coral_manifest.of_string text)) in
  Fmt.pr "%d configurations, every ID recomputed@."
    (List.length m.configurations);
  List.iter
    (fun name ->
      let p = or_exit (Err.payload (Coral_profile.load ~dir name)) in
      let argv = Coral_profile.argv ~dir:"$DIR" ~elf:"guest.elf" p in
      Fmt.pr "%a@.%s@." Coral_profile.pp p (String.concat " " argv))
    Coral_profile.baseline
