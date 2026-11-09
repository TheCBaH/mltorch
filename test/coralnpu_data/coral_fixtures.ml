(* The archive's own guest programs on every MPACT configuration the manifest
   lists them for: the producer's fixtures, run by the profile's launch command,
   each required to exit 0 and print its pass marker. MPACT runs one in
   milliseconds where the RTL testbenches take minutes, so this is the
   functional check of the matrix, vector and bfloat16 extensions and of the
   packaged kernels; the cycle counts of the RTL are for the suites that need
   them. Three fixtures are left out because the archive's own scripts drive
   them rather than the bare launch command: [batch_limits.elf] (the
   simulator's limits), [exit_status.elf] (status modes) and [host_ram.elf]
   (16 bytes of stdin). Exits 1 when any fixture fails. *)

let harness_driven = [ "batch_limits.elf"; "exit_status.elf"; "host_ram.elf" ]
let dir = Sys.argv.(1)

let or_exit pp = function
  | Ok v -> v
  | Error e ->
      Fmt.epr "%a@." pp e;
      exit 1

let manifest =
  let text =
    In_channel.with_open_bin
      (Filename.concat dir Coral_profile.manifest_file)
      In_channel.input_all
  in
  or_exit Coral_profile.pp_error (Err.payload (Coral_manifest.of_string text))

let contains ~sub s =
  let n = String.length sub and m = String.length s in
  let rec go i = i + n <= m && (String.sub s i n = sub || go (i + 1)) in
  go 0

(* The simulator's output and exit status, killed after a minute. *)
let run argv =
  let out = Filename.temp_file "coral_fixture" ".log" in
  let fd = Unix.openfile out [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let pid =
    Unix.create_process "timeout"
      (Array.of_list ("timeout" :: "60" :: argv))
      Unix.stdin fd fd
  in
  Unix.close fd;
  let _, status = Unix.waitpid [] pid in
  let text = In_channel.with_open_bin out In_channel.input_all in
  Sys.remove out;
  (status, text)

let sha256_file path =
  let h = Pt2_sha256.create () in
  Pt2_sha256.add_string h (In_channel.with_open_bin path In_channel.input_all);
  Pt2_sha256.Digest.to_hex (Pt2_sha256.finish h)

let failed = ref false

let () =
  let mpact =
    List.filter_map
      (fun (c : Coral_manifest.Configuration.t) ->
        if String.equal c.backend "mpact" then Some c.name else None)
      manifest.configurations
  in
  List.iter
    (fun name ->
      let p =
        or_exit Coral_profile.pp_error
          (Err.payload (Coral_profile.load ~dir name))
      in
      List.iter
        (fun (f : Coral_manifest.Fixture.t) ->
          let elf = Filename.concat dir f.path in
          if
            List.mem p.configuration_id f.configuration_ids
            && not (List.mem (Filename.basename f.path) harness_driven)
          then (
            let verdict =
              if not (String.equal (sha256_file elf) f.sha256) then
                "the ELF does not match the manifest's SHA-256"
              else
                match run (Coral_profile.argv ~dir ~elf p) with
                | Unix.WEXITED 0, text when contains ~sub:f.marker text ->
                    f.marker
                | Unix.WEXITED 0, _ -> "exit 0 without " ^ f.marker
                | Unix.WEXITED n, _ -> Fmt.str "exit %d" n
                | (Unix.WSIGNALED n | Unix.WSTOPPED n), _ ->
                    Fmt.str "signal %d" n
            in
            if not (String.equal verdict f.marker) then failed := true;
            Fmt.pr "%s %s: %s@." name (Filename.basename f.path) verdict))
        manifest.fixtures)
    mpact;
  if !failed then exit 1
