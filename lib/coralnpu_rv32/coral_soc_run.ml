module Run_error = struct
  type t =
    | Digest_mismatch of int32
    | Guest_trap of Coral_run.Trap.t
    | Mailbox_malformed of Coral_mailbox.Malformed.t
    | No_core_csr
    | Simulator_crashed of { status : string; tail : string }
    | Simulator_missing of string
    | Spi_failed of string
    | Timeout

  let pp ppf = function
    | Digest_mismatch d ->
        Fmt.pf ppf "the guest reports layout digest %08lx, not the image's" d
    | Guest_trap { Coral_run.Trap.mcause; mepc; mtval } ->
        Fmt.pf ppf "guest trap: mcause %ld, mepc 0x%08lx, mtval 0x%08lx" mcause
          mepc mtval
    | Mailbox_malformed m ->
        Fmt.pf ppf "malformed mailbox (%a)" Coral_mailbox.pp_malformed m
    | No_core_csr -> Fmt.string ppf "the profile has no core CSR region"
    | Simulator_crashed { status; tail } ->
        Fmt.pf ppf "the simulator %s before the guest was done; it said: %S"
          status tail
    | Simulator_missing path -> Fmt.pf ppf "no simulator at %s" path
    | Spi_failed why -> Fmt.pf ppf "the SPI bridge failed: %s" why
    | Timeout -> Fmt.string ppf "the guest was not done by the deadline"
end

type outcome = {
  argv : string list;
  hashes : (int * int32) list;
  mcycle : int64;
  minstret : int64;
  status : int32;
  text : string;
  wall : float;
}

let matches o k bytes =
  List.assoc_opt k o.hashes = Some (Coral_mailbox.hash bytes)

let listening = "DPI: Server listening on port"

(* The camera model of the SoC opens this image in its working directory and
   complains when it is missing. *)
let camera_image = "grey_bars_320x240.raw"

(* Every slot's bytes as a section of its own. *)
let contents (image : Coral_image.t) ~bound =
  List.mapi
    (fun k (s : Coral_layout.Placed_slot.t) ->
      let size = Int64.to_int s.slot.size in
      let bytes =
        if Coral_layout.Placement.read s.slot.placement then
          Option.value (bound k) ~default:""
        else ""
      in
      let n = min size (String.length bytes) in
      ( Printf.sprintf ".slot.%d" k,
        s.address,
        String.sub bytes 0 n ^ String.make (size - n) '\000' ))
    image.layout.slots

let free_port () =
  let s = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close s)
    (fun () ->
      Unix.bind s (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
      match Unix.getsockname s with
      | Unix.ADDR_INET (_, port) -> port
      | Unix.ADDR_UNIX _ -> assert false)

let read_file path =
  try In_channel.with_open_bin path In_channel.input_all
  with Sys_error _ -> ""

let tail_bytes = 4096

let tail s =
  let n = String.length s in
  if n <= tail_bytes then s else String.sub s (n - tail_bytes) tail_bytes

let contains s needle =
  let n = String.length needle in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = needle || go (i + 1))
  in
  go 0

let rec remove_tree path =
  match Unix.lstat path with
  | { Unix.st_kind = S_DIR; _ } ->
      Array.iter
        (fun n -> remove_tree (Filename.concat path n))
        (Sys.readdir path);
      Unix.rmdir path
  | _ -> Unix.unlink path
  | exception Unix.Unix_error _ -> ()

let spawn ~cwd ~env argv log =
  match Unix.fork () with
  | 0 -> (
      try
        ignore (Unix.setsid ());
        Unix.chdir cwd;
        let fd =
          Unix.openfile log [ O_WRONLY; O_CREAT; O_TRUNC; O_CLOEXEC ] 0o600
        in
        let null = Unix.openfile "/dev/null" [ O_RDONLY; O_CLOEXEC ] 0 in
        Unix.dup2 ~cloexec:false null Unix.stdin;
        Unix.dup2 ~cloexec:false fd Unix.stdout;
        Unix.dup2 ~cloexec:false fd Unix.stderr;
        Unix.execve (List.hd argv) (Array.of_list argv) env
      with e ->
        let msg = Printexc.to_string e ^ "\n" in
        ignore (Unix.write_substring Unix.stderr msg 0 (String.length msg));
        Unix._exit 127)
  | pid -> pid

let describe = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "was killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "stopped on signal %d" n

(* Polling interval for the simulator's log and the mailbox. *)
let poll = 0.05

let run ?(deadline = 600.) ~dir (image : Coral_image.t) ~bound =
  let open Err.Syntax in
  let dir =
    if Filename.is_relative dir then Filename.concat (Sys.getcwd ()) dir
    else dir
  in
  let* csr =
    List.find_map
      (fun (r : Coral_profile.Region.t) ->
        if r.role = Csr then Some r.base else None)
      image.profile.regions
    |> Err.of_option Run_error.No_core_csr
  in
  let pc = Option.value image.image.entry ~default:0L in
  let elf =
    Coral_elf.write ~contents:(contents image ~bound) image.profile image.layout
      image.image
  in
  match Err.payload elf with
  | Error e ->
      (* The image was built from this layout, so its sections were placed. *)
      Fmt.invalid_arg "Coral_soc_run: %a" Coral_elf.pp_error e
  | Ok elf_bytes ->
      let work = Filename.temp_file "mltorch_coral_soc_" "" in
      Sys.remove work;
      Sys.mkdir work 0o700;
      Fun.protect
        ~finally:(fun () -> remove_tree work)
        (fun () ->
          let elf = Filename.concat work "guest.elf" in
          Out_channel.with_open_bin elf (fun oc -> output_string oc elf_bytes);
          let camera =
            Filename.concat (Filename.concat dir "share") camera_image
          in
          if Sys.file_exists camera then
            Unix.symlink camera (Filename.concat work camera_image);
          let argv = Coral_profile.argv ~dir ~elf image.profile in
          let program = List.hd argv in
          if not (Sys.file_exists program) then
            Err.fail (Run_error.Simulator_missing program)
          else
            let port = free_port () in
            let env =
              Array.append
                [| Printf.sprintf "SPI_DPI_PORT=%d" port |]
                (Unix.environment ())
            in
            let log = Filename.concat work "sim.out" in
            let pid = spawn ~cwd:work ~env argv log in
            let stop = Unix.gettimeofday () +. deadline in
            let exited = ref None in
            let alive () =
              match !exited with
              | Some _ -> false
              | None -> (
                  match Unix.waitpid [ WNOHANG ] pid with
                  | 0, _ -> true
                  | _, status ->
                      exited := Some status;
                      false
                  | exception Unix.Unix_error (Unix.EINTR, _, _) -> true)
            in
            let crashed () =
              let status =
                match !exited with Some s -> describe s | None -> "stopped"
              in
              Err.fail
                (Run_error.Simulator_crashed
                   { status; tail = tail (read_file log) })
            in
            Fun.protect
              ~finally:(fun () ->
                (try Unix.kill (-pid) Sys.sigkill with Unix.Unix_error _ -> ());
                if !exited = None then
                  try ignore (Unix.waitpid [] pid)
                  with Unix.Unix_error _ -> ())
              (fun () ->
                let rec wait_listening () =
                  if contains (read_file log) listening then Err.return ()
                  else if not (alive ()) then crashed ()
                  else if Unix.gettimeofday () > stop then
                    Err.fail Run_error.Timeout
                  else (
                    Unix.sleepf poll;
                    wait_listening ())
                in
                let* () = wait_listening () in
                match Coral_spi.connect ~port with
                | exception Coral_spi.Failed why ->
                    Err.fail (Run_error.Spi_failed why)
                | spi -> (
                    Fun.protect ~finally:(fun () -> Coral_spi.close spi)
                    @@ fun () ->
                    let mailbox = image.layout.htif in
                    let done_line =
                      Int64.add mailbox
                        (Int64.of_int (Coral_mailbox.done_offset land lnot 15))
                    in
                    let done_at = Coral_mailbox.done_offset land 15 in
                    match
                      Coral_spi.idle spi 20;
                      Coral_spi.start spi ~csr ~pc;
                      let t0 = Unix.gettimeofday () in
                      let rec wait () =
                        if
                          Coral_mailbox.is_done
                            (Coral_spi.read spi done_line ~beats:1)
                            done_at
                        then `Done (Unix.gettimeofday () -. t0)
                        else if not (alive ()) then `Crashed
                        else if Unix.gettimeofday () > stop then `Timeout
                        else (
                          Unix.sleepf poll;
                          wait ())
                      in
                      match wait () with
                      | `Done wall ->
                          `Done
                            ( wall,
                              Coral_spi.read spi mailbox
                                ~beats:(Coral_mailbox.size / 16) )
                      | (`Crashed | `Timeout) as other -> other
                    with
                    | exception Coral_spi.Failed why ->
                        if alive () then Err.fail (Run_error.Spi_failed why)
                        else crashed ()
                    | `Crashed -> crashed ()
                    | `Timeout -> Err.fail Run_error.Timeout
                    | `Done (wall, record) -> (
                        let text = read_file log in
                        match Coral_mailbox.decode record with
                        | Error m -> Err.fail (Run_error.Mailbox_malformed m)
                        | Ok (Trap { mcause; mepc; mtval }) ->
                            Err.fail
                              (Run_error.Guest_trap
                                 { Coral_run.Trap.mcause; mepc; mtval })
                        | Ok (Result r) ->
                            if r.digest <> image.layout.digest then
                              Err.fail (Run_error.Digest_mismatch r.digest)
                            else
                              let written =
                                List.filteri
                                  (fun _ (s : Coral_layout.Placed_slot.t) ->
                                    Coral_layout.Placement.written
                                      s.slot.placement)
                                  image.layout.slots
                                |> List.length
                              in
                              if List.length r.hashes <> written then
                                Err.fail
                                  (Run_error.Mailbox_malformed
                                     (Hash_count (List.length r.hashes)))
                              else
                                let indices =
                                  List.concat
                                    (List.mapi
                                       (fun k (s : Coral_layout.Placed_slot.t)
                                          ->
                                         if
                                           Coral_layout.Placement.written
                                             s.slot.placement
                                         then [ k ]
                                         else [])
                                       image.layout.slots)
                                in
                                Err.return
                                  {
                                    argv;
                                    hashes = List.combine indices r.hashes;
                                    mcycle = r.mcycle;
                                    minstret = r.minstret;
                                    status = r.status;
                                    text;
                                    wall;
                                  }))))
