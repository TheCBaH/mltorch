let online () =
  match In_channel.with_open_text "/proc/cpuinfo" In_channel.input_all with
  | text ->
      let n =
        List.length
          (List.filter
             (fun l -> String.length l >= 9 && String.sub l 0 9 = "processor")
             (String.split_on_char '\n' text))
      in
      max 1 n
  | exception Sys_error _ -> 1

let pool () =
  match Option.bind (Sys.getenv_opt "MLTORCH_CORAL_CPUS") int_of_string_opt with
  | Some n when n >= 1 -> n
  | _ -> online ()

let dir () =
  let d = Printf.sprintf "/tmp/mltorch-coral-cpus-%d" (Unix.getuid ()) in
  (try Unix.mkdir d 0o700 with Unix.Unix_error (Unix.EEXIST, _, _) -> ());
  d

(* every slot free now, locked; or none held *)
let try_take ~dir ~size n =
  let rec go k held =
    if List.length held = n then Some held
    else if k >= size then (
      List.iter Unix.close held;
      None)
    else
      let fd =
        Unix.openfile
          (Filename.concat dir (string_of_int k))
          [ Unix.O_RDWR; Unix.O_CREAT; Unix.O_CLOEXEC ]
          0o600
      in
      match Unix.lockf fd Unix.F_TLOCK 0 with
      | () -> go (k + 1) (fd :: held)
      | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EACCES), _, _) ->
          Unix.close fd;
          go (k + 1) held
  in
  go 0 []

let with_cpus n f =
  let size = pool () in
  let n = max 1 (min n size) in
  let dir = dir () in
  let rec take () =
    match try_take ~dir ~size n with
    | Some held -> held
    | None ->
        Unix.sleepf 0.25;
        take ()
  in
  let held = take () in
  (* closing a descriptor drops its record lock *)
  Fun.protect ~finally:(fun () -> List.iter Unix.close held) f
