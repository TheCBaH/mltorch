type status = Exited of int | Signaled of int

let pp_status ppf = function
  | Exited n -> Format.fprintf ppf "exited with status %d" n
  | Signaled s -> Format.fprintf ppf "killed by signal %d" s

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let write_file path text =
  let oc = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out_noerr oc)
    (fun () -> output_string oc text)

let temp_dir prefix =
  let base = Filename.get_temp_dir_name () in
  let rec attempt n =
    let dir =
      Filename.concat base (Printf.sprintf "%s-%d-%d" prefix (Unix.getpid ()) n)
    in
    match Unix.mkdir dir 0o700 with
    | () -> dir
    | exception Unix.Unix_error (Unix.EEXIST, _, _) -> attempt (n + 1)
  in
  attempt 0

let run ?cwd argv =
  match argv with
  | [] -> Error "empty argument vector"
  | prog :: _ ->
      let log = Filename.temp_file "c_proc" ".log" in
      let out = Unix.openfile log [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
      let null = Unix.openfile "/dev/null" [ Unix.O_RDONLY ] 0 in
      let saved = Sys.getcwd () in
      Fun.protect
        ~finally:(fun () ->
          Unix.close out;
          Unix.close null;
          (try Sys.remove log with Sys_error _ -> ());
          if cwd <> None then Sys.chdir saved)
        (fun () ->
          Option.iter Sys.chdir cwd;
          match Unix.create_process prog (Array.of_list argv) null out out with
          | exception Unix.Unix_error (e, _, _) ->
              Error (prog ^ ": " ^ Unix.error_message e)
          | pid -> (
              let rec wait () =
                match Unix.waitpid [] pid with
                | _, st -> st
                | exception Unix.Unix_error (Unix.EINTR, _, _) -> wait ()
              in
              let status =
                match wait () with
                | Unix.WEXITED n -> Exited n
                | Unix.WSIGNALED s | Unix.WSTOPPED s -> Signaled s
              in
              match read_file log with
              | text -> Ok (status, text)
              | exception Sys_error m -> Error m))

let rec remove_tree path =
  match Unix.lstat path with
  | exception Unix.Unix_error _ -> ()
  | { Unix.st_kind = Unix.S_DIR; _ } -> (
      Array.iter
        (fun f -> remove_tree (Filename.concat path f))
        (Sys.readdir path);
      try Unix.rmdir path with Unix.Unix_error _ -> ())
  | _ -> ( try Sys.remove path with Sys_error _ -> ())
