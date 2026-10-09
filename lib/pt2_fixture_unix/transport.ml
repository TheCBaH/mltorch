type t = url:string -> dest:string -> (unit, string) result

let curl ~url ~dest =
  let args =
    [|
      "curl";
      "--fail";
      "--silent";
      "--show-error";
      "--location";
      "--proto";
      "=https";
      "--proto-redir";
      "=https";
      "--max-redirs";
      "5";
      "--retry";
      "2";
      "--output";
      dest;
      "--";
      url;
    |]
  in
  match Unix.create_process "curl" args Unix.stdin Unix.stdout Unix.stderr with
  | exception Unix.Unix_error (e, _, _) -> Error (Unix.error_message e)
  | pid -> (
      match Unix.waitpid [] pid with
      | _, Unix.WEXITED 0 -> Ok ()
      | _, Unix.WEXITED n ->
          Error (Printf.sprintf "curl exited with status %d" n)
      | _, (Unix.WSIGNALED n | Unix.WSTOPPED n) ->
          Error (Printf.sprintf "curl was stopped by signal %d" n))
