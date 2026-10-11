(* Small filesystem helpers. Every failure is a value; nothing here raises on
   an I/O error. Creation of evidence files is exclusive and finalization is a
   rename, so a half-written file never has its final name. *)

type error = { path : string; message : string }

let pp_error ppf e = Fmt.pf ppf "%s: %s" e.path e.message

let guard path f =
  try Ok (f ()) with
  | Sys_error m | Failure m -> Error { path; message = m }
  | Unix.Unix_error (e, _, _) -> Error { path; message = Unix.error_message e }

let read path =
  guard path (fun () ->
      let ic = open_in_bin path in
      Fun.protect
        ~finally:(fun () -> close_in_noerr ic)
        (fun () -> really_input_string ic (in_channel_length ic)))

let exists path = Sys.file_exists path

(* [O_EXCL]: refuses to overwrite, so two runs cannot share a file. *)
let write_exclusive path contents =
  guard path (fun () ->
      let fd =
        Unix.openfile path [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_EXCL ] 0o644
      in
      let oc = Unix.out_channel_of_descr fd in
      Fun.protect
        ~finally:(fun () -> close_out_noerr oc)
        (fun () -> output_string oc contents))

(* Write beside the destination, then rename: readers see the old file or the
   whole new one. *)
let write_atomic path contents =
  let tmp = path ^ ".tmp" in
  let ( let* ) = Result.bind in
  let* () = write_exclusive tmp contents in
  guard path (fun () -> Unix.rename tmp path)

let mkdir_exclusive path = guard path (fun () -> Unix.mkdir path 0o755)

let mkdir_p path =
  let rec go p =
    if p = "" || p = "/" || Sys.file_exists p then ()
    else (
      go (Filename.dirname p);
      Unix.mkdir p 0o755)
  in
  guard path (fun () -> go path)

let resolve ~base path =
  if Filename.is_relative path then Filename.concat base path else path
