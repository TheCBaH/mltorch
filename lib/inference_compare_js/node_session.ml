(* One node process running the helper. Requests go out one at a time and each
   is answered before the next is sent, so neither pipe can fill against the
   other. The process lives while any prepared body does; stderr goes to a file
   whose tail is part of any failure. *)

module W = Inference_compare_js_wire.Wire

(* The script node runs: it loads the helper, which exports [mlcjHandle], and
   serves length-prefixed messages on stdin and stdout until stdin closes. *)
let driver =
  {|const fs = require("fs");
require(process.argv[1]);
const handle = globalThis.mlcjHandle;
function readExact(n) {
  const buf = Buffer.alloc(n);
  let got = 0;
  while (got < n) {
    let r;
    try { r = fs.readSync(0, buf, got, n - got, null); }
    catch (e) { if (e.code === "EAGAIN") continue; if (e.code === "EOF") r = 0; else throw e; }
    if (r === 0) return got === 0 ? null : (() => { throw new Error("stdin ended inside a message"); })();
    got += r;
  }
  return buf;
}
function writeAll(buf) {
  let at = 0;
  while (at < buf.length) {
    try { at += fs.writeSync(1, buf, at, buf.length - at); }
    catch (e) { if (e.code === "EAGAIN") continue; throw e; }
  }
}
for (;;) {
  const head = readExact(4);
  if (head === null) break;
  const body = readExact(head.readInt32LE(0)) || Buffer.alloc(0);
  const out = handle(new Uint8Array(body.buffer, body.byteOffset, body.length));
  const len = Buffer.alloc(4);
  len.writeInt32LE(out.length, 0);
  writeAll(len);
  writeAll(Buffer.from(out.buffer, out.byteOffset, out.length));
}
|}

type t = {
  pid : int;
  to_child : out_channel;
  from_child : in_channel;
  log : string;
  mutable live : int;  (** prepared bodies *)
  mutable closed : bool;
}

let tail path =
  match open_in_bin path with
  | exception Sys_error _ -> ""
  | ic ->
      let n = in_channel_length ic in
      let k = min n 400 in
      seek_in ic (n - k);
      let s = really_input_string ic k in
      close_in ic;
      String.trim s

let start ~node ~helper =
  (try Sys.set_signal Sys.sigpipe Sys.Signal_ignore with _ -> ());
  let log = Filename.temp_file "compare_js" ".log" in
  match node with
  | [] -> Error "no node command"
  | prog :: args -> (
      try
        let in_r, in_w = Unix.pipe ~cloexec:true () in
        let out_r, out_w = Unix.pipe ~cloexec:true () in
        let err = Unix.openfile log [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
        let pid =
          Unix.create_process_env prog
            (Array.of_list ((prog :: args) @ [ "-e"; driver; helper ]))
            (Unix.environment ()) in_r out_w err
        in
        Unix.close in_r;
        Unix.close out_w;
        Unix.close err;
        Ok
          {
            pid;
            to_child = Unix.out_channel_of_descr in_w;
            from_child = Unix.in_channel_of_descr out_r;
            log;
            live = 0;
            closed = false;
          }
      with Unix.Unix_error (e, f, a) ->
        Error (Printf.sprintf "%s %s: %s" f a (Unix.error_message e)))

let status_text = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "stopped by signal %d" n

let close t =
  if not t.closed then (
    t.closed <- true;
    (try close_out t.to_child with _ -> ());
    (try close_in t.from_child with _ -> ());
    (try ignore (Unix.waitpid [] t.pid) with _ -> ());
    try Sys.remove t.log with _ -> ())

(* The helper stopped answering: what node printed and how it ended. *)
let died t why =
  let status =
    match Unix.waitpid [ Unix.WNOHANG ] t.pid with
    | 0, _ -> "still running"
    | _, st -> status_text st
    | exception Unix.Unix_error _ -> "gone"
  in
  let log = tail t.log in
  Error
    (Printf.sprintf "node helper: %s (%s)%s" why status
       (if log = "" then "" else ": " ^ log))

let request t (r : W.request) : (W.response, string) result =
  if t.closed then Error "node helper is closed"
  else
    match
      output_string t.to_child (W.frame (W.encode_request r));
      flush t.to_child
    with
    | exception Sys_error m -> died t m
    | () -> (
        match
          let head = really_input_string t.from_child 4 in
          match W.frame_length head with
          | Error m -> Error m
          | Ok n -> Ok (really_input_string t.from_child n)
        with
        | exception End_of_file -> died t "closed its output"
        | exception Sys_error m -> died t m
        | Error m -> Error ("node helper: " ^ m)
        | Ok body ->
            Result.map_error
              (fun m -> "node helper: " ^ m)
              (W.decode_response body))
