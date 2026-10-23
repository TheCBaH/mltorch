type t = Unix.file_descr

exception Failed of string

let failed fmt = Printf.ksprintf (fun s -> raise (Failed s)) fmt

let connect ~port =
  let s = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  match Unix.connect s (Unix.ADDR_INET (Unix.inet_addr_loopback, port)) with
  | () -> s
  | exception Unix.Unix_error (e, _, _) ->
      Unix.close s;
      failed "connect to port %d: %s" port (Unix.error_message e)

let close s = try Unix.close s with Unix.Unix_error _ -> ()

let send s b =
  let n = Bytes.length b in
  let rec go o =
    if o < n then
      match Unix.write s b o (n - o) with
      | k -> go (o + k)
      | exception Unix.Unix_error (e, _, _) ->
          failed "send: %s" (Unix.error_message e)
  in
  go 0

let recv s n =
  let b = Bytes.create n in
  let rec go o =
    if o < n then
      match Unix.read s b o (n - o) with
      | 0 -> failed "the server closed the connection"
      | k -> go (o + k)
      | exception Unix.Unix_error (e, _, _) ->
          failed "receive: %s" (Unix.error_message e)
  in
  go 0;
  Bytes.unsafe_to_string b

(* The server's command codes. *)
let idle_clocking = 2
let v2_write = 8
let v2_read = 9

let command s ~code ?(addr = 0L) ?(count = 0) ?(payload = "") () =
  let h = Bytes.create 17 in
  Bytes.set_uint8 h 0 code;
  Bytes.set_int32_le h 1 (Int64.to_int32 addr);
  Bytes.set_int64_le h 5 0L;
  Bytes.set_int32_le h 13 (Int32.of_int count);
  send s h;
  send s (Bytes.unsafe_of_string payload);
  let r = recv s 9 in
  if Char.code r.[8] = 0 then failed "command %d at 0x%08Lx refused" code addr

let idle s cycles = command s ~code:idle_clocking ~count:cycles ()

let aligned addr =
  if Int64.rem addr 16L <> 0L then failed "0x%08Lx is not beat-aligned" addr

(* One beat per command: the server puts a framing byte between the beats of a
   longer read, which the release's loader never meets because it reads one
   beat at a time too. *)
let read s addr ~beats =
  aligned addr;
  String.concat ""
    (List.init (max 0 beats) (fun k ->
         command s ~code:v2_read
           ~addr:(Int64.add addr (Int64.of_int (16 * k)))
           ();
         recv s 16))

let write s addr bytes =
  aligned addr;
  let n = String.length bytes in
  if n mod 16 <> 0 then failed "%d bytes are not whole beats" n;
  if n > 0 then
    command s ~code:v2_write ~addr ~count:((n / 16) - 1) ~payload:bytes ()

let write_word s addr v =
  let line = Int64.logand addr (Int64.lognot 15L) in
  let b = Bytes.of_string (read s line ~beats:1) in
  Bytes.set_int32_le b (Int64.to_int (Int64.sub addr line)) v;
  write s line (Bytes.unsafe_to_string b)

let start s ~csr ~pc =
  write_word s (Int64.add csr 4L) (Int64.to_int32 pc);
  write_word s csr 1l;
  write_word s csr 0l
