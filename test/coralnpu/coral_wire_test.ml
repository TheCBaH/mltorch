module W = Coral_wire

let digest = 0x90bf9718l
let payload = "\x01\x02\x03\x04\x05\x06\x07\x08"

let frame =
  W.result ~digest ~status:7l ~mcycle:0x1_0000_0002L ~minstret:42L payload

let show out =
  match W.scan ~digest ~payload_size:8L out with
  | Error m -> Fmt.pr "malformed: %a@." W.Malformed.pp m
  | Ok { W.Scan.before; after; frame } -> (
      Fmt.pr "before %S after %S@." before after;
      match frame with
      | W.Frame.Result { status; mcycle; minstret; payload } ->
          Fmt.pr "result status %ld mcycle %Ld minstret %Ld payload %S@." status
            mcycle minstret payload
      | W.Frame.Trap { mcause; mepc; mtval } ->
          Fmt.pr "trap mcause %ld mepc %08lx mtval %08lx@." mcause mepc mtval)

let hex s =
  String.to_seq s
  |> Seq.map (fun c -> Printf.sprintf "%02x" (Char.code c))
  |> List.of_seq |> String.concat ""

let%expect_test "the input stream" =
  print_endline (hex (W.input ~digest "abc"));
  [%expect {| 434e5055494e3031010000001897bf9003000000616263434e5055454e4431 |}]

let%expect_test "frames among simulator text" =
  show ("banner\n" ^ frame ^ "exit 0\n");
  show
    ("Simulation done\n"
    ^ W.trap ~mcause:2l ~mepc:0x124l ~mtval:0x02000073l
    ^ frame);
  [%expect
    {|
    before "banner\n" after "exit 0\n"
    result status 7 mcycle 4294967298 minstret 42 payload "\001\002\003\004\005\006\007\b"
    before "Simulation done\n" after "CNPUOUT1\001\000\000\000\024\151\191\144\007\000\000\000\002\000\000\000\001\000\000\000*\000\000\000\000\000\000\000\b\000\000\000\001\002\003\004\005\006\007\bCNPUEND1"
    trap mcause 2 mepc 00000124 mtval 02000073 |}]

(* Each breakage is its own row; none decodes as a result. *)
let%expect_test "malformed output" =
  let n = String.length frame in
  let set32 s at v =
    let b = Bytes.of_string s in
    Bytes.set_int32_le b at v;
    Bytes.to_string b
  in
  show "banner only\n";
  show (String.sub frame 0 (n - 3));
  show (String.sub frame 0 20);
  show (String.sub frame 0 (n - 8) ^ "CNPUEND2");
  show (set32 frame 8 2l);
  show (set32 frame 12 0x90bf9719l);
  show (set32 frame 36 9l);
  (* MPACT's banner landing inside the payload. *)
  show
    (String.sub frame 0 44 ^ "Simulation done\n" ^ String.sub frame 44 (n - 44));
  [%expect
    {|
    malformed: no frame in the output
    malformed: the output ends inside a frame
    malformed: the output ends inside a frame
    malformed: the frame's trailer is missing
    malformed: frame version 2
    malformed: layout digest 90bf9719, expected 90bf9718
    malformed: payload of 9 bytes, expected 8
    malformed: the frame's trailer is missing |}]
