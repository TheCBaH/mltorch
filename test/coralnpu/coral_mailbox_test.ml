(* The subsystem's result record: the hash the guest computes, and records
   decoded as the host reads them back. *)

module M = Coral_mailbox

let words l =
  let b = Bytes.create (4 * List.length l) in
  List.iteri (fun i w -> Bytes.set_int32_le b (4 * i) w) l;
  Bytes.to_string b

let%expect_test "hash" =
  let show what s = Fmt.pr "%-24s %08lx@." what (M.hash s) in
  show "empty" "";
  show "one zero word" (words [ 0l ]);
  show "1.0 2.0" (words [ 0x3f80_0000l; 0x4000_0000l ]);
  show "-1.0 -2.0" (words [ 0xbf80_0000l; 0xc000_0000l ]);
  show "2.0 1.0" (words [ 0x4000_0000l; 0x3f80_0000l ]);
  show "tail bytes" "\001\002\003\004\005";
  [%expect
    {|
    empty                    811c9dc5
    one zero word            050c5707
    1.0 2.0                  757aa7f1
    -1.0 -2.0                73e7aacb
    2.0 1.0                  74496896
    tail bytes               8f1603e6 |}]

let record ?(magic = M.magic_result) f =
  let b = Bytes.make M.size '\000' in
  Bytes.blit_string magic 0 b 0 8;
  f b;
  Bytes.set_int32_le b M.done_offset M.done_word;
  Bytes.to_string b

let show s =
  match M.decode s with
  | Ok (Result r) ->
      Fmt.pr
        "result: digest %08lx status %ld mcycle %Ld minstret %Ld hashes [%a]@."
        r.digest r.status r.mcycle r.minstret
        Fmt.(list ~sep:sp (fun ppf -> pf ppf "%08lx"))
        r.hashes
  | Ok (Trap t) ->
      Fmt.pr "trap: mcause %ld mepc %08lx mtval %08lx@." t.mcause t.mepc t.mtval
  | Error m -> Fmt.pr "malformed: %a@." M.pp_malformed m

let%expect_test "records" =
  let r =
    record (fun b ->
        Bytes.set_int32_le b 8 0x1234_5678l;
        Bytes.set_int32_le b 12 7l;
        Bytes.set_int64_le b 16 0x1_0000_0002L;
        Bytes.set_int64_le b 24 99L;
        Bytes.set_int32_le b 32 2l;
        Bytes.set_int32_le b 36 0xAAl;
        Bytes.set_int32_le b 40 0xBBl)
  in
  show r;
  Fmt.pr "done %b, not done %b@."
    (M.is_done (String.sub r 240 16) (M.done_offset - 240))
    (M.is_done (String.make 16 '\000') (M.done_offset - 240));
  show
    (record ~magic:M.magic_trap (fun b ->
         Bytes.set_int32_le b 12 2l;
         Bytes.set_int32_le b 16 0x100l;
         Bytes.set_int32_le b 20 0x7l));
  show (record (fun b -> Bytes.set_int32_le b 32 55l));
  show (record ~magic:"CNPUXXXX" ignore);
  show (String.make 16 '\000');
  [%expect
    {|
    result: digest 12345678 status 7 mcycle 4294967298 minstret 99 hashes [000000aa
    000000bb]
    done true, not done false
    trap: mcause 2 mepc 00000100 mtval 00000007
    malformed: 55 hashes, at most 54 fit
    malformed: magic "CNPUXXXX"
    malformed: 16 bytes, not 256 |}]
