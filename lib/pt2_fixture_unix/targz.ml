open Err.Syntax

type limits = {
  max_archive_bytes : int;
  max_member_bytes : int;
  max_members : int;
}

let default_limits =
  {
    max_archive_bytes = 0x2000_0000;
    max_member_bytes = 0x1000_0000;
    max_members = 4096;
  }

type member = { data : string; name : string }

let safe_name n =
  n <> ""
  && String.length n <= 255
  && n.[0] <> '/'
  && String.for_all (fun c -> c <> '\\' && c <> '\000') n
  && List.for_all
       (fun seg -> seg <> "" && seg <> "." && seg <> "..")
       (String.split_on_char '/' n)

let gzip_fail m = Err.fail (`Gzip m)

(* RFC 1952: ID1 ID2 CM FLG MTIME(4) XFL OS, then optional fields, the deflate
   stream, CRC-32 and ISIZE. Returns the offset of the deflate stream. *)
let gzip_body s =
  let n = String.length s in
  if n < 18 then gzip_fail "shorter than a gzip header and trailer"
  else if s.[0] <> '\x1f' || s.[1] <> '\x8b' then gzip_fail "bad magic"
  else if s.[2] <> '\x08' then gzip_fail "compression method is not deflate"
  else
    let flags = Char.code s.[3] in
    if flags land 0xE0 <> 0 then gzip_fail "reserved flags set"
    else
      let truncated () = gzip_fail "truncated header" in
      (* Each optional field advances [pos] or fails. *)
      let extra pos =
        if flags land 4 = 0 then Some pos
        else if pos + 2 > n then None
        else Some (pos + 2 + String.get_uint16_le s pos)
      in
      let zero_terminated bit pos =
        match pos with
        | Some p when flags land bit <> 0 && p < n ->
            Option.map (fun i -> i + 1) (String.index_from_opt s p '\000')
        | Some p when flags land bit = 0 -> Some p
        | _ -> None
      in
      let header_crc pos =
        match pos with
        | Some p when flags land 2 <> 0 -> Some (p + 2)
        | other -> other
      in
      match header_crc (zero_terminated 16 (zero_terminated 8 (extra 10))) with
      | Some pos when pos <= n - 8 -> Err.return pos
      | _ -> truncated ()

let gunzip ~max_bytes s =
  let* start = gzip_body s in
  let n = String.length s in
  let len = n - 8 - start in
  match
    Zipc_deflate.inflate_and_crc_32 ~decompressed_size:max_bytes ~start ~len s
  with
  | Error m -> gzip_fail m
  | Ok (data, crc) ->
      if not (Int32.equal crc (String.get_int32_le s (n - 8))) then
        gzip_fail "CRC-32 mismatch"
      else if
        not
          (Int32.equal
             (Int32.of_int (String.length data))
             (String.get_int32_le s (n - 4)))
      then gzip_fail "length mismatch"
      else Err.return data

let block = 512

let all_zero s pos =
  let rec go i = i = block || (s.[pos + i] = '\000' && go (i + 1)) in
  go 0

(* A NUL- or space-terminated octal field. *)
let octal s pos len =
  let stop = ref (pos + len) in
  while !stop > pos && (s.[!stop - 1] = '\000' || s.[!stop - 1] = ' ') do
    decr stop
  done;
  let start = ref pos in
  while !start < !stop && s.[!start] = ' ' do
    incr start
  done;
  if !start = !stop then None
  else
    let rec go i acc =
      if i = !stop then Some acc
      else
        match s.[i] with
        | '0' .. '7' as c ->
            if acc > max_int / 8 then None
            else go (i + 1) ((acc * 8) + Char.code c - Char.code '0')
        | _ -> None
    in
    go !start 0

let cstring s pos len =
  let stop = ref pos in
  while !stop < pos + len && s.[!stop] <> '\000' do
    incr stop
  done;
  String.sub s pos (!stop - pos)

let checksum_ok s pos =
  let sum = ref 0 in
  for i = 0 to block - 1 do
    sum :=
      !sum
      + if i >= 148 && i < 156 then Char.code ' ' else Char.code s.[pos + i]
  done;
  octal s (pos + 148) 8 = Some !sum

let members ?(limits = default_limits) gz =
  let* tar = gunzip ~max_bytes:limits.max_archive_bytes gz in
  let n = String.length tar in
  let fail kind name = Err.fail (`Tar { Fault.Tar_fault.kind; name }) in
  let seen = Hashtbl.create 32 in
  let rec go pos count acc =
    if pos + block > n then
      (* A well-formed archive ends with zero blocks, but a stream that stops
         at a member boundary is still complete. *)
      if pos = n then Err.return (List.rev acc) else fail Fault.Truncated ""
    else if all_zero tar pos then Err.return (List.rev acc)
    else if not (checksum_ok tar pos) then fail Fault.Bad_checksum ""
    else
      let name =
        let base = cstring tar pos 100 in
        let prefix =
          if String.sub tar (pos + 257) 5 = "ustar" then
            cstring tar (pos + 345) 155
          else ""
        in
        if prefix = "" then base else prefix ^ "/" ^ base
      in
      let typeflag = tar.[pos + 156] in
      match octal tar (pos + 124) 12 with
      | None -> fail Fault.Bad_size name
      | Some size -> (
          if count >= limits.max_members then
            Err.fail
              (`Limit
                 {
                   Fault.Limit_fault.what = Fault.Member_count;
                   limit = Int64.of_int limits.max_members;
                   actual = Int64.of_int (count + 1);
                 })
          else
            match typeflag with
            | '1' | '2' -> fail Fault.Link_member name
            | '0' | '\000' ->
                if String.length name > 255 then fail Fault.Name_too_long name
                else if name <> "" && name.[0] = '/' then
                  fail Fault.Absolute_name name
                else if not (safe_name name) then fail Fault.Unsafe_name name
                else if Hashtbl.mem seen name then
                  fail Fault.Duplicate_name name
                else if size > limits.max_member_bytes then
                  Err.fail
                    (`Limit
                       {
                         Fault.Limit_fault.what = Fault.Member_bytes;
                         limit = Int64.of_int limits.max_member_bytes;
                         actual = Int64.of_int size;
                       })
                else if size > n - (pos + block) then fail Fault.Truncated name
                else begin
                  Hashtbl.add seen name ();
                  let data = String.sub tar (pos + block) size in
                  let padded = (size + block - 1) / block * block in
                  go (pos + block + padded) (count + 1) ({ data; name } :: acc)
                end
            | _ -> fail Fault.Unsupported_member name)
  in
  go 0 0 []
