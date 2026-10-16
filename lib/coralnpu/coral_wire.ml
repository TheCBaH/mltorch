let version = 1l
let input_magic = "CNPUIN01"
let result_magic = "CNPUOUT1"
let trap_magic = "CNPUTRP1"
let trailer = "CNPUEND1"

let u32 b v =
  let s = Bytes.create 4 in
  Bytes.set_int32_le s 0 v;
  Buffer.add_bytes b s

let u64 b v =
  let s = Bytes.create 8 in
  Bytes.set_int64_le s 0 v;
  Buffer.add_bytes b s

let length s = Int32.of_int (String.length s)

let input ~digest payload =
  let b = Buffer.create (String.length payload + 32) in
  Buffer.add_string b input_magic;
  u32 b version;
  u32 b digest;
  u32 b (length payload);
  Buffer.add_string b payload;
  Buffer.add_string b trailer;
  Buffer.contents b

module Frame = struct
  type t =
    | Result of {
        mcycle : int64;
        minstret : int64;
        payload : string;
        status : int32;
      }
    | Trap of { mcause : int32; mepc : int32; mtval : int32 }
end

let result ~digest ~status ~mcycle ~minstret payload =
  let b = Buffer.create (String.length payload + 48) in
  Buffer.add_string b result_magic;
  u32 b version;
  u32 b digest;
  u32 b status;
  u64 b mcycle;
  u64 b minstret;
  u32 b (length payload);
  Buffer.add_string b payload;
  Buffer.add_string b trailer;
  Buffer.contents b

let trap ~mcause ~mepc ~mtval =
  let b = Buffer.create 32 in
  Buffer.add_string b trap_magic;
  u32 b version;
  u32 b mcause;
  u32 b mepc;
  u32 b mtval;
  Buffer.add_string b trailer;
  Buffer.contents b

module Malformed = struct
  type t =
    | Bad_trailer
    | Digest_mismatch of { actual : int32; expected : int32 }
    | No_frame
    | Payload_size of { actual : int64; expected : int64 }
    | Truncated
    | Version of int32

  let pp ppf = function
    | Bad_trailer -> Fmt.string ppf "the frame's trailer is missing"
    | Digest_mismatch { actual; expected } ->
        Fmt.pf ppf "layout digest %08lx, expected %08lx" actual expected
    | No_frame -> Fmt.string ppf "no frame in the output"
    | Payload_size { actual; expected } ->
        Fmt.pf ppf "payload of %Ld bytes, expected %Ld" actual expected
    | Truncated -> Fmt.string ppf "the output ends inside a frame"
    | Version v -> Fmt.pf ppf "frame version %ld" v
end

module Scan = struct
  type t = { after : string; before : string; frame : Frame.t }
end

(* The first index of [needle] in [s]. *)
let find s needle =
  let n = String.length needle in
  let rec go i =
    if i + n > String.length s then None
    else if String.sub s i n = needle then Some i
    else go (i + 1)
  in
  go 0

(* Magic, version, mcause, mepc, mtval; and magic, version, digest, status,
   mcycle, minstret, length. *)
let trap_header = 24
let result_header = 40

let scan ~digest ~payload_size out =
  let start =
    match (find out result_magic, find out trap_magic) with
    | Some r, Some t -> Some (min r t)
    | (Some _ as i), None | None, (Some _ as i) -> i
    | None, None -> None
  in
  let len = String.length out in
  let u32 start at = String.get_int32_le out (start + at) in
  let finish ~start ~stop frame =
    if stop + 8 > len then Error Malformed.Truncated
    else if String.sub out stop 8 <> trailer then Error Malformed.Bad_trailer
    else
      Ok
        {
          Scan.before = String.sub out 0 start;
          frame;
          after = String.sub out (stop + 8) (len - stop - 8);
        }
  in
  match start with
  | None -> Error Malformed.No_frame
  | Some start when String.sub out start 8 = trap_magic ->
      if start + trap_header > len then Error Malformed.Truncated
      else if u32 start 8 <> version then
        Error (Malformed.Version (u32 start 8))
      else
        finish ~start ~stop:(start + trap_header)
          (Frame.Trap
             {
               mcause = u32 start 12;
               mepc = u32 start 16;
               mtval = u32 start 20;
             })
  | Some start ->
      if start + result_header > len then Error Malformed.Truncated
      else if u32 start 8 <> version then
        Error (Malformed.Version (u32 start 8))
      else if u32 start 12 <> digest then
        Error
          (Malformed.Digest_mismatch
             { actual = u32 start 12; expected = digest })
      else
        let size = Int64.logand (Int64.of_int32 (u32 start 36)) 0xFFFF_FFFFL in
        if size <> payload_size then
          Error
            (Malformed.Payload_size { actual = size; expected = payload_size })
        else
          let body = start + result_header in
          let stop = body + Int64.to_int size in
          if stop > len then Error Malformed.Truncated
          else
            finish ~start ~stop
              (Frame.Result
                 {
                   status = u32 start 16;
                   mcycle = String.get_int64_le out (start + 20);
                   minstret = String.get_int64_le out (start + 28);
                   payload = String.sub out body (Int64.to_int size);
                 })
