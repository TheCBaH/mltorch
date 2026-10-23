let size = 256
let done_offset = 252
let done_word = 0x454E_4F44l
let hashes_offset = 36
let max_hashes = (done_offset - hashes_offset) / 4
let magic_result = "CNPUOUT1"
let magic_trap = "CNPUTRP1"

let hash s =
  let n = String.length s in
  let step h w =
    let h = Int32.mul (Int32.logxor h w) 0x0100_0193l in
    Int32.logxor h (Int32.shift_right_logical h 15)
  in
  let rec words h i =
    if i + 4 <= n then words (step h (String.get_int32_le s i)) (i + 4)
    else bytes h i
  and bytes h i =
    if i < n then bytes (step h (Int32.of_int (Char.code s.[i]))) (i + 1) else h
  in
  words 0x811C_9DC5l 0

module Result = struct
  type t = {
    digest : int32;
    hashes : int32 list;
    mcycle : int64;
    minstret : int64;
    status : int32;
  }
end

module Trap = struct
  type t = { mcause : int32; mepc : int32; mtval : int32 }
end

type t = Result of Result.t | Trap of Trap.t

module Malformed = struct
  type t = Hash_count of int | Magic of string | Size of int
end

let pp_malformed ppf : Malformed.t -> unit = function
  | Hash_count n -> Fmt.pf ppf "%d hashes, at most %d fit" n max_hashes
  | Magic m -> Fmt.pf ppf "magic %S" m
  | Size n -> Fmt.pf ppf "%d bytes, not %d" n size

let is_done line off =
  off + 4 <= String.length line && String.get_int32_le line off = done_word

let decode s =
  let word off = String.get_int32_le s off in
  let u64 off = String.get_int64_le s off in
  if String.length s <> size then Error (Malformed.Size (String.length s))
  else
    match String.sub s 0 8 with
    | m when String.equal m magic_result ->
        let n = Int32.to_int (word 32) in
        if n < 0 || n > max_hashes then Error (Malformed.Hash_count n)
        else
          Ok
            (Result
               {
                 Result.digest = word 8;
                 hashes = List.init n (fun k -> word (hashes_offset + (4 * k)));
                 mcycle = u64 16;
                 minstret = u64 24;
                 status = word 12;
               })
    | m when String.equal m magic_trap ->
        Ok (Trap { Trap.mcause = word 12; mepc = word 16; mtval = word 20 })
    | m -> Error (Malformed.Magic m)
