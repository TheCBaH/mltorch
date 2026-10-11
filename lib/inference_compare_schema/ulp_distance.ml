module W = Wire

(* An unsigned 64-bit distance held in an [int64]'s bits: the ordered
   encodings of F32/F64 differ by at most 2^64 - 1. *)
type t = int64

let of_unsigned_bits v = v
let to_unsigned_bits v = v
let compare = Int64.unsigned_compare
let equal = Int64.equal
let to_decimal_string v = Printf.sprintf "%Lu" v

let of_decimal_string s =
  (* [Int64.of_string "0u..."] reads an unsigned decimal. *)
  let digits = s <> "" && String.for_all (fun c -> c >= '0' && c <= '9') s in
  if (not digits) || (String.length s > 1 && s.[0] = '0') then None
  else Int64.of_string_opt ("0u" ^ s)

let decode : t W.decoder =
 fun j ->
  let ( let* ) = W.( let* ) in
  let* s = W.string j in
  match of_decimal_string s with
  | Some v -> Ok v
  | None -> W.fail (W.Invalid_decimal s)

let encode v = W.enc_string (to_decimal_string v)
