type t = Compcert_scalar | Gnu

let i64 d n =
  match d with
  | Gnu ->
      if Int64.equal n Int64.min_int then "(-INT64_C(9223372036854775807) - 1)"
      else if Int64.compare n 0L < 0 then
        "(-INT64_C" ^ "(" ^ Int64.to_string (Int64.neg n) ^ "))"
      else "INT64_C(" ^ Int64.to_string n ^ ")"
  | Compcert_scalar ->
      if Int64.equal n Int64.min_int then "(-9223372036854775807L - 1)"
      else if Int64.compare n 0L < 0 then
        "(-" ^ Int64.to_string (Int64.neg n) ^ "L)"
      else Int64.to_string n ^ "L"

let u64 d n =
  match d with
  | Gnu -> Printf.sprintf "UINT64_C(%Ld)" n
  | Compcert_scalar -> Printf.sprintf "%LuUL" n

let nan = function Gnu -> "NAN" | Compcert_scalar -> "(0.0 / 0.0)"
let infinity = function Gnu -> "INFINITY" | Compcert_scalar -> "(1.0 / 0.0)"

let signbit d x =
  match d with
  | Gnu -> "signbit(" ^ x ^ ")"
  | Compcert_scalar ->
      "(" ^ x ^ " < 0.0 || (" ^ x ^ " == 0.0 && 1.0 / " ^ x ^ " < 0.0))"

let isfinite d x =
  match d with
  | Gnu -> "isfinite(" ^ x ^ ")"
  | Compcert_scalar -> "(" ^ x ^ " - " ^ x ^ " == 0.0)"

let host_symbols =
  [
    "cos";
    "exp";
    "fabs";
    "fabsf";
    "fma";
    "fmaf";
    "ldexp";
    "log";
    "memcpy";
    "memset";
    "sin";
    "sqrt";
    "sqrtf";
    "trunc";
    "truncf";
  ]
