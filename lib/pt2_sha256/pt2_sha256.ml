type bigstring =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

module Digest = struct
  type t = string

  let equal = String.equal
  let hex = "0123456789abcdef"

  let to_hex d =
    String.init 64 (fun i ->
        let byte = Char.code d.[i / 2] in
        hex.[if i land 1 = 0 then byte lsr 4 else byte land 15])

  let nibble = function
    | '0' .. '9' as c -> Some (Char.code c - Char.code '0')
    | 'a' .. 'f' as c -> Some (Char.code c - Char.code 'a' + 10)
    | _ -> None

  let of_hex s =
    if String.length s <> 64 then None
    else
      let bytes = Bytes.create 32 in
      let rec go i =
        if i = 32 then Some (Bytes.unsafe_to_string bytes)
        else
          match (nibble s.[2 * i], nibble s.[(2 * i) + 1]) with
          | Some hi, Some lo ->
              Bytes.set bytes i (Char.chr ((hi lsl 4) lor lo));
              go (i + 1)
          | _ -> None
      in
      go 0

  let pp ppf d = Fmt.string ppf (to_hex d)
end

let k =
  [|
    0x428a2f98l;
    0x71374491l;
    0xb5c0fbcfl;
    0xe9b5dba5l;
    0x3956c25bl;
    0x59f111f1l;
    0x923f82a4l;
    0xab1c5ed5l;
    0xd807aa98l;
    0x12835b01l;
    0x243185bel;
    0x550c7dc3l;
    0x72be5d74l;
    0x80deb1fel;
    0x9bdc06a7l;
    0xc19bf174l;
    0xe49b69c1l;
    0xefbe4786l;
    0x0fc19dc6l;
    0x240ca1ccl;
    0x2de92c6fl;
    0x4a7484aal;
    0x5cb0a9dcl;
    0x76f988dal;
    0x983e5152l;
    0xa831c66dl;
    0xb00327c8l;
    0xbf597fc7l;
    0xc6e00bf3l;
    0xd5a79147l;
    0x06ca6351l;
    0x14292967l;
    0x27b70a85l;
    0x2e1b2138l;
    0x4d2c6dfcl;
    0x53380d13l;
    0x650a7354l;
    0x766a0abbl;
    0x81c2c92el;
    0x92722c85l;
    0xa2bfe8a1l;
    0xa81a664bl;
    0xc24b8b70l;
    0xc76c51a3l;
    0xd192e819l;
    0xd6990624l;
    0xf40e3585l;
    0x106aa070l;
    0x19a4c116l;
    0x1e376c08l;
    0x2748774cl;
    0x34b0bcb5l;
    0x391c0cb3l;
    0x4ed8aa4al;
    0x5b9cca4fl;
    0x682e6ff3l;
    0x748f82eel;
    0x78a5636fl;
    0x84c87814l;
    0x8cc70208l;
    0x90befffal;
    0xa4506cebl;
    0xbef9a3f7l;
    0xc67178f2l;
  |]

let h0 =
  [|
    0x6a09e667l;
    0xbb67ae85l;
    0x3c6ef372l;
    0xa54ff53al;
    0x510e527fl;
    0x9b05688cl;
    0x1f83d9abl;
    0x5be0cd19l;
  |]

(* The chaining value lives in a [Bytes] rather than an [Int32] array, so a
   native build stores and loads words without boxing them. *)
type t = {
  state : Bytes.t; (* 8 big-endian words *)
  block : Bytes.t; (* a partial block of up to 63 bytes *)
  schedule : Bytes.t; (* 64 words *)
  mutable filled : int;
  mutable total : int64; (* bytes added *)
  mutable finished : bool;
}

let create () =
  let state = Bytes.create 32 in
  Array.iteri (fun i w -> Bytes.set_int32_be state (4 * i) w) h0;
  {
    state;
    block = Bytes.create 64;
    schedule = Bytes.create 256;
    filled = 0;
    total = 0L;
    finished = false;
  }

let ( +% ) = Int32.add
let ( ^% ) = Int32.logxor

(* [&%] shares the comparison operators' precedence, below [^%]: always
   parenthesise a mix of the two. *)
let ( &% ) = Int32.logand

let[@inline] rotr x n =
  Int32.logor (Int32.shift_right_logical x n) (Int32.shift_left x (32 - n))

let[@inline] shr x n = Int32.shift_right_logical x n
let[@inline] get w i = Bytes.get_int32_ne w (4 * i)
let[@inline] set w i v = Bytes.set_int32_ne w (4 * i) v

(* [read i] is the big-endian word [i] of the block being compressed. *)
let compress t (read : int -> int32) =
  let w = t.schedule in
  for i = 0 to 15 do
    set w i (read i)
  done;
  for i = 16 to 63 do
    let w15 = get w (i - 15) and w2 = get w (i - 2) in
    let s0 = rotr w15 7 ^% rotr w15 18 ^% shr w15 3 in
    let s1 = rotr w2 17 ^% rotr w2 19 ^% shr w2 10 in
    set w i (get w (i - 16) +% s0 +% get w (i - 7) +% s1)
  done;
  let st = t.state in
  let a = ref (Bytes.get_int32_be st 0)
  and b = ref (Bytes.get_int32_be st 4)
  and c = ref (Bytes.get_int32_be st 8)
  and d = ref (Bytes.get_int32_be st 12)
  and e = ref (Bytes.get_int32_be st 16)
  and f = ref (Bytes.get_int32_be st 20)
  and g = ref (Bytes.get_int32_be st 24)
  and h = ref (Bytes.get_int32_be st 28) in
  for i = 0 to 63 do
    let e' = !e and a' = !a in
    let s1 = rotr e' 6 ^% rotr e' 11 ^% rotr e' 25 in
    let ch = (e' &% !f) ^% (Int32.lognot e' &% !g) in
    let t1 = !h +% s1 +% ch +% Array.unsafe_get k i +% get w i in
    let s0 = rotr a' 2 ^% rotr a' 13 ^% rotr a' 22 in
    let maj = (a' &% !b) ^% (a' &% !c) ^% (!b &% !c) in
    let t2 = s0 +% maj in
    h := !g;
    g := !f;
    f := e';
    e := !d +% t1;
    d := !c;
    c := !b;
    b := a';
    a := t1 +% t2
  done;
  let add i v =
    Bytes.set_int32_be st (4 * i) (Bytes.get_int32_be st (4 * i) +% v)
  in
  add 0 !a;
  add 1 !b;
  add 2 !c;
  add 3 !d;
  add 4 !e;
  add 5 !f;
  add 6 !g;
  add 7 !h

let check_open t what =
  if t.finished then invalid_arg ("Pt2_sha256." ^ what ^ ": already finished")

let range what ~length ?(pos = 0) ?len () =
  let len = Option.value len ~default:(length - pos) in
  if pos < 0 || len < 0 || pos > length - len then
    invalid_arg ("Pt2_sha256." ^ what ^ ": range outside the input");
  (pos, len)

(* Whole blocks go straight from the input; only a tail is copied. [byte j] is
   the input byte at offset [j] from [base]. *)
let add t ~(byte : int -> int) ~(word : int -> int32) ~pos ~len =
  t.total <- Int64.add t.total (Int64.of_int len);
  let pos = ref pos and len = ref len in
  let flush_partial () =
    compress t (fun i -> Bytes.get_int32_be t.block (4 * i));
    t.filled <- 0
  in
  if t.filled > 0 then begin
    let n = min !len (64 - t.filled) in
    for j = 0 to n - 1 do
      Bytes.unsafe_set t.block (t.filled + j)
        (Char.unsafe_chr (byte (!pos + j)))
    done;
    t.filled <- t.filled + n;
    pos := !pos + n;
    len := !len - n;
    if t.filled = 64 then flush_partial ()
  end;
  while !len >= 64 do
    let base = !pos in
    compress t (fun i -> word (base + (4 * i)));
    pos := !pos + 64;
    len := !len - 64
  done;
  if !len > 0 then begin
    for j = 0 to !len - 1 do
      Bytes.unsafe_set t.block j (Char.unsafe_chr (byte (!pos + j)))
    done;
    t.filled <- !len
  end

let add_string t ?pos ?len s =
  check_open t "add_string";
  let pos, len = range "add_string" ~length:(String.length s) ?pos ?len () in
  add t
    ~byte:(fun j -> Char.code (String.unsafe_get s j))
    ~word:(fun j -> String.get_int32_be s j)
    ~pos ~len

let add_bigstring t ?pos ?len (b : bigstring) =
  check_open t "add_bigstring";
  let pos, len =
    range "add_bigstring" ~length:(Bigarray.Array1.dim b) ?pos ?len ()
  in
  let byte j = Char.code (Bigarray.Array1.unsafe_get b j) in
  let word j =
    Int32.logor
      (Int32.shift_left (Int32.of_int (byte j)) 24)
      (Int32.of_int
         ((byte (j + 1) lsl 16) lor (byte (j + 2) lsl 8) lor byte (j + 3)))
  in
  add t ~byte ~word ~pos ~len

let finish t =
  check_open t "finish";
  let bits = Int64.shift_left t.total 3 in
  let pad = if t.filled < 56 then 56 - t.filled else 120 - t.filled in
  let tail = Bytes.make (pad + 8) '\000' in
  Bytes.set tail 0 '\x80';
  Bytes.set_int64_be tail pad bits;
  (* [add] would count the padding; restore the length it must not include. *)
  let total = t.total in
  add t
    ~byte:(fun j -> Char.code (Bytes.get tail j))
    ~word:(fun j -> Bytes.get_int32_be tail j)
    ~pos:0 ~len:(pad + 8);
  t.total <- total;
  t.finished <- true;
  Bytes.to_string t.state

let string s =
  let t = create () in
  add_string t s;
  finish t

let bigstring b =
  let t = create () in
  add_bigstring t b;
  finish t
