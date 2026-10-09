let hex_value = function
  | '0' .. '9' as c -> Char.code c - Char.code '0'
  | 'a' .. 'f' as c -> Char.code c - Char.code 'a' + 10
  | _ -> -1

let hex_decode s =
  let n = String.length s in
  if n = 0 || n land 1 = 1 then None
  else
    let out = Bytes.create (n / 2) in
    let rec go i =
      if i = n / 2 then Some (Bytes.unsafe_to_string out)
      else
        let hi = hex_value s.[2 * i] and lo = hex_value s.[(2 * i) + 1] in
        if hi < 0 || lo < 0 then None
        else begin
          Bytes.set out i (Char.chr ((hi lsl 4) lor lo));
          go (i + 1)
        end
    in
    go 0

let b64_value = function
  | 'A' .. 'Z' as c -> Char.code c - Char.code 'A'
  | 'a' .. 'z' as c -> Char.code c - Char.code 'a' + 26
  | '0' .. '9' as c -> Char.code c - Char.code '0' + 52
  | '+' -> 62
  | '/' -> 63
  | _ -> -1

(* Number of '=' closing the string: 0, 1 or 2, and only at the very end. *)
let padding s =
  let n = String.length s in
  if n = 0 then Some 0
  else if s.[n - 1] <> '=' then Some 0
  else if n >= 2 && s.[n - 2] = '=' then Some 2
  else Some 1

let base64_decoded_length s =
  let n = String.length s in
  if n land 3 <> 0 then None
  else match padding s with Some p -> Some ((n / 4 * 3) - p) | None -> None

let base64_decode s =
  match base64_decoded_length s with
  | None -> None
  | Some len ->
      let n = String.length s in
      let pad = (n / 4 * 3) - len in
      let out = Bytes.create len in
      let quad = ref 0 and bad = ref false in
      for g = 0 to (n / 4) - 1 do
        let value k =
          let c = s.[(4 * g) + k] in
          if c = '=' && g = (n / 4) - 1 && k >= 4 - pad then 0
          else
            let v = b64_value c in
            if v < 0 then begin
              bad := true;
              0
            end
            else v
        in
        let a = value 0 and b = value 1 and c = value 2 and d = value 3 in
        quad := (a lsl 18) lor (b lsl 12) lor (c lsl 6) lor d;
        let put k byte =
          if (3 * g) + k < len then Bytes.set out ((3 * g) + k) (Char.chr byte)
        in
        put 0 ((!quad lsr 16) land 255);
        put 1 ((!quad lsr 8) land 255);
        put 2 (!quad land 255);
        (* Canonical: the bits that fall off the end must be zero. *)
        if g = (n / 4) - 1 then begin
          if pad = 2 && !quad land 0xFFFF <> 0 then bad := true;
          if pad = 1 && !quad land 0xFF <> 0 then bad := true
        end
      done;
      if !bad then None else Some (Bytes.unsafe_to_string out)
