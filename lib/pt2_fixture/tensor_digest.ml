module Dtype = Pt2_checkpoint_map.Dtype

let torch_dtype_name : Dtype.t -> string = function
  | BF16 -> "torch.bfloat16"
  | BOOL -> "torch.bool"
  | F16 -> "torch.float16"
  | F32 -> "torch.float32"
  | F64 -> "torch.float64"
  | I16 -> "torch.int16"
  | I32 -> "torch.int32"
  | I64 -> "torch.int64"
  | I8 -> "torch.int8"
  | U8 -> "torch.uint8"

(* Python's json.dumps of a plain ASCII string: quotes and backslash escaped,
   control characters as \n \r \t \b \f or \u00XX. *)
let json_string s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"';
  String.iter
    (function
      | '"' -> Buffer.add_string b "\\\""
      | '\\' -> Buffer.add_string b "\\\\"
      | '\n' -> Buffer.add_string b "\\n"
      | '\r' -> Buffer.add_string b "\\r"
      | '\t' -> Buffer.add_string b "\\t"
      | '\b' -> Buffer.add_string b "\\b"
      | '\012' -> Buffer.add_string b "\\f"
      | c when Char.code c < 0x20 ->
          Buffer.add_string b (Printf.sprintf "\\u%04x" (Char.code c))
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"';
  Buffer.contents b

let preamble name dtype shape =
  if String.exists (fun c -> Char.code c >= 0x80 || c = '\127') name then
    Error (`Digest_name name)
  else
    Ok
      (Printf.sprintf "[%s, %s, [%s]]\n" (json_string name)
         (json_string (torch_dtype_name dtype))
         (String.concat ", " (List.map Int64.to_string shape)))

let digest named =
  let h = Pt2_sha256.create () in
  let rec go = function
    | [] -> Ok (Pt2_sha256.finish h)
    | (name, (t : Logical.t)) :: rest -> (
        match preamble name t.dtype t.shape with
        | Error _ as e -> e
        | Ok line ->
            Pt2_sha256.add_string h line;
            Pt2_sha256.add_bigstring h t.data;
            go rest)
  in
  go named
