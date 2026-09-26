type t = string

(* ECMAScript reserved words, plus the names strict mode forbids binding and
   the value-like globals a program must not shadow. *)
let reserved =
  [
    "Infinity";
    "NaN";
    "arguments";
    "await";
    "break";
    "case";
    "catch";
    "class";
    "const";
    "continue";
    "debugger";
    "default";
    "delete";
    "do";
    "else";
    "enum";
    "eval";
    "export";
    "extends";
    "false";
    "finally";
    "for";
    "function";
    "if";
    "implements";
    "import";
    "in";
    "instanceof";
    "interface";
    "let";
    "new";
    "null";
    "package";
    "private";
    "protected";
    "public";
    "return";
    "static";
    "super";
    "switch";
    "this";
    "throw";
    "true";
    "try";
    "typeof";
    "undefined";
    "var";
    "void";
    "while";
    "with";
    "yield";
  ]

let start_char = function
  | 'A' .. 'Z' | 'a' .. 'z' | '_' | '$' -> true
  | _ -> false

let part_char = function '0' .. '9' -> true | c -> start_char c

let v s =
  let bad why = invalid_arg (Printf.sprintf "Js_ident.v %S: %s" s why) in
  if s = "" then bad "empty";
  if not (start_char s.[0]) then bad "not an identifier start";
  String.iter (fun c -> if not (part_char c) then bad "not an identifier") s;
  if List.mem s reserved then bad "reserved word";
  if List.exists (fun g -> Js_global.name g = s) Js_global.all then
    bad "names a global";
  s

let to_string s = s
let compare = String.compare
let equal = String.equal
let pp = Fmt.string

module Set = Set.Make (struct
  type nonrec t = t

  let compare = compare
end)
