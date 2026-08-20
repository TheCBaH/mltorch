(* OCaml identifiers derived from ATen schema argument names.

   Shared by every generator that emits an argument name as CODE rather than as
   data: [Aten_spec_gen] (record fields) and [Aten_decode_gen] (let bindings).
   [Aten_config_gen] does not need it -- there the name only ever appears inside
   a string literal.

   The escape is load-bearing, not defensive: [slice.Tensor]'s bounds argument is
   literally named [end]. A generator that emitted it raw would produce a source
   file that does not parse, and it would do so only once that op became
   decodable -- which is exactly the kind of gap that hides behind a green
   build. *)

let keywords =
  [
    "and";
    "as";
    "assert";
    "begin";
    "class";
    "constraint";
    "do";
    "done";
    "downto";
    "else";
    "end";
    "exception";
    "external";
    "false";
    "for";
    "fun";
    "function";
    "functor";
    "if";
    "in";
    "include";
    "inherit";
    "initializer";
    "lazy";
    "let";
    "match";
    "method";
    "module";
    "mutable";
    "new";
    "nonrec";
    "object";
    "of";
    "open";
    "or";
    "private";
    "rec";
    "sig";
    "struct";
    "then";
    "to";
    "true";
    "try";
    "type";
    "val";
    "virtual";
    "when";
    "while";
    "with";
    "mod";
    "land";
    "lor";
    "lxor";
    "lsl";
    "lsr";
    "asr";
  ]

(* OCaml identifier for an arg name (the JSON key keeps the real name). *)
let ml_id name = if List.mem name keywords then name ^ "_" else name
