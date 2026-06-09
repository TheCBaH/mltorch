open Func_ast

let parse (s : string) : (t, string) result =
  let lexbuf = Lexing.from_string s in
  match Func_parser.func_schema Func_lexer.token lexbuf with
  | v -> Ok v
  | exception Failure msg -> Error msg
  | exception Func_parser.Error ->
      let pos = Lexing.lexeme_start lexbuf in
      Error (Printf.sprintf "parse error at char %d in: %s" pos s)
