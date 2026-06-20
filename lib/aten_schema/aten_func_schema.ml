let parse s =
  let lexbuf = Lexing.from_string s in
  match Aten_func_parser.func_schema Aten_func_lexer.token lexbuf with
  | v -> Ok v
  | exception Failure msg -> Error msg
  | exception Aten_func_parser.Error ->
      let pos = Lexing.lexeme_start lexbuf in
      Error (Printf.sprintf "parse error at char %d in: %s" pos s)
