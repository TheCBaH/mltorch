open Js_ast

let reject s =
  match Ident.v s with
  | id -> Fmt.pr "%S accepted as %a@." s Ident.pp id
  | exception Invalid_argument m -> Fmt.pr "%s@." m

let%expect_test "every rejection class, and the names that pass" =
  List.iter reject
    [
      "";
      "1a";
      "a-b";
      "a b";
      "function";
      "let";
      "NaN";
      "Infinity";
      "Math";
      "Uint8Array";
      "BigInt";
      "_ok";
      "$ok";
      "bf16_to_float";
      "x0";
    ];
  [%expect
    {|
    Js_ident.v "": empty
    Js_ident.v "1a": not an identifier start
    Js_ident.v "a-b": not an identifier
    Js_ident.v "a b": not an identifier
    Js_ident.v "function": reserved word
    Js_ident.v "let": reserved word
    Js_ident.v "NaN": reserved word
    Js_ident.v "Infinity": reserved word
    Js_ident.v "Math": names a global
    Js_ident.v "Uint8Array": names a global
    Js_ident.v "BigInt": names a global
    "_ok" accepted as _ok
    "$ok" accepted as $ok
    "bf16_to_float" accepted as bf16_to_float
    "x0" accepted as x0
    |}]

let%expect_test "every global is rejected as an identifier" =
  List.iter
    (fun g ->
      match Ident.v (Global.name g) with
      | _ -> Fmt.pr "%s accepted@." (Global.name g)
      | exception Invalid_argument _ -> ())
    Global.all;
  [%expect {||}]
