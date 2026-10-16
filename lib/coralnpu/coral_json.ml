open Err.Syntax

type error = [ `Duplicate_member of string | `Inexact_number of float ]

let pp_error ppf : error -> unit = function
  | `Duplicate_member name -> Fmt.pf ppf "member %S appears twice" name
  | `Inexact_number x -> Fmt.pf ppf "number %h is not an exact integer" x

(* Python's [ensure_ascii=False] escapes: the quote, the backslash, the five
   short control escapes, and every other control character as [\u00xx]. *)
let add_string b s =
  Buffer.add_char b '"';
  String.iter
    (function
      | '"' -> Buffer.add_string b "\\\""
      | '\\' -> Buffer.add_string b "\\\\"
      | '\b' -> Buffer.add_string b "\\b"
      | '\012' -> Buffer.add_string b "\\f"
      | '\n' -> Buffer.add_string b "\\n"
      | '\r' -> Buffer.add_string b "\\r"
      | '\t' -> Buffer.add_string b "\\t"
      | c when Char.code c < 0x20 -> Printf.bprintf b "\\u%04x" (Char.code c)
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"'

let exact_limit = 9007199254740992. (* 2^53 *)

let add_number b x =
  if Float.is_integer x && Float.abs x < exact_limit then (
    (* [-0] is the integer [0] in Python. *)
    Buffer.add_string b (Printf.sprintf "%.0f" (if x = 0. then 0. else x));
    Err.return ())
  else Err.fail (`Inexact_number x)

(* The first name equal to its successor in a sorted list. *)
let rec duplicate = function
  | a :: (b :: _ as rest) -> if String.equal a b then Some a else duplicate rest
  | [] | [ _ ] -> None

(* [items] between [opening] and [closing], separated by commas. *)
let add_seq b opening closing f items =
  Buffer.add_char b opening;
  let* _ =
    Err.List.fold_left
      (fun first v ->
        if not first then Buffer.add_char b ',';
        let* () = f v in
        Err.return false)
      true items
  in
  Buffer.add_char b closing;
  Err.return ()

let rec add b : Jsont.json -> (unit, [> error ]) Err.t = function
  | Null _ ->
      Buffer.add_string b "null";
      Err.return ()
  | Bool (v, _) ->
      Buffer.add_string b (if v then "true" else "false");
      Err.return ()
  | Number (x, _) -> add_number b x
  | String (s, _) ->
      add_string b s;
      Err.return ()
  | Array (items, _) -> add_seq b '[' ']' (add b) items
  | Object (mems, _) -> (
      let mems =
        List.stable_sort
          (fun (((x : string), _), _) ((y, _), _) -> String.compare x y)
          mems
      in
      match duplicate (List.map (fun ((name, _), _) -> name) mems) with
      | Some name -> Err.fail (`Duplicate_member name)
      | None ->
          add_seq b '{' '}'
            (fun ((name, _), v) ->
              add_string b name;
              Buffer.add_char b ':';
              add b v)
            mems)

let to_string json =
  let b = Buffer.create 4096 in
  let* () = add b json in
  Err.return (Buffer.contents b)
