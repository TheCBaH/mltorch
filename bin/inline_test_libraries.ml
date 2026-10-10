(* Discover inline-test runners from tracked Dune stanzas without Python. *)
open Err.Syntax

type form = Atom of string | List of form list
type token = Open | Close | Value of string

let bad text = Err.fail (`Discovery text)

let tokenize text =
  let length = String.length text in
  let space = function ' ' | '\n' | '\r' | '\t' -> true | _ -> false in
  let rec comment i =
    if i = length || text.[i] = '\n' then i else comment (i + 1)
  in
  let rec quoted buffer i =
    if i = length then bad "unterminated quoted atom"
    else
      match text.[i] with
      | '"' -> Ok (Buffer.contents buffer, i + 1)
      | '\\' when i + 1 < length ->
          Buffer.add_char buffer text.[i + 1];
          quoted buffer (i + 2)
      | c ->
          Buffer.add_char buffer c;
          quoted buffer (i + 1)
  in
  let rec atom i =
    if i = length || space text.[i] || List.mem text.[i] [ '('; ')'; ';' ] then
      i
    else atom (i + 1)
  in
  let rec loop i tokens =
    if i = length then Ok (List.rev tokens)
    else
      match text.[i] with
      | c when space c -> loop (i + 1) tokens
      | ';' -> loop (comment (i + 1)) tokens
      | '(' -> loop (i + 1) (Open :: tokens)
      | ')' -> loop (i + 1) (Close :: tokens)
      | '"' ->
          let* value, next = quoted (Buffer.create 16) (i + 1) in
          loop next (Value value :: tokens)
      | _ ->
          let next = atom i in
          loop next (Value (String.sub text i (next - i)) :: tokens)
  in
  loop 0 []

let parse tokens =
  let rec form = function
    | Value value :: rest -> Ok (Atom value, rest)
    | Open :: rest ->
        let rec items acc = function
          | Close :: rest -> Ok (List (List.rev acc), rest)
          | [] -> bad "unclosed list"
          | tokens ->
              let* value, rest = form tokens in
              items (value :: acc) rest
        in
        items [] rest
    | _ -> bad "unexpected close/end"
  in
  let rec all acc = function
    | [] -> Ok (List.rev acc)
    | tokens ->
        let* value, rest = form tokens in
        all (value :: acc) rest
  in
  all [] tokens

let field name forms =
  List.find_map
    (function
      | List (Atom key :: args) when key = name -> Some args | _ -> None)
    forms

let rec gated = function
  | Atom value ->
      (* Dune's env expansions can appear inside a quoted enabled_if. *)
      let contains word =
        let rec loop i =
          i + String.length word <= String.length value
          && (String.sub value i (String.length word) = word || loop (i + 1))
        in
        loop 0
      in
      contains "MLTORCH_WASM" || contains "MLTORCH_COMPCERT"
  | List values -> List.exists gated values

let runners forms =
  Err.List.fold_left
    (fun acc -> function
      | List (Atom "library" :: fields) -> (
          match (field "name" fields, field "inline_tests" fields) with
          | Some [ Atom name ], Some inline ->
              let gates =
                Option.value ~default:[] (field "enabled_if" fields)
                @ Option.value ~default:[] (field "enabled_if" inline)
              in
              if List.exists gated gates then Ok acc
              else
                let modes =
                  Option.value ~default:[ Atom "best" ] (field "modes" inline)
                in
                let+ modes =
                  Err.List.map
                    (function
                      | Atom mode -> Ok mode | _ -> bad "non-atom runner mode")
                    modes
                in
                acc @ List.map (fun mode -> (name, mode)) modes
          | _ -> Ok acc)
      | _ -> Ok acc)
    [] forms

let read path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in channel)
    (fun () -> really_input_string channel (in_channel_length channel))

let files () =
  let channel =
    Unix.open_process_args_in "git" [| "git"; "ls-files"; "--"; "*dune" |]
  in
  let rec loop acc =
    match input_line channel with
    | line -> loop (line :: acc)
    | exception End_of_file -> List.rev acc
  in
  let paths = loop [] in
  match Unix.close_process_in channel with
  | Unix.WEXITED 0 -> Ok paths
  | _ -> bad "git ls-files failed"

let run () =
  let* paths =
    match Array.to_list Sys.argv with
    | [ _ ] -> files ()
    | [ _; "--file"; path ] -> Ok [ path ]
    | _ -> bad "usage: inline_test_libraries [--file PATH]"
  in
  Err.List.iter
    (fun path ->
      if
        Filename.basename path <> "dune"
        || String.starts_with ~prefix:"vendored/" path
        || String.starts_with ~prefix:"modules/" path
      then Ok ()
      else
        let* tokens = tokenize (read path) in
        let* forms = parse tokens in
        let+ rows = runners forms in
        List.iter
          (fun (name, mode) ->
            Printf.printf "%s\t%s\t%s\n" (Filename.dirname path) name mode)
          rows)
    paths

let () =
  let result =
    try run () with
    | Sys_error text -> bad text
    | Unix.Unix_error (error, op, _) ->
        bad (op ^ ": " ^ Unix.error_message error)
  in
  match Err.payload result with
  | Ok () -> ()
  | Error (`Discovery text) ->
      prerr_endline ("inline-test discovery: " ^ text);
      exit 2
