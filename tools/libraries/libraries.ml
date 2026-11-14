(* Library index, read from the dune files alone.

     libraries list <root> [name...]   name, dir, role, dependencies
     libraries check <root>            fail on any drift

   The role is the first sentence of the comment that opens the library's dune
   file. The check fails on a library with no such comment and on two libraries
   with one name; it reads no design document. *)

let read_file path =
  let ic = open_in_bin path in
  let s = really_input_string ic (in_channel_length ic) in
  close_in ic;
  s

(* ---- dune files --------------------------------------------------------- *)

type sexp = Atom of string | List of sexp list

(* Comments ([;] to end of line, [#| ... |#]) are skipped. *)
let parse_sexps src =
  let n = String.length src in
  let rec skip i =
    if i >= n then i
    else
      match src.[i] with
      | ' ' | '\t' | '\n' | '\r' -> skip (i + 1)
      | ';' ->
          let rec eol j = if j >= n || src.[j] = '\n' then j else eol (j + 1) in
          skip (eol i)
      | '#' when i + 1 < n && src.[i + 1] = '|' ->
          let rec close j depth =
            if j + 1 >= n then n
            else if src.[j] = '|' && src.[j + 1] = '#' then
              if depth = 1 then j + 2 else close (j + 2) (depth - 1)
            else if src.[j] = '#' && src.[j + 1] = '|' then
              close (j + 2) (depth + 1)
            else close (j + 1) depth
          in
          skip (close (i + 2) 1)
      | _ -> i
  in
  let rec one i =
    let i = skip i in
    if i >= n then None
    else
      match src.[i] with
      | ')' -> None
      | '(' ->
          let rec items i acc =
            let i = skip i in
            if i >= n then (List (List.rev acc), n)
            else if src.[i] = ')' then (List (List.rev acc), i + 1)
            else
              match one i with
              | Some (x, j) -> items j (x :: acc)
              | None -> (List (List.rev acc), i + 1)
          in
          let l, j = items (i + 1) [] in
          Some (l, j)
      | '"' ->
          let b = Buffer.create 16 in
          let rec str j =
            if j >= n then j
            else if src.[j] = '\\' && j + 1 < n then (
              Buffer.add_char b src.[j + 1];
              str (j + 2))
            else if src.[j] = '"' then j + 1
            else (
              Buffer.add_char b src.[j];
              str (j + 1))
          in
          let j = str (i + 1) in
          Some (Atom (Buffer.contents b), j)
      | _ ->
          let rec atom j =
            if j >= n then j
            else
              match src.[j] with
              | ' ' | '\t' | '\n' | '\r' | '(' | ')' | ';' | '"' -> j
              | _ -> atom (j + 1)
          in
          let j = atom i in
          Some (Atom (String.sub src i (j - i)), j)
  in
  let rec all i acc =
    match one i with Some (x, j) -> all j (x :: acc) | None -> List.rev acc
  in
  all 0 []

type lib = {
  name : string;
  dir : string;  (** [lib/<dir>] *)
  deps : string list;
  role : string;
}

let rec atoms = function
  | Atom a -> [ a ]
  | List (Atom "re_export" :: xs) -> List.concat_map atoms xs
  | List _ -> []

(* The first sentence of the leading [;] comment's first paragraph, whitespace
   collapsed, cut at 200 characters. *)
let leading_role src =
  let lines = String.split_on_char '\n' src in
  let rec comment acc = function
    | l :: rest when String.length l > 0 && l.[0] = ';' ->
        let t = String.trim (String.sub l 1 (String.length l - 1)) in
        if t = "" && acc <> [] then List.rev acc else comment (t :: acc) rest
    | _ -> List.rev acc
  in
  let words =
    String.concat " " (comment [] lines)
    |> String.split_on_char ' '
    |> List.filter (fun w -> w <> "")
  in
  let text = String.concat " " words in
  let cut =
    let n = String.length text in
    let rec find i =
      if i + 1 >= n then n
      else if text.[i] = '.' && text.[i + 1] = ' ' then i + 1
      else find (i + 1)
    in
    find 0
  in
  let text = String.sub text 0 cut in
  if String.length text > 200 then String.sub text 0 197 ^ "..." else text

let libs_of_dir root dir =
  let path = Filename.concat root (Filename.concat "lib" dir) in
  let dune = Filename.concat path "dune" in
  if not (Sys.file_exists dune) then []
  else
    let src = read_file dune in
    let role = leading_role src in
    List.filter_map
      (function
        | List (Atom "library" :: fields) ->
            let field k =
              List.find_map
                (function
                  | List (Atom k' :: xs) when k = k' -> Some xs | _ -> None)
                fields
            in
            Option.map
              (fun name ->
                let name = match name with Atom a :: _ -> a | _ -> dir in
                let deps =
                  match field "libraries" with
                  | Some xs -> List.sort_uniq compare (List.concat_map atoms xs)
                  | None -> []
                in
                { name; dir; deps; role })
              (field "name")
        | _ -> None)
      (parse_sexps src)

let all_libs root =
  let lib = Filename.concat root "lib" in
  let dirs = Sys.readdir lib in
  Array.sort compare dirs;
  Array.to_list dirs
  |> List.filter (fun d -> Sys.is_directory (Filename.concat lib d))
  |> List.concat_map (libs_of_dir root)

(* ---- commands ----------------------------------------------------------- *)

let list root names =
  List.iter
    (fun l ->
      if names = [] || List.mem l.name names then
        Printf.printf "%s\tlib/%s\t%s\t%s\n" l.name l.dir l.role
          (String.concat " " l.deps))
    (all_libs root)

let check root =
  let ls = all_libs root in
  let errors = ref [] in
  let err fmt = Printf.ksprintf (fun s -> errors := s :: !errors) fmt in
  List.iter
    (fun l ->
      if l.role = "" then
        err
          "lib/%s: library %s has no role; open its dune file with a comment \
           whose first sentence says what the library is for"
          l.dir l.name)
    ls;
  let rec dup = function
    | [] -> ()
    | l :: rest ->
        (match
           List.find_opt (fun m -> m.name = l.name && m.dir <> l.dir) rest
         with
        | Some m ->
            err "library name %s is defined in lib/%s and lib/%s" l.name l.dir
              m.dir
        | None -> ());
        dup rest
  in
  dup ls;
  match List.rev !errors with
  | [] -> ()
  | es ->
      List.iter (fun e -> prerr_endline ("FAIL: " ^ e)) es;
      exit 1

let () =
  match Array.to_list Sys.argv |> List.tl with
  | [ "list"; root ] -> list root []
  | "list" :: root :: names -> list root names
  | [ "check"; root ] -> check root
  | _ ->
      prerr_endline
        "usage: libraries list <root> [name...] | libraries check <root>";
      exit 2
