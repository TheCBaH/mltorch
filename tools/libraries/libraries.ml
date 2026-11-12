(* Library ownership index.

   Every library under lib/ should be claimed by exactly one design doc in .ai/,
   through a header line in the doc's first lines:

     Libraries: name, name, ...
     Status: YYYY-MM-DD <one line>

     libraries list <root> [name...]      name, dir, owner, role, dependencies
     libraries check <unowned> <root>     fail on any drift

   The check fails on:
     - a library no doc claims that is not listed in <unowned>;
     - a library claimed by more than one doc;
     - a [Libraries:] name that is no library (stale);
     - a doc with [Libraries:] and no dated [Status:];
     - an <unowned> entry that is claimed or does not exist (stale);
     - more unowned entries than the recorded ceiling, or fewer without
       lowering it.

   <unowned> lines are library names; [#] starts a comment and
   [@unowned-ceiling N] records the ceiling. The role is the first sentence of
   the comment that opens the library's dune file. *)

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

(* ---- docs --------------------------------------------------------------- *)

type doc = {
  file : string;
  claims : string list;
  status : string option;  (** the ISO date *)
}

let is_date s =
  String.length s >= 10
  &&
    try Scanf.sscanf (String.sub s 0 10) "%4d-%2d-%2d%!" (fun _ _ _ -> true)
    with _ -> false

let header_lines = 40

let docs root =
  let dir = Filename.concat root ".ai" in
  let files = Sys.readdir dir in
  Array.sort compare files;
  Array.to_list files
  |> List.filter (fun f -> Filename.check_suffix f ".md")
  |> List.map (fun f ->
      let lines =
        String.split_on_char '\n' (read_file (Filename.concat dir f))
      in
      let lines = List.filteri (fun i _ -> i < header_lines) lines in
      let after prefix l =
        let p = String.length prefix in
        if String.length l >= p && String.sub l 0 p = prefix then
          Some (String.trim (String.sub l p (String.length l - p)))
        else None
      in
      let claims =
        List.concat_map
          (fun l ->
            match after "Libraries:" l with
            | Some r ->
                String.map (fun c -> if c = ',' then ' ' else c) r
                |> String.split_on_char ' '
                |> List.filter (fun w -> w <> "")
            | None -> [])
          lines
      in
      let status =
        List.find_map
          (fun l ->
            match after "Status:" l with
            | Some r when is_date r -> Some (String.sub r 0 10)
            | _ -> None)
          lines
      in
      { file = f; claims; status })

let owners ds name =
  List.filter_map
    (fun d -> if List.mem name d.claims then Some d.file else None)
    ds

(* ---- commands ----------------------------------------------------------- *)

let list root names =
  let ds = docs root in
  List.iter
    (fun l ->
      if names = [] || List.mem l.name names then
        let owner =
          match owners ds l.name with [] -> "-" | o -> String.concat "," o
        in
        Printf.printf "%s\tlib/%s\t%s\t%s\t%s\n" l.name l.dir owner l.role
          (String.concat " " l.deps))
    (all_libs root)

let ceiling_prefix = "@unowned-ceiling "

let read_unowned path =
  let ceiling = ref None and names = ref [] in
  let p = String.length ceiling_prefix in
  List.iter
    (fun l ->
      let l = String.trim l in
      if l = "" || l.[0] = '#' then ()
      else if String.length l > p && String.sub l 0 p = ceiling_prefix then
        ceiling :=
          int_of_string_opt (String.trim (String.sub l p (String.length l - p)))
      else names := l :: !names)
    (String.split_on_char '\n' (read_file path));
  (!ceiling, List.rev !names)

let check unowned root =
  let ls = all_libs root in
  let ds = docs root in
  let known = List.map (fun l -> l.name) ls in
  let errors = ref [] in
  let err fmt = Printf.ksprintf (fun s -> errors := s :: !errors) fmt in
  let ceiling, listed = read_unowned unowned in
  List.iter
    (fun l ->
      match (owners ds l.name, List.mem l.name listed) with
      | [], false ->
          err
            "lib/%s: library %s has no owner; add it to a `Libraries:` line in \
             the .ai design doc that describes it"
            l.dir l.name
      | [], true -> ()
      | [ _ ], true ->
          err "%s is owned but still listed as unowned; delete it from %s"
            l.name unowned
      | [ _ ], false -> ()
      | os, _ ->
          err "library %s is claimed by several docs: %s" l.name
            (String.concat ", " os))
    ls;
  List.iter
    (fun d ->
      List.iter
        (fun c ->
          if not (List.mem c known) then
            err ".ai/%s: `Libraries:` names %s, which is no library under lib/"
              d.file c)
        d.claims;
      if d.claims <> [] && d.status = None then
        err ".ai/%s: has `Libraries:` but no `Status: YYYY-MM-DD` line" d.file)
    ds;
  List.iter
    (fun n ->
      if not (List.mem n known) then
        err "%s: %s is no library under lib/ (stale entry)" unowned n)
    listed;
  let count = List.length listed in
  (match ceiling with
  | None -> err "%s: missing %sN" unowned ceiling_prefix
  | Some c when count > c ->
      err "unowned libraries: %d, ceiling %d; claim them in a doc instead" count
        c
  | Some c when count < c ->
      err "unowned libraries are down to %d; lower %s from %d" count
        (String.trim ceiling_prefix)
        c
  | Some _ -> ());
  match List.rev !errors with
  | [] -> ()
  | es ->
      List.iter (fun e -> prerr_endline ("FAIL: " ^ e)) es;
      exit 1

let () =
  match Array.to_list Sys.argv |> List.tl with
  | [ "list"; root ] -> list root []
  | "list" :: root :: names -> list root names
  | [ "check"; unowned; root ] -> check unowned root
  | _ ->
      prerr_endline
        "usage: libraries list <root> [name...] | libraries check <unowned> \
         <root>";
      exit 2
