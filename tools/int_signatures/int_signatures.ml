(* Signature ratchet for domain-typed integers.

   A bare [int] in an .mli of an in-scope library must be dimensionless or
   external (see .ai/, domain-typed integers). This tool lists every declaration
   in those signatures that mentions [int] and compares the list with a
   checked-in allowlist, each entry carrying a category.

     int_signatures check <allowlist> <root> <dir>...   fail on any drift

   A [dir] is read as [.mli] text. [ml:<dir>] also reads the interface the
   compiler infers for each [.ml] of that library that has no [.mli] (the text
   [ocamlc -i] prints), through the same parser, for a library whose interface
   is its implementation. The library's object directories must be built first:
   the dune rule depends on its [check] alias.
     int_signatures print <root> <dir>...               file, line, declaration
     int_signatures update <allowlist> <root> <dir>...  the allowlist to write

   Drift in either direction fails, so the list can only shrink by deleting the
   entry of a declaration that no longer mentions [int]:
     - a declaration with an [int] that is not listed;
     - a listed entry no declaration matches (stale);
     - a category outside the fixed vocabulary;
     - more [todo-domain] entries than the recorded ceiling, or fewer without
       lowering it.

   A declaration is a chunk of the signature that starts at a line beginning
   with [val], [type], [exception], [external], [module], [include], [and],
   [open] or [class], comments and string contents removed, whitespace
   collapsed. Entries are matched as a multiset on (file, declaration), so two
   identical declarations in one file need two entries.

   Entries are lines [file<TAB>category<TAB>declaration]; [#] starts a comment
   and [@todo-domain-ceiling N] records the ceiling. *)

let categories =
  [
    "arity";
    "as-written";
    "bits";
    "budget";
    "cell";
    "defn";
    "expr-literal";
    "gate";
    "hash";
    "quantum";
    "tally";
    "todo-domain";
    "wire";
  ]

let read_file path =
  let ic = open_in_bin path in
  let s = really_input_string ic (in_channel_length ic) in
  close_in ic;
  s

(* Blank out comments (nested) and string contents, keeping newlines so line
   numbers survive. A quote that starts a character literal is skipped. *)
let strip src =
  let n = String.length src in
  let b = Buffer.create n in
  let blank c = Buffer.add_char b (if c = '\n' then '\n' else ' ') in
  let rec go i depth =
    if i >= n then ()
    else if i + 1 < n && src.[i] = '(' && src.[i + 1] = '*' then (
      Buffer.add_string b "  ";
      go (i + 2) (depth + 1))
    else if depth > 0 && i + 1 < n && src.[i] = '*' && src.[i + 1] = ')' then (
      Buffer.add_string b "  ";
      go (i + 2) (depth - 1))
    else if depth > 0 then (
      blank src.[i];
      go (i + 1) depth)
    else if src.[i] = '"' then (
      Buffer.add_char b '"';
      let rec str j =
        if j >= n then j
        else if src.[j] = '\\' then (
          Buffer.add_string b "__";
          str (j + 2))
        else if src.[j] = '"' then j
        else (
          blank src.[j];
          str (j + 1))
      in
      let j = str (i + 1) in
      Buffer.add_char b '"';
      go (j + 1) 0)
    else if i + 2 < n && src.[i] = '\'' && src.[i + 2] = '\'' then (
      Buffer.add_string b "'_'";
      go (i + 3) 0)
    else (
      Buffer.add_char b src.[i];
      go (i + 1) 0)
  in
  go 0 0;
  Buffer.contents b

let is_ident c =
  (c >= 'a' && c <= 'z')
  || (c >= 'A' && c <= 'Z')
  || (c >= '0' && c <= '9')
  || c = '_' || c = '\''

(* Standalone [int] tokens, as the census counts them: not preceded by an
   identifier character or '.', not followed by an identifier character. *)
let count_int s =
  let n = String.length s in
  let rec go i acc =
    if i + 3 > n then acc
    else if
      String.sub s i 3 = "int"
      && (i = 0 || not (is_ident s.[i - 1] || s.[i - 1] = '.'))
      && (i + 3 = n || not (is_ident s.[i + 3]))
    then go (i + 3) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

let starts =
  [
    "val";
    "type";
    "exception";
    "external";
    "module";
    "include";
    "and";
    "open";
    "class";
  ]

let first_word l =
  let l = String.trim l in
  let n = String.length l in
  let rec e i = if i < n && is_ident l.[i] then e (i + 1) else i in
  String.sub l 0 (e 0)

let normalise s =
  String.split_on_char '\n' s
  |> List.concat_map (String.split_on_char ' ')
  |> List.filter (fun w -> w <> "")
  |> String.concat " "

(* (line, declaration) of every declaration that mentions [int]. *)
let declarations_of_text text =
  let lines = String.split_on_char '\n' (strip text) in
  let flush acc = function
    | None -> acc
    | Some (ln, ls) ->
        let d = normalise (String.concat "\n" (List.rev ls)) in
        if count_int d > 0 then (ln, d) :: acc else acc
  in
  let acc, cur, _ =
    List.fold_left
      (fun (acc, cur, ln) l ->
        let ln = ln + 1 in
        if List.mem (first_word l) starts then
          (flush acc cur, Some (ln, [ l ]), ln)
        else
          ( acc,
            (match cur with None -> None | Some (s, ls) -> Some (s, l :: ls)),
            ln ))
      ([], None, 0) lines
  in
  List.rev (flush acc cur)

let declarations path = declarations_of_text (read_file path)

(* Inside dune's build directory the source tree also holds preprocessed copies
   ([.pp.mli]); only the authored signature counts. *)
let rec mlis root dir =
  let full = Filename.concat root dir in
  Sys.readdir full |> Array.to_list |> List.sort compare
  |> List.concat_map (fun e ->
      let rel = Filename.concat dir e in
      let p = Filename.concat root rel in
      if Sys.is_directory p then mlis root rel
      else if
        Filename.check_suffix e ".mli"
        && not (Filename.check_suffix e ".pp.mli")
      then [ rel ]
      else [])

(* ---- [ml:<dir>]: the inferred interface of a library without .mli files ---- *)

let atoms_of_dune text =
  let text =
    String.split_on_char '\n' text
    |> List.map (fun l ->
        match String.index_opt l ';' with
        | Some i -> String.sub l 0 i
        | None -> l)
    |> String.concat "\n"
  in
  let b = Buffer.create 16 and acc = ref [] in
  let flush () =
    if Buffer.length b > 0 then (
      acc := Buffer.contents b :: !acc;
      Buffer.clear b)
  in
  String.iter
    (function
      | ('(' | ')') as c ->
          flush ();
          acc := String.make 1 c :: !acc
      | ' ' | '\n' | '\t' | '\r' -> flush ()
      | c -> Buffer.add_char b c)
    text;
  flush ();
  List.rev !acc

(* The atoms of the first [(key ...)] form, nested forms flattened. *)
let field toks key =
  let rec find = function
    | "(" :: k :: rest when k = key -> Some rest
    | _ :: rest -> find rest
    | [] -> None
  in
  match find toks with
  | None -> []
  | Some rest ->
      let rec take depth acc = function
        | [] -> List.rev acc
        | ")" :: _ when depth = 0 -> List.rev acc
        | ")" :: r -> take (depth - 1) acc r
        | "(" :: r -> take (depth + 1) acc r
        | a :: r -> take depth (a :: acc) r
      in
      take 0 [] rest

(* Every [.<library>.objs/byte] directory under [rel], as (library, path). *)
let rec objs_dirs root rel =
  let full = Filename.concat root rel in
  Sys.readdir full |> Array.to_list |> List.sort compare
  |> List.concat_map (fun e ->
      let r = Filename.concat rel e in
      let p = Filename.concat root r in
      if not (Sys.is_directory p) then []
      else if e <> "" && e.[0] = '.' then
        if Filename.check_suffix e ".objs" then
          [ (String.sub e 1 (String.length e - 6), Filename.concat p "byte") ]
        else []
      else objs_dirs root r)

(* [Ssa_c__.Ssa_c_ctx.t] -> [Ssa_c_ctx.t]: the wrapped library's alias module
   shows through the paths [ocamlc -i] prints. *)
let drop_alias_prefix s =
  let n = String.length s in
  let b = Buffer.create n in
  let rec go i =
    if i >= n then ()
    else if i + 2 < n && s.[i] = '_' && s.[i + 1] = '_' && s.[i + 2] = '.' then (
      let rec start j =
        if j > 0 && is_ident (Buffer.nth b (j - 1)) then start (j - 1) else j
      in
      Buffer.truncate b (start (Buffer.length b));
      go (i + 3))
    else (
      Buffer.add_char b s.[i];
      go (i + 1))
  in
  go 0;
  Buffer.contents b

(* The text [ocamlc -i] prints for [ml] (a path relative to [root]) in library
   directory [dir]: the library's object directories on the include path, its
   alias module opened as dune does, its external libraries as packages. *)
let inferred_text root dir ml =
  let toks =
    atoms_of_dune
      (read_file (Filename.concat (Filename.concat root dir) "dune"))
  in
  let name =
    match field toks "name" with
    | n :: _ -> n
    | [] -> failwith (dir ^ ": no library name")
  in
  let wrapped = field toks "wrapped" <> [ "false" ] in
  (* dune names the alias module [Name__] when the library has a module of its
     own name, and [Name] itself otherwise. *)
  let has_main =
    Sys.file_exists (Filename.concat (Filename.concat root dir) (name ^ ".ml"))
  in
  (* A vendored library built in the workspace (elf) shadows an installed one of
     the same name, so its objects must win over the ocamlfind package. *)
  let vendored =
    if Sys.file_exists (Filename.concat root "vendored") then
      objs_dirs root "vendored"
    else []
  in
  let all = objs_dirs root "lib" @ vendored in
  let internal = List.map fst all in
  let packages =
    List.filter (fun l -> not (List.mem l internal)) (field toks "libraries")
  in
  let out = Filename.temp_file "int_signatures" ".mli" in
  let cmd =
    Printf.sprintf "ocamlfind ocamlc %s%s%s -w -a -i %s > %s 2>&1"
      (if packages = [] then ""
       else "-package " ^ String.concat "," packages ^ " ")
      (String.concat " " (List.map (fun (_, p) -> "-I " ^ Filename.quote p) all)
      ^ " ")
      (if not wrapped then ""
       else if has_main then "-open " ^ String.capitalize_ascii name ^ "__ "
       else "-open " ^ String.capitalize_ascii name ^ " ")
      (Filename.quote (Filename.concat root ml))
      (Filename.quote out)
  in
  let code = Sys.command cmd in
  let text = read_file out in
  Sys.remove out;
  if code <> 0 then
    failwith (Printf.sprintf "%s: ocamlc -i failed:\n%s" ml text);
  drop_alias_prefix text

let mls root dir =
  let rec go rel =
    let full = Filename.concat root rel in
    Sys.readdir full |> Array.to_list |> List.sort compare
    |> List.concat_map (fun e ->
        let r = Filename.concat rel e in
        let p = Filename.concat root r in
        if Sys.is_directory p then if e <> "" && e.[0] = '.' then [] else go r
        else if
          Filename.check_suffix e ".ml"
          && (not (Filename.check_suffix e ".pp.ml"))
          && not (Sys.file_exists (p ^ "i"))
        then [ r ]
        else [])
  in
  go dir

let current root dirs =
  List.concat_map
    (fun dir ->
      let from_mli d =
        mlis root d
        |> List.concat_map (fun f ->
            List.map
              (fun (ln, decl) -> (f, ln, decl))
              (declarations (Filename.concat root f)))
      in
      match String.split_on_char ':' dir with
      | [ "ml"; d ] ->
          from_mli d
          @ List.concat_map
              (fun ml ->
                List.map
                  (fun (ln, decl) -> (ml, ln, decl))
                  (declarations_of_text (inferred_text root d ml)))
              (mls root d)
      | _ -> from_mli dir)
    dirs

let parse_allowlist path =
  let ceiling = ref None in
  let entries =
    String.split_on_char '\n' (read_file path)
    |> List.filter_map (fun l ->
        if l = "" || l.[0] = '#' then None
        else if String.length l > 1 && l.[0] = '@' then (
          (match String.split_on_char ' ' l with
          | [ "@todo-domain-ceiling"; n ] -> ceiling := int_of_string_opt n
          | _ -> ());
          None)
        else
          match String.split_on_char '\t' l with
          | [ f; c; d ] -> Some (f, c, d)
          | _ -> failwith ("malformed allowlist line: " ^ l))
  in
  (!ceiling, entries)

let check allowlist root dirs =
  let ceiling, entries = parse_allowlist allowlist in
  let cur = current root dirs in
  let problems = ref [] in
  let problem fmt = Printf.ksprintf (fun s -> problems := s :: !problems) fmt in
  List.iter
    (fun (f, c, _) ->
      if not (List.mem c categories) then problem "%s: unknown category %S" f c)
    entries;
  (* Multiset match: consume one allowlist entry per current declaration. *)
  let remaining = ref entries in
  List.iter
    (fun (f, ln, d) ->
      let rec take seen = function
        | [] -> None
        | ((f', _, d') as e) :: rest when f = f' && d = d' ->
            Some (List.rev_append seen rest, e)
        | e :: rest -> take (e :: seen) rest
      in
      match take [] !remaining with
      | Some (rest, _) -> remaining := rest
      | None ->
          problem
            "%s:%d: an [int] in a signature has no domain and is not \
             allowlisted:\n\
            \    %s"
            f ln d)
    cur;
  List.iter
    (fun (f, c, d) ->
      problem
        "%s: stale allowlist entry [%s] (declaration gone or no longer \
         mentions int); delete it:\n\
        \    %s"
        f c d)
    !remaining;
  let todo =
    List.length (List.filter (fun (_, c, _) -> c = "todo-domain") entries)
  in
  (match ceiling with
  | None -> problem "allowlist has no @todo-domain-ceiling line"
  | Some n when todo > n ->
      problem "todo-domain entries grew: %d > ceiling %d" todo n
  | Some n when todo < n ->
      problem "todo-domain entries shrank to %d: lower the ceiling (was %d)"
        todo n
  | Some _ -> ());
  match List.rev !problems with
  | [] -> ()
  | ps ->
      List.iter prerr_endline ps;
      Printf.eprintf
        "%d problem(s); see tools/int_signatures/int_signatures.ml\n"
        (List.length ps);
      exit 1

(* The allowlist that would pass: matched entries keep their category, stale
   ones are dropped, a declaration with no entry is written as [UNCLASSIFIED],
   which [check] rejects until a person gives it a category, and the ceiling is
   recounted. Comment lines are kept in place. *)
let update allowlist root dirs =
  let _, entries = parse_allowlist allowlist in
  let comments =
    String.split_on_char '\n' (read_file allowlist)
    |> List.filter (fun l -> l <> "" && l.[0] = '#')
  in
  let remaining = ref entries in
  let rows =
    List.map
      (fun (f, _, d) ->
        let rec take seen = function
          | [] -> None
          | ((f', _, d') as e) :: rest when f = f' && d = d' ->
              Some (List.rev_append seen rest, e)
          | e :: rest -> take (e :: seen) rest
        in
        match take [] !remaining with
        | Some (rest, (_, c, _)) ->
            remaining := rest;
            (f, c, d)
        | None -> (f, "UNCLASSIFIED", d))
      (current root dirs)
  in
  List.iter print_endline comments;
  Printf.printf "@todo-domain-ceiling %d\n"
    (List.length (List.filter (fun (_, c, _) -> c = "todo-domain") rows));
  List.iter (fun (f, c, d) -> Printf.printf "%s\t%s\t%s\n" f c d) rows

let () =
  match Array.to_list Sys.argv |> List.tl with
  | "update" :: allowlist :: root :: dirs -> update allowlist root dirs
  | "check" :: allowlist :: root :: dirs -> check allowlist root dirs
  | "print" :: root :: dirs ->
      List.iter
        (fun (f, ln, d) -> Printf.printf "%s\t%d\t%s\n" f ln d)
        (current root dirs)
  | _ ->
      prerr_endline
        "usage: int_signatures (check <allowlist> <root> | print <root>) \
         <dir>...";
      exit 2
