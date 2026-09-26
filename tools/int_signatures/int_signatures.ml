(* Signature ratchet for domain-typed integers.

   A bare [int] in an .mli of an in-scope library must be dimensionless or
   external (see .ai/, domain-typed integers). This tool lists every declaration
   in those signatures that mentions [int] and compares the list with a
   checked-in allowlist, each entry carrying a category.

     int_signatures check <allowlist> <root> <dir>...   fail on any drift
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
let declarations path =
  let lines = String.split_on_char '\n' (strip (read_file path)) in
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

let current root dirs =
  List.concat_map (mlis root) dirs
  |> List.concat_map (fun f ->
      List.map
        (fun (ln, d) -> (f, ln, d))
        (declarations (Filename.concat root f)))

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
      (List.map (fun (f, _, d) -> (f, (), d)) (current root dirs))
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
