(* Variant-order check.

   A type whose constructors are meant to stay in alphabetical order opts in
   with a marker comment alone on the line above its [type] (or [and]) line:

     (* @alphabetical *)
     type t = Add | Add_scalar | Addcmul

   The check lists the constructors of every marked type (ordinary or
   polymorphic variants) and fails on any adjacent pair out of order. The
   collation is the byte order of the lowercased name (LC_ALL=C), so [_] sorts
   before letters and a name's suffixed forms stay beside it: [Add],
   [Add_scalar], [Addcmul]. A marker not followed by a variant type also fails,
   so a marker cannot rot into a claim nothing checks.

     variant_order check <root> <dir>...   fail on any marked type out of order

   Comments and string contents are blanked before the constructors are read
   (the same scan as tools/int_signatures), so a comment between constructors
   or a [|] inside a string does not matter. *)

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

let marker = "(* @alphabetical *)"

(* The declaration that starts on line [start] of the stripped source: every
   following line that is blank or indented. *)
let chunk lines start =
  let n = Array.length lines in
  let rec stop k =
    if k >= n then k
    else
      let l = lines.(k) in
      if String.trim l = "" || l.[0] = ' ' || l.[0] = '\t' then stop (k + 1)
      else k
  in
  String.concat "\n"
    (Array.to_list (Array.sub lines start (stop (start + 1) - start)))

(* Split [s] at the [|] that sit outside every bracket pair. *)
let split_bars s =
  let n = String.length s in
  let rec go i depth cur acc =
    if i >= n then List.rev (String.sub s cur (n - cur) :: acc)
    else
      match s.[i] with
      | '(' | '[' | '{' -> go (i + 1) (depth + 1) cur acc
      | ')' | ']' | '}' -> go (i + 1) (depth - 1) cur acc
      | '|' when depth = 0 && not (i + 1 < n && s.[i + 1] = '|') ->
          go (i + 1) depth (i + 1) (String.sub s cur (i - cur) :: acc)
      | _ -> go (i + 1) depth cur acc
  in
  go 0 0 0 []

(* The leading constructor of a segment, without a polymorphic variant's tick. *)
let constructor seg =
  let s = String.trim seg in
  let s =
    if s <> "" && s.[0] = '`' then String.sub s 1 (String.length s - 1) else s
  in
  let n = String.length s in
  let rec stop i = if i < n && is_ident s.[i] then stop (i + 1) else i in
  String.sub s 0 (stop 0)

let names_of body =
  List.filter (( <> ) "") (List.map constructor (split_bars body))

(* The constructors of the declaration text, or why there are none. *)
let constructors decl =
  match String.index_opt decl '=' with
  | None -> Error "no [=] after the type name"
  | Some eq -> (
      let body =
        String.trim (String.sub decl (eq + 1) (String.length decl - eq - 1))
      in
      let body =
        if String.length body >= 7 && String.sub body 0 7 = "private" then
          String.trim (String.sub body 7 (String.length body - 7))
        else body
      in
      if body = "" then Error "an empty body"
      else
        match body.[0] with
        | '{' -> Error "a record, not a variant"
        | '[' -> (
            match String.rindex_opt body ']' with
            | None -> Error "an unclosed polymorphic variant"
            | Some close ->
                let inner = String.sub body 1 (close - 1) in
                let inner =
                  if inner <> "" && (inner.[0] = '>' || inner.[0] = '<') then
                    String.sub inner 1 (String.length inner - 1)
                  else inner
                in
                Ok (names_of inner))
        | _ ->
            let names = names_of body in
            if names = [] then Error "no constructors" else Ok names)

let collate = String.lowercase_ascii

(* Adjacent pairs out of order, as (earlier, later). *)
let inversions names =
  let rec go acc = function
    | a :: (b :: _ as rest) ->
        go
          (if compare (collate a) (collate b) >= 0 then (a, b) :: acc else acc)
          rest
    | _ -> List.rev acc
  in
  go [] names

let check_file ~root rel =
  let src = read_file (Filename.concat root rel) in
  let raw = Array.of_list (String.split_on_char '\n' src) in
  let stripped = Array.of_list (String.split_on_char '\n' (strip src)) in
  let errs = ref [] and marked = ref 0 in
  Array.iteri
    (fun i l ->
      if String.trim l = marker then (
        incr marked;
        let rec next j =
          if j >= Array.length stripped then None
          else if String.trim stripped.(j) = "" then next (j + 1)
          else Some j
        in
        let fail msg =
          errs := Printf.sprintf "%s:%d: %s" rel (i + 1) msg :: !errs
        in
        match next (i + 1) with
        | None -> fail "the marker is at the end of the file"
        | Some j -> (
            let first = String.trim stripped.(j) in
            let starts p =
              let m = String.length p in
              String.length first >= m
              && String.sub first 0 m = p
              && (String.length first = m || not (is_ident first.[m]))
            in
            if not (starts "type" || starts "and") then
              fail "the marker is not followed by a type"
            else
              match constructors (chunk stripped j) with
              | Error why -> fail ("the marked type is " ^ why)
              | Ok names ->
                  List.iter
                    (fun (a, b) ->
                      fail
                        (Printf.sprintf
                           "%s must come after %s (lowercased byte order, [_] \
                            before letters)"
                           a b))
                    (inversions names))))
    raw;
  (List.rev !errs, !marked)

let rec sources root dir =
  let full = Filename.concat root dir in
  Sys.readdir full |> Array.to_list |> List.sort compare
  |> List.concat_map (fun e ->
      let rel = Filename.concat dir e in
      let p = Filename.concat root rel in
      if Sys.is_directory p then
        if e <> "" && (e.[0] = '.' || e.[0] = '_') then [] else sources root rel
      else if
        (Filename.check_suffix e ".ml" || Filename.check_suffix e ".mli")
        && not
             (Filename.check_suffix e ".pp.ml"
             || Filename.check_suffix e ".pp.mli")
      then [ rel ]
      else [])

let () =
  match List.tl (Array.to_list Sys.argv) with
  | "check" :: root :: (_ :: _ as dirs) ->
      let results =
        List.concat_map (sources root) dirs |> List.map (check_file ~root)
      in
      let errs = List.concat_map fst results in
      let marked = List.fold_left (fun n (_, m) -> n + m) 0 results in
      List.iter prerr_endline errs;
      if errs <> [] then exit 1
      else Printf.printf "variant-order: %d marked type(s) in order\n" marked
  | _ ->
      prerr_endline "usage: variant_order check <root> <dir>...";
      exit 2
