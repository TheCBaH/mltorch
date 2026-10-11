type expect = Agree | Disagree | Invalid | Refused of string
type check = Has of string | Lacks of string

type row = {
  id : string;
  left : string;
  right : string;
  env : (string * string) list;
  expect : expect;
  checks : check list;
  line : int;
}

let ( let* ) = Result.bind

let words r =
  List.concat_map
    (fun v ->
      match String.index_opt v ':' with
      | None -> [ v ]
      | Some i ->
          String.sub v 0 i
          :: String.split_on_char ','
               (String.sub v (i + 1) (String.length v - i - 1)))
    [ r.left; r.right ]

let prefix p s =
  let n = String.length p in
  if String.length s >= n && String.sub s 0 n = p then
    Some (String.trim (String.sub s n (String.length s - n)))
  else None

let expect_of s =
  match String.trim s with
  | "agree" -> Ok Agree
  | "disagree" -> Ok Disagree
  | "invalid" -> Ok Invalid
  | s -> (
      match prefix "refused " s with
      | Some t when t <> "" -> Ok (Refused t)
      | _ -> Error (Printf.sprintf "unknown expectation %S" s))

let env_of s =
  match String.trim s with
  | "-" -> Ok []
  | s ->
      List.fold_right
        (fun kv acc ->
          let* acc = acc in
          match String.index_opt kv '=' with
          | Some i when i > 0 ->
              Ok
                (( String.sub kv 0 i,
                   String.sub kv (i + 1) (String.length kv - i - 1) )
                :: acc)
          | _ -> Error (Printf.sprintf "environment %S is not NAME=VALUE" kv))
        (List.filter (( <> ) "") (String.split_on_char ' ' s))
        (Ok [])

let checks_of s =
  match String.trim s with
  | "-" -> Ok []
  | s ->
      List.fold_right
        (fun c acc ->
          let* acc = acc in
          let c = String.trim c in
          if String.length c < 2 then Error (Printf.sprintf "check %S" c)
          else
            let t = String.trim (String.sub c 1 (String.length c - 1)) in
            match c.[0] with
            | '+' -> Ok (Has t :: acc)
            | '-' -> Ok (Lacks t :: acc)
            | _ -> Error (Printf.sprintf "check %S starts with + or -" c))
        (String.split_on_char ';' s)
        (Ok [])

let row_of ~line text =
  let at msg = Printf.sprintf "line %d: %s" line msg in
  match List.map String.trim (String.split_on_char '|' text) with
  | [ id; left; right; env; expect; checks ] ->
      if id = "" || left = "" || right = "" then Error (at "empty field")
      else
        Result.map_error at
          (let* env = env_of env in
           let* expect = expect_of expect in
           let* checks = checks_of checks in
           Ok { id; left; right; env; expect; checks; line })
  | fields ->
      Error (at (Printf.sprintf "%d fields, expected 6" (List.length fields)))

let parse text =
  let lines = String.split_on_char '\n' text in
  let rows =
    List.mapi (fun i l -> (i + 1, String.trim l)) lines
    |> List.filter (fun (_, l) -> l <> "" && l.[0] <> '#')
  in
  let* rows =
    List.fold_right
      (fun (line, l) acc ->
        let* acc = acc in
        let* r = row_of ~line l in
        Ok (r :: acc))
      rows (Ok [])
  in
  let rec dup seen = function
    | [] -> Ok rows
    | r :: rest ->
        if List.mem r.id seen then
          Error (Printf.sprintf "line %d: duplicate id %s" r.line r.id)
        else dup (r.id :: seen) rest
  in
  dup [] rows
