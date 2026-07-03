(* Thin wrapper over the [zipc] archive reader, exposing the lookups the pt2
   loader needs. PyTorch writes every entry STORED (uncompressed), but zipc also
   handles Deflate, so this works regardless. Archives are held in memory as a
   string. The single top-level directory all entries nest under is computed
   once at open time (see [prefix]). *)

type t = { zip : Zipc.t; prefix : string }

type error =
  [ `Zip_parse_failed of string
  | `Zip_read_failed of string * string
  | `Zip_missing_entry of string ]

let entries_of zip = Zipc.fold (fun m acc -> Zipc.Member.path m :: acc) zip []

(* PyTorch nests everything under a single top-level directory ("<model>/" in a
   .pt2, "archive/" in a .pt). Find it from the first member that has one (a bare
   directory member, if any, has no '/'). *)
let compute_prefix zip =
  let rec first = function
    | [] -> ""
    | name :: rest -> (
        match String.index_opt name '/' with
        | Some i -> String.sub name 0 i
        | None -> first rest)
  in
  first (entries_of zip)

let pp_error ppf : error -> unit = function
  | `Zip_parse_failed msg -> Fmt.pf ppf "zip parse failed: %s" msg
  | `Zip_read_failed (name, message) ->
      Fmt.pf ppf "zip entry %S read failed: %s" name message
  | `Zip_missing_entry name -> Fmt.pf ppf "zip entry %S is missing" name

let of_string s =
  match Zipc.of_binary_string s with
  | Ok zip -> Core.return { zip; prefix = compute_prefix zip }
  | Error e -> Core.fail (`Zip_parse_failed e)

let entries t = entries_of t.zip
let prefix t = t.prefix

let read t name =
  match Zipc.find name t.zip with
  | None -> Core.return None
  | Some m -> (
      match Zipc.Member.kind m with
      | Zipc.Member.Dir -> Core.return None
      | Zipc.Member.File f -> (
          match Zipc.File.to_binary_string f with
          | Ok s -> Core.return (Some s)
          | Error e -> Core.fail (`Zip_read_failed (name, e))))

let read_required t name =
  let open Core.Syntax in
  let* data = read t name in
  match data with
  | Some data -> Core.return data
  | None -> Core.fail (`Zip_missing_entry name)

let read_rel t rel = read t (t.prefix ^ "/" ^ rel)
let read_rel_required t rel = read_required t (t.prefix ^ "/" ^ rel)
