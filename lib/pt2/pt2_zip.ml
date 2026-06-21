(* Thin wrapper over the [zipc] archive reader, exposing the lookups the pt2
   loader needs. PyTorch writes every entry STORED (uncompressed), but zipc also
   handles Deflate, so this works regardless. Archives are held in memory as a
   string. The single top-level directory all entries nest under is computed
   once at open time (see [prefix]). *)

type t = { zip : Zipc.t; prefix : string }

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

let of_string s =
  match Zipc.of_binary_string s with
  | Ok zip -> { zip; prefix = compute_prefix zip }
  | Error e -> failwith ("Pt2_zip: " ^ e)

let entries t = entries_of t.zip
let prefix t = t.prefix

let read t name =
  match Zipc.find name t.zip with
  | None -> None
  | Some m -> (
      match Zipc.Member.kind m with
      | Zipc.Member.Dir -> None
      | Zipc.Member.File f -> (
          match Zipc.File.to_binary_string f with
          | Ok s -> Some s
          | Error e -> failwith ("Pt2_zip: " ^ name ^ ": " ^ e)))

let read_exn t name =
  match read t name with
  | Some s -> s
  | None -> failwith ("Pt2_zip: no entry " ^ name)

let read_rel t rel = read t (t.prefix ^ "/" ^ rel)
let read_rel_exn t rel = read_exn t (t.prefix ^ "/" ^ rel)
