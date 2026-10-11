(* How the Native direct evaluator holds a body's tensors: the configurations
   inference itself runs under. [Default] keeps every edge and allocates freely;
   [Release] frees an edge after its last reader; the arena configurations place
   the intermediates in pools; the role ones place every block of a run by role.
   [scheduled] first reorders the body to lower its live payload. *)

type storage =
  | Arena_best_effort
  | Arena_required of int64
  | Default
  | Release
  | Separate
  | Shared

type t = { storage : storage; scheduled : bool }

let default = { storage = Default; scheduled = false }
let is_default t = t.storage = Default && not t.scheduled

(* One word of a variant's configuration, [None] when it is not a storage word. *)
let word_of_string s =
  match s with
  | "arena" -> Some (`Storage Arena_best_effort)
  | "release" -> Some (`Storage Release)
  | "scheduled" -> Some `Scheduled
  | "separate" -> Some (`Storage Separate)
  | "shared" -> Some (`Storage Shared)
  | _ -> (
      match String.split_on_char '=' s with
      | [ "arena"; n ] -> (
          match Int64.of_string_opt n with
          | Some b when Int64.compare b 0L >= 0 ->
              Some (`Storage (Arena_required b))
          | _ -> None)
      | _ -> None)

(* The words, in the order [to_words] prints them. *)
let ( let* ) = Result.bind

let of_words words =
  let rec go t = function
    | [] -> Ok t
    | `Scheduled :: rest -> go { t with scheduled = true } rest
    | `Storage s :: rest ->
        if t.storage <> Default then Error "more than one storage"
        else go { t with storage = s } rest
  in
  let* t = go default words in
  if t.scheduled && t.storage = Default then
    Error "scheduled needs release, arena, separate or shared"
  else Ok t

let to_words t =
  (match t.storage with
    | Default -> []
    | Arena_best_effort -> [ "arena" ]
    | Arena_required b -> [ "arena=" ^ Int64.to_string b ]
    | Release -> [ "release" ]
    | Separate -> [ "separate" ]
    | Shared -> [ "shared" ])
  @ if t.scheduled then [ "scheduled" ] else []
