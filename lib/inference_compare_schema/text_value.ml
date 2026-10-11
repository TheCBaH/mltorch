module type S = sig
  type t

  val of_string : string -> (t, Text_error.t) Err.t
  val to_string : t -> string
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

module type ROLE = sig
  val role : string
end

(* Bytes, not characters: the bound exists to keep identifiers small enough to
   index and print, and UTF-8 text is admitted as-is. *)
let max_length = 4096
let is_control c = Char.code c < 0x20 || Char.code c = 0x7f
let is_blank c = c = ' ' || c = '\t' || c = '\n' || c = '\r'

let check_identifier s : Text_error.reason option =
  let n = String.length s in
  if n = 0 then Some Empty
  else if n > max_length then Some Too_long
  else if is_blank s.[0] || is_blank s.[n - 1] then Some Invalid_character
  else if String.exists is_control s then Some Invalid_character
  else None

let check_prose s : Text_error.reason option =
  if String.length s > max_length then Some Too_long
  else if String.exists (fun c -> is_control c && c <> '\n') s then
    Some Invalid_character
  else None

module Make
    (C : sig
      val check : string -> Text_error.reason option
    end)
    (R : ROLE) : S = struct
  type t = string

  let of_string s =
    match C.check s with
    | None -> Ok s
    | Some reason ->
        Err.fail ~pp_error:Text_error.pp { Text_error.role = R.role; reason }

  let to_string s = s
  let compare = String.compare
  let equal = String.equal
  let pp = Fmt.string
end

module Make_identifier (R : ROLE) () : S =
  Make
    (struct
      let check = check_identifier
    end)
    (R)

module Make_prose (R : ROLE) () : S =
  Make
    (struct
      let check = check_prose
    end)
    (R)
