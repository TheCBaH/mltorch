(* The failure of one validated-text constructor, shared by every role so a
   decoder can carry it without knowing which module produced it. *)

type reason =
  | Empty
  | Too_long
  | Invalid_character  (** A control character, or whitespace at either end. *)

type t = { role : string; reason : reason }

let pp_reason ppf = function
  | Empty -> Fmt.string ppf "empty"
  | Too_long -> Fmt.string ppf "too long"
  | Invalid_character -> Fmt.string ppf "invalid character"

let pp ppf { role; reason } = Fmt.pf ppf "%s: %a" role pp_reason reason
