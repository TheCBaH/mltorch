(* Error payload types and printer for [Me_limits], split from me_limits.ml. No dependency on the domain vocabulary, the checked
   arithmetic, or the profile machinery, so this compiles first. Not
   published under a [.mli] of its own -- me_limits.mli documents the
   public contract and me_limits.ml re-exports every item here by
   manifest type/value alias. *)

(* Ceilings and profiles for the Model Explorer export path. See the .mli for
   what the three layers are and why they are three. *)

module Invalid = struct
  type t = { name : string; value : int64 }
end

type live_error = [ `Live_overflow of string ]

type error =
  [ `Invalid_limit of Invalid.t
  | `Invalid_limits of Invalid.t list
  | live_error ]

let pp_invalid fmt { Invalid.name; value } = Fmt.pf fmt "%s = %Ld" name value

let pp_error fmt : [< error ] -> unit = function
  | `Invalid_limit i -> Fmt.pf fmt "invalid limit %a" pp_invalid i
  | `Invalid_limits is ->
      Fmt.pf fmt "@[<hov 2>%d invalid limits:@ %a@]" (List.length is)
        (Fmt.list ~sep:(Fmt.any ",@ ") pp_invalid)
        is
  | `Live_overflow phase -> Fmt.pf fmt "live allocation overflow in %s" phase
