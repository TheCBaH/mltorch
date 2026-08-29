(* See role.mli. A bare [type t] in a structure declares an abstract,
   uninhabited type -- the same idiom [Dim]'s phantom tags use. *)

module Position = struct
  type t
end

module Delta = struct
  type t
end
