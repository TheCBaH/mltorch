(* See [Core.Dim] for the roles and their operations; this module is the same
   interface, with types equal to it, plus the wire codec. *)

include module type of struct
  include Core.Dim
end

val extent_jsont : extent t Jsont.t
val fence_jsont : fence t Jsont.t
val index_jsont : index t Jsont.t
