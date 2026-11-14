(* Scalar-local identities share Builder's ordinal supply with reducer
   identities, but have a distinct type and namespace. *)

include
  Core.Tagged_int.Make
    (struct
      let prefix = "#"
    end)
    ()

let hash n = to_int n
