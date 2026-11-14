(* Reducer identities are library-private integers.  The public [Expr]
   signature keeps the representation abstract. *)

include
  Core.Tagged_int.Make
    (struct
      let prefix = "#"
    end)
    ()

let hash n = to_int n
